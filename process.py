import os
import subprocess
import uuid

libdir = os.path.join(os.getcwd(), 'lib')

import warnings
from astropy.utils.data import CacheMissingWarning
warnings.simplefilter('ignore', CacheMissingWarning)

# should be installed by default?
import boto3
import glob
import numpy as np
import httplib as httplib
from urllib import urlencode
from urllib import urlopen

import scipy.ndimage as nd
from scipy.spatial import cKDTree as KDT

from photutils import detect_threshold, detect_sources
from photutils import source_properties, properties_table

import skimage.transform
from skimage.measure import ransac

from astropy import wcs
import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma, median_absolute_deviation
from astropy.convolution import Gaussian2DKernel

def xymatch(x1, y1, x2, y2, tol=None, nnearest=1):
    """Fast cross-matching of xy coordinates: from https://gist.github.com/eteq/4599814"""
    x1 = np.array(x1, copy=False)
    y1 = np.array(y1, copy=False)
    x2 = np.array(x2, copy=False)
    y2 = np.array(y2, copy=False)

    if x1.shape != y1.shape:
        raise ValueError('x1 and y1 do not match!')
    if x2.shape != y2.shape:
        raise ValueError('x2 and y2 do not match!')

    # this is equivalent to, but faster than just doing np.array([x1, y1])
    coords1 = np.empty((x1.size, 2))
    coords1[:, 0] = x1
    coords1[:, 1] = y1

    # this is equivalent to, but faster than just doing np.array([x2, y2])
    coords2 = np.empty((x2.size, 2))
    coords2[:, 0] = x2
    coords2[:, 1] = y2

    kdt = KDT(coords2)
    if nnearest == 1:
        ds,idxs2 = kdt.query(coords1)
    elif nnearest > 1:
        retval = kdt.query(coords1, nnearest)
        ds = retval[0]
        idxs2 = retval[1][:, -1]
    else:
        raise ValueError('invalid nnearest ' + str(nnearest))

    idxs1 = np.arange(x1.size)

    if tol is not None:
        msk = ds < tol
        idxs1 = idxs1[msk]
        idxs2 = idxs2[msk]
        ds = ds[msk]

    return idxs1, idxs2, ds

def get_panstarrs_catalog(ra=0., dec=0., radius=3):
    """Download PANSTARRS data and return an astropy table"""

    columns='objName,objID,raStack,decStack,raStackErr,decStackErr,rMeanKronMag,rMeanKronMagErr,iMeanKronMag,iMeanKronMagErr'
    max_records=10000

    query_url = "http://archive.stsci.edu/panstarrs/search.php?RA={ra}&DEC={dec}&radius={radius}&max_records={max_records}&outputformat=CSV&action=Search&coordformat=dec&selectedColumnsCsv={columns}&raStack%3E=0".format(ra=ra, dec=dec, radius=radius, max_records=int(max_records), columns=columns)
    query = urlopen(query_url)
    lines = [bytes(columns+'\n')]
    lines.extend(query.readlines()[2:])

    # Only /tmp is writable with Lambda (up to 512MB data)
    csv_file = '/tmp/tmp.csv'
    fp = open(csv_file,'wb')
    fp.writelines(lines)
    fp.close()

    table = Table.read(csv_file)
    clip = (table['rMeanKronMag'] > 0) & (table['raStack'] > 0)
    table['ra'] = table['raStack']
    table['dec'] = table['decStack']
    return table[clip]


def detect_with_photutils(sci, err=None, detect_thresh=2., npixels=8, grow_seg=5,
                          gauss_fwhm=2., gsize=3, wcs=None, root='mycat'):
    """Detect sources in a drizzled image"""
    ### DQ masks
    mask = (sci == 0)

    ### Detection threshold
    threshold = (detect_thresh * err)*(~mask)
    threshold[mask] = np.median(threshold[~mask])

    sigma = gauss_fwhm * gaussian_fwhm_to_sigma    # FWHM = 2.
    kernel = Gaussian2DKernel(sigma, x_size=gsize, y_size=gsize)
    kernel.normalize()

    mean, median, std = sigma_clipped_stats(sci*(~mask), sigma=3.0, iters=5)
    #print(mean, median, std)

    ## Detect sources
    segm = detect_sources(sci*(~mask)-median, threshold, npixels=npixels,filter_kernel=kernel)
    grow = nd.maximum_filter(segm.array, grow_seg)
    seg = np.cast[np.float32](grow)

    props = source_properties(sci, segm, error=threshold/detect_thresh,
                              mask=mask, background=None, wcs=wcs)

    catalog = properties_table(props)

    return catalog

def transform_wcs(in_wcs, translation=[0.,0.], rotation=0., scale=1.):
    """Transform drizzled WCS with translation, rotation and scale."""
    out_wcs = in_wcs.deepcopy()
    out_wcs.wcs.crpix += np.array(translation)
    theta = -rotation
    _mat = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

    out_wcs.wcs.cd = np.dot(out_wcs.wcs.cd, _mat)/scale

    det = np.linalg.det(in_wcs.wcs.cd)
    out_wcs.pscale = np.sqrt(np.abs(det))*3600.
    out_wcs.wcs.crpix *= scale
    if hasattr(out_wcs, '_naxis1'):
        out_wcs._naxis1 = int(np.round(out_wcs._naxis1*scale))
        out_wcs._naxis2 = int(np.round(out_wcs._naxis2*scale))

    return out_wcs

def match_lists(xy1, xy2, tolerance=20.):
    """Match 2 lists of xy coorinates"""
    idx1, idx2, ds = xymatch(xy1[:,0], xy1[:,1], xy2[:,0], xy2[:,1], tol=tolerance, nnearest=1)

    transform = skimage.transform.SimilarityTransform
    tf = transform()
    tf.estimate(xy1[idx1,:], xy2[idx2,:])

    model, inliers = ransac((xy1[idx1,:], xy2[idx2,:]),transform, min_samples=3,
                            residual_threshold=2, max_trials=100)
    outliers = ~inliers

    return idx1, idx2, outliers, model

def align_drizzled_image(event, NITER=5, clip=20, log=True, outlier_threshold=5):
    drz_file = event['fits_s3_key']
    drz_file_bucket = event['fits_s3_bucket']

    root = drz_file.split('/')[-1].split('_')[0]

    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(drz_file_bucket)
    bkt.download_file(fits_s3_key, '/tmp/{0}'.format(root), ExtraArgs={"RequestPayer": "requester"})

    drz_im = fits.open('/tmp/{0}'.format(root))
    sh = drz_im[1].data.shape

    drz_wcs = wcs.WCS(drz_im[1].header, relax=True)
    orig_wcs = drz_wcs.copy()

    ### catalog of image
    cat_drz = detect_with_photutils(drz_im[1].data, err=1/np.sqrt(drz_im[2].data), wcs=orig_wcs)

    ### catalog from PANSTARRS
    cat_panstarrs = get_panstarrs_catalog(ra=drz_im[0].header['RA_TARG'], dec=drz_im[0].header['DEC_TARG'], radius=3)
    rd_panstarrs = np.array([cat_panstarrs['ra'].data,cat_panstarrs['dec'].data]).T

    out_shift, out_rot, out_scale = np.zeros(2), 0., 1.

    NGOOD, rms = 0, 0

    for iter in range(NITER):
        xy_panstarrs = np.array(drz_wcs.all_world2pix(rd_panstarrs, 1))
        pix = np.cast[int](np.round(xy_panstarrs)).T

        ### Find objects that are within the image footprint
        okp = (pix[0,:] > 0) & (pix[1,:] > 0) & (pix[0,:] < sh[1]) & (pix[1,:] < sh[0])

        xy_panstarrs = xy_panstarrs[okp]
        xy_drz = np.array([cat_drz['xcentroid'].value,cat_drz['ycentroid'].value]).T

        toler=5
        titer=0
        while (titer < 3):
            try:
                results = match_lists(xy_panstarrs, xy_drz, tolerance=toler)
                ps_ix, drz_ix, outliers, tf = results
                break
            except:
                toler += 5
                titer += 1


        tf_out = tf(xy_panstarrs[ps_ix])
        dx = xy_drz[drz_ix] - tf_out
        rms = 1.48*median_absolute_deviation(np.sqrt((dx**2).sum(axis=1)))
        outliers = (np.sqrt((dx**2).sum(axis=1)) > 4*rms)

        if outliers.sum() > 0:
            results2 = match_lists(xy_panstarrs[ps_ix][~outliers], xy_drz[drz_ix][~outliers], tolerance=toler)
            ps_ix2, drz_ix2, outliers2, tf2 = results2

        shift = tf.translation
        NGOOD = (~outliers).sum()

        out_shift += tf.translation
        out_rot -= tf.rotation
        out_scale *= tf.scale
        drz_wcs = transform_wcs(drz_wcs, tf.translation, tf.rotation,
                                      tf.scale)

    print('shift1 shift2 rot scale rms ngood')
    print(out_shift[0], out_shift[1], out_rot/np.pi*180, out_scale, rms, NGOOD)

    fp = open('/tmp/{0}_wcs.log'.format(root), 'w')
    fp.write('# root xshift yshift rot scale rms N\n')
    fp.write('{0} {1:13.4f} {2:13.4f} {3:13.4f} {4:13.5f} {5:13.3f} {6:4d}\n'.format(root, shift[0], shift[1], out_rot/np.pi*180, out_scale, rms, NGOOD))
    fp.close()

    # Write out to S3

    s3 = boto3.resource('s3')
    s3.meta.client.upload_file('/tmp/{0}_wcs.log'.format(root), event['s3_output_bucket'], '{0}/{1}_wcs.log'.format(root, root))

    orig_hdul = fits.HDUList()
    hdu = drz_wcs.to_fits()[0]
    orig_hdul.append(hdu)
    orig_hdul.writeto('/tmp/{0}_wcs.fits'.format(root), overwrite=True)

    s3.meta.client.upload_file('/tmp/{0}_wcs.fits'.format(root), event['s3_output_bucket'], '{0}/{1}_wcs.fits'.format(root, root))


def handler(event, context):
    print event['s3_output_bucket']
    print event['fits_location']
    align_drizzled_image(event)

if __name__ == "__main__":
    handler('', '')
