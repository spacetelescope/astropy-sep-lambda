import os
import subprocess
import uuid

libdir = os.path.join(os.getcwd(), 'lib')

import warnings
from astropy.utils.data import CacheMissingWarning
warnings.simplefilter('ignore', CacheMissingWarning)

import boto3
import glob
import numpy as np
import sep

from astropy import wcs
from astropy import units as u
import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma, median_absolute_deviation

GAUSS_3_7x7 = np.array(
[[ 0.004963,  0.021388,  0.051328,  0.068707,  0.051328,  0.021388,  0.004963],
 [ 0.021388,  0.092163,  0.221178,  0.296069,  0.221178,  0.092163,  0.021388],
 [ 0.051328,  0.221178,  0.530797,  0.710525,  0.530797,  0.221178,  0.051328],
 [ 0.068707,  0.296069,  0.710525,  0.951108,  0.710525,  0.296069,  0.068707],
 [ 0.051328,  0.221178,  0.530797,  0.710525,  0.530797,  0.221178,  0.051328],
 [ 0.021388,  0.092163,  0.221178,  0.296069,  0.221178,  0.092163,  0.021388],
[ 0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963]])

def detect_with_sep(event, detect_thresh=2., npixels=8, grow_seg=5,
                          gauss_fwhm=2., gsize=3, im_wcs=None, root='mycat'):

    drz_file = event['fits_s3_key']
    drz_file_bucket = event['fits_s3_bucket']
    root = drz_file.split('/')[-1].split('_')[0]

    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(drz_file_bucket)
    bkt.download_file(drz_file, '/tmp/{0}'.format(root), ExtraArgs={"RequestPayer": "requester"})

    im = fits.open('/tmp/{0}'.format(root))
    im_wcs = wcs.WCS(im[1].header, relax=True)

    data = im[1].data.byteswap().newbyteorder()
    wht_data = im[2].data.byteswap().newbyteorder()
    data_mask = np.cast[data.dtype](data == 0)

    ## Get AB zeropoint
    if 'PHOTFNU' in im[0].header:
        ZP = -2.5*np.log10(im[0].header['PHOTFNU'])+8.90
    elif 'PHOTFLAM' in im[0].header:
        ZP = (-2.5*np.log10(im[0].header['PHOTFLAM']) - 21.10 -
              5*np.log10(im[0].header['PHOTPLAM']) + 18.6921)
    else:
        print('Couldn\'t find PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25')
        return None

    # Scale fluxes to mico-Jy
    uJy_to_dn = 1/(3631*1e6*10**(-0.4*ZP))

    err=1/np.sqrt(wht_data)


    # set up the error array
    err = 1/np.sqrt(wht_data)
    err[~np.isfinite(err)] = 0
    mask = (err == 0)

    # get the background
    bkg = sep.Background(data, mask=mask, bw=32, bh=32, fw=3, fh=3)
    bkg_data = bkg.back()

    ratio = bkg.rms()/err
    err_scale = np.median(ratio[(~mask) & np.isfinite(ratio)])

    err *= err_scale

    objects = sep.extract(data - bkg_data, detect_thresh, err=err,
                          mask=mask, minarea=14,
                          filter_kernel=GAUSS_3_7x7,
                          filter_type='conv', deblend_nthresh=32,
                          deblend_cont=0.005, clean=True, clean_param=1.,
                          segmentation_map=False)

    catalog = Table(objects)

    # add things to catalog

    autoparams=[2.5, 3.5]

    catalog['number'] = np.arange(len(catalog), dtype=np.int32)+1
    catalog['theta'] = np.clip(catalog['theta'], -np.pi/2, np.pi/2)
    for c in ['a','b','x','y','theta']:
        catalog = catalog[np.isfinite(catalog[c])]

    catalog['ra'], catalog['dec'] = im_wcs.all_pix2world(catalog['x'], catalog['y'], 1)
    catalog['ra'].unit = u.deg
    catalog['dec'].unit = u.deg
    catalog['x_world'], catalog['y_world'] = catalog['ra'], catalog['dec']

    kronrad, krflag = sep.kron_radius(data - bkg_data,
                                       catalog['x'], catalog['y'],
                                       catalog['a'], catalog['b'], catalog['theta'], 6.0)

    kronrad *= autoparams[0]
    kronrad[~np.isfinite(kronrad)] = autoparams[1]
    kronrad = np.maximum(kronrad, autoparams[1])

    kron_out = sep.sum_ellipse(data - bkg_data,
                                catalog['x'], catalog['y'],
                                catalog['a'], catalog['b'],
                                catalog['theta'],
                                kronrad, subpix=5, err=err)

    kron_flux, kron_fluxerr, kron_flag = kron_out
    kron_flux_flag = kron_flag

    catalog['mag_auto_raw'] = ZP - 2.5*np.log10(kron_flux)
    catalog['magerr_auto_raw'] = 2.5/np.log(10)*kron_fluxerr/kron_flux

    catalog['mag_auto'] = catalog['mag_auto_raw']*1.
    catalog['magerr_auto'] = catalog['magerr_auto_raw']*1.

    catalog['kron_radius'] = kronrad*u.pixel
    catalog['kron_flag'] = krflag
    catalog['kron_flux_flag'] = kron_flux_flag

    # Write the catalog to local disk
    catalog.write('/tmp/{0}.catalog.fits'.format(root), format='fits')

    # Write out to S3
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file('/tmp/{0}.catalog.fits'.format(root), event['s3_output_bucket'], '{0}/{1}.catalog.fits'.format(root, root))

def handler(event, context):
    print event['s3_output_bucket']
    print event['fits_s3_key']
    print event['fits_s3_bucket']
    detect_with_sep(event)

if __name__ == "__main__":
    handler('', '')
