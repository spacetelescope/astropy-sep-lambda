import glob
import logging
import os
libdir = os.path.join(os.getcwd(), 'lib')
import warnings

from astropy.convolution import kernels
from astropy.stats import gaussian_sigma_to_fwhm
import astropy.io.fits as fits
from astropy.table import Table, Column
from astropy import units as u
from astropy import wcs
from astropy.utils.data import CacheMissingWarning
warnings.simplefilter('ignore', CacheMissingWarning)
import boto3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sep

GAUSS_3_7x7 = np.array(
[[ 0.004963,  0.021388,  0.051328,  0.068707,  0.051328,  0.021388,  0.004963],
 [ 0.021388,  0.092163,  0.221178,  0.296069,  0.221178,  0.092163,  0.021388],
 [ 0.051328,  0.221178,  0.530797,  0.710525,  0.530797,  0.221178,  0.051328],
 [ 0.068707,  0.296069,  0.710525,  0.951108,  0.710525,  0.296069,  0.068707],
 [ 0.051328,  0.221178,  0.530797,  0.710525,  0.530797,  0.221178,  0.051328],
 [ 0.021388,  0.092163,  0.221178,  0.296069,  0.221178,  0.092163,  0.021388],
[ 0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963]])


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    )
LOG = logging.getLogger('fits_handler')


def detect_with_sep(
        event,
        detect_thresh=2.,
        npixels=8,
        grow_seg=5,
        gauss_fwhm=2.,
        gsize=3,
        im_wcs=None,
        fname=None
):
    """

    Parameters
    ----------
    event : dict
        dict containing the data passed to the lamba function
    detect_thresh: int,
        detection threshold to use for sextractor
    npixels: int,
        minimum number of pixels comprising an object
    grow_seg: int,

    gauss_fwhm: float,
        FWHM of the kernel to use for filtering prior to source finding
    gsize: float

    im_wcs: astropy.wcs.WCS
        WCS object defining the coordinate system of the observation
    fname: str


    Returns
    -------

    """

    drz_file = event['fits_s3_key']
    drz_file_bucket = event['fits_s3_bucket']

    if fname is None:
        fname = drz_file.split('/')[-1]

    s3 = boto3.resource('s3')
    bkt = s3.Bucket(drz_file_bucket)
    bkt.download_file(
        drz_file, f"/tmp/{fname}",
        ExtraArgs={"RequestPayer": "requester"}
    )

    im = fits.open(f"/tmp/{fname}")
    if im_wcs is None:
        im_wcs = wcs.WCS(im[1].header, relax=True)

    data = im[1].data.byteswap().newbyteorder()
    wht_data = im[2].data.byteswap().newbyteorder()
    data_mask = np.cast[data.dtype](data == 0)

    ## Get AB zeropoint
    try:
        photfnu = im[0].header['PHOTFNU']
    except KeyError as e:
        LOG.warning(e)
        ZP=None
    else:
        ZP= -2.5 * np.log10(photfnu) + 8.90

    try:
        photflam = im[0].header['PHOTFLAM']
    except KeyError as e:
        LOG.warning(e)
        ZP=None
    else:
        ZP = -2.5*np.log10(photflam) - 21.10 - \
             5*np.log10(im[0].header['PHOTPLAM']) + 18.6921

    if ZP is None:
        msg = (
            "Whoops! No zeropoint information found in primary header, "
            f"skipping file {fname}"
        )
        LOG.warning(msg)

    # Scale fluxes to mico-Jy
    uJy_to_dn = 1/(3631*1e6*10**(-0.4*ZP))

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

    # Generate a kernel to use for filtering
    gaussian_kernel = kernels.Gaussian2DKernel(
        x_stddev=gauss_fwhm/gaussian_sigma_to_fwhm,
        y_stddev=gauss_fwhm/gaussian_sigma_to_fwhm,
        x_size=7,
        y_size=7
    )
    # Normalize the kernel
    gaussian_kernel.normalize()

    # Package the inputs for sextractor
    inputs = {
        'err': err,
        'mask': mask,
        'filter_kernel': gaussian_kernel.array,
        'filter_type': 'conv',
        'minarea': npixels,
        'deblend_nthresh': 32,
        'deblend_constrast': 0.005,
        'clean': True,
        'clean_param': 1,
        'segmentation_map': False
    }

    objects = sep.extract(
        data - bkg_data,
        detect_thresh,
        **inputs
        )

    catalog = Table(objects)

    # add things to catalog
    autoparams=[2.5, 3.5]
    catalog['number'] = np.arange(len(catalog), dtype=np.int32)+1
    catalog['theta'] = np.clip(catalog['theta'], -np.pi/2, np.pi/2)

    # filter out any NaNs
    for c in ['a','b','x','y','theta']:
        catalog = catalog[np.isfinite(catalog[c])]

    catalog['ra'], catalog['dec'] = im_wcs.all_pix2world(
        catalog['x'],
        catalog['y'],
        1
    )

    catalog['ra'].unit = u.deg
    catalog['dec'].unit = u.deg
    catalog['x_world'], catalog['y_world'] = catalog['ra'], catalog['dec']

    kronrad, krflag = sep.kron_radius(
        data - bkg_data,
        catalog['x'],
        catalog['y'],
        catalog['a'],
        catalog['b'],
        catalog['theta'],
        6.0
    )

    kronrad *= autoparams[0]
    kronrad[~np.isfinite(kronrad)] = autoparams[1]
    kronrad = np.maximum(kronrad, autoparams[1])

    kron_out = sep.sum_ellipse(
        data - bkg_data,
        catalog['x'],
        catalog['y'],
        catalog['a'],
        catalog['b'],
        catalog['theta'],
        kronrad,
        subpix=5,
        err=err
    )

    kron_flux, kron_fluxerr, kron_flag = kron_out
    kron_flux_flag = kron_flag

    catalog['mag_auto_raw'] = ZP - 2.5*np.log10(kron_flux)
    catalog['magerr_auto_raw'] = 2.5/np.log(10)*kron_fluxerr/kron_flux

    catalog['mag_auto'] = catalog['mag_auto_raw']*1.
    catalog['magerr_auto'] = catalog['magerr_auto_raw']*1.

    catalog['kron_radius'] = kronrad*u.pixel
    catalog['kron_flag'] = krflag
    catalog['kron_flux_flag'] = kron_flux_flag
    
    # Make a plot
    im_data = im[1].data
    im_shape = im_data.shape
    im_data[np.isnan(im_data)] = 0.0
    
    # Trim the top and bottom 1 percent of pixel values
    top = np.percentile(im_data, 99)
    im_data[im_data > top] = top
    bottom = np.percentile(im_data, 1)
    im_data[im_data < bottom] = bottom

    # Scale the data.
    im_data = im_data - im_data.min()
    im_data = (im_data / im_data.max()) * 255.
    im_data = np.uint8(im_data)

    f, (ax) = plt.subplots(1,1, sharex=True)
    f.set_figheight(12)
    f.set_figwidth(12)
    ax.imshow(im_data, cmap="Greys", clim=(0, 255), origin='lower')
    ax.plot(
        catalog['x'],
        catalog['y'],
        'o',
        markeredgewidth=1,
        markeredgecolor='red',
        markerfacecolor='None'
    )
    ax.set_xlim([-0.05*im_shape[1],1.05*im_shape[1]])
    ax.set_ylim([-0.05*im_shape[0],1.05*im_shape[0]])

    basename = fname.split('_')[0]
    f.savefig(f"/tmp/{basename}.png")

    # Write the catalog to local disk
    catalog.write(f"/tmp/{basename}.catalog.fits", format='fits')

    # Write out to S3
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(
        f"/tmp/{basename}.catalog.fits",
        event['s3_output_bucket'],
        f"{basename}/{basename}.catalog.fits"
    )
    s3.meta.client.upload_file(
        f"/tmp/{fname}.png",
        event['s3_output_bucket'],
        f"PNG/{fname}.png"
    )

def clean_up(dirname="/tmp"):
    """ Delete all files created and stored in the lambda tmp directory

    Lambda can only store so much data in tmp and when many function
    invocations occur in a short period of time, memory will sometimes persist
    innovcations and so to avoid filling up the maximum allowable memory for
    a function, we need to delete the files we stored

    Parameters
    ----------
    dirname : str
        path to /tmp on lambda function

    Returns
    -------

    """
    flist = glob.glob(f"{dirname}/*")
    for f in flist:
        try:
            os.remove(f)
        except OSError as e:
            LOG.warning(e)
        else:
            LOG.info('removed')


def handler(event, context):
    LOG.info(event['s3_output_bucket'])
    LOG.info(event['fits_s3_key'])
    LOG.info(event['fits_s3_bucket'])
    detect_with_sep(event)
    clean_up()

