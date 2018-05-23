# astropy-build-lambda

> Build the astropy, photutils, skimage, numpy, scipy, scikit-learn packages and strip them down to run in Lambda

This repo contains a `build.sh` script that's intended to be run in an Amazon Linux docker container, and build astropy, photutils, skimage, numpy, scipy, and scikitlearn for use in AWS Lambda. For more info about how the script works, and how to use it, see the blob post by `ryansb` [on deploying sklearn to Lambda](https://serverlesscode.com/post/scikitlearn-with-amazon-linux-container/).

This repository is a fork of the excellent work by `ryanb` in https://github.com/ryansb/sklearn-build-lambda .

To build the zipfile, pull the Amazon Linux image and run the build script in it.

```
$ docker pull amazonlinux:2017.09
$ docker run -v $(pwd):/outputs -it amazonlinux:2017.09 /bin/bash /outputs/build.sh
```

That will make a file called `venv.zip` in the local directory that's around 100MB.

Once you run this, you'll have a zipfile containing astropy, photutils, skimage, numpy, scipy, scikit-learn and their
dependencies. This repository also contains a file called `process.py` which imports these packages and ~~carries out a simple task with astropy.~~ does some crazy rad stuff with Astropy, Photutils, and Pan-STARRS.

```python
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
    
 ...
 ...
 ...
```

## Extra Packages

To add extra packages to the build, add them to the `requirements.txt` file alongside the `build.sh` in this repo. All packages listed there will be installed in addition to those already described in [`build.sh`](https://github.com/arfon/astropy-build-lambda/blob/81b12db7d29c5fc90bc1f4c0f6773eb6a38aa24a/build.sh#L19-L21)

## Sizing and Future Work

In its current form, this set of packages weighs in at 110MB and could probably be reduced further by:

1. Pre-compiling all .pyc files and deleting their source
1. Removing test files
1. Removing documentation

According to [this article](https://docs.aws.amazon.com/lambda/latest/dg/limits.html) the size limit for a zipped Lambda package (the `venv.zip` file) is 50MB, however, reading around it seems like Lambda is tolerant of significantly larger packages when the zipped package is posted to S3.
 
