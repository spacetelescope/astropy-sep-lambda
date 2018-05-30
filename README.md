# astropy-sep-lambda

> Build the astropy, sep, numpy, and scipy packages and strip them down to run in Lambda

This repo contains a `build.sh` script that's intended to be run in an Amazon Linux docker container, and build astropy, sep, numpy, and scipy for use in AWS Lambda. For more info about how the script works, and how to use it, see the blob post by `ryansb` [on deploying sklearn to Lambda](https://serverlesscode.com/post/scikitlearn-with-amazon-linux-container/).

This repository is a fork of the excellent work by `ryanb` in https://github.com/ryansb/sklearn-build-lambda .

To build the zipfile, pull the Amazon Linux image and run the build script in it.

```
$ docker pull amazonlinux:2017.09
$ docker run -v $(pwd):/outputs -it amazonlinux:2017.09 /bin/bash /outputs/build.sh
```

That will make a file called `venv.zip` in the local directory that's around 50MB.

Once you run this, you'll have a zipfile containing astropy, sep, numpy, scipy and their dependencies. This repository also contains a file called `process.py` which imports these packages and detects sources with [SEP package](http://sep.readthedocs.io/en/v1.0.x/) (based on Source Extractor).

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

...
...
...
```

## Extra Packages

To add extra packages to the build, add them to the `requirements.txt` file alongside the `build.sh` in this repo. All packages listed there will be installed in addition to those already described in [`build.sh`](https://github.com/spacetelescope/astropy-sep-lambda/blob/f3f34a6c1b8e6bd451de5c8ff6dc1f5e5cd193f8/build.sh#L18-L20)

## Sizing and Future Work

In its current form, this set of packages weighs in at 50MB and could probably be reduced further by:

1. Pre-compiling all .pyc files and deleting their source
1. Removing test files
1. Removing documentation

According to [this article](https://docs.aws.amazon.com/lambda/latest/dg/limits.html) the size limit for a zipped Lambda package (the `venv.zip` file) is 50MB, however, reading around it seems like Lambda is tolerant of significantly larger packages when the zipped package is posted to S3.
 
