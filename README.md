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
dependencies, to use them add your handler file to the zip, and add the `lib` directory so it can be used for shared libs. 

This repository also contains a file called `process.py` which imports these packages and carries out a simple task with astropy

```python
import os
import subprocess
import uuid

libdir = os.path.join(os.getcwd(), 'lib')

FITS_LOCATION = "https://hla.stsci.edu/cgi-bin/ecfproxy?file_id=hag_j004524.96+403851.8_j8hpdcaoq_v01.drizzle.fits"

import warnings
from astropy.utils.data import CacheMissingWarning
warnings.simplefilter('ignore', CacheMissingWarning)

from astropy.io import fits
import numpy as np
from photutils import datasets
from skimage import data

def do_science(fits_location):
    hdul = fits.open(fits_location)
    print(hdul.info())

def handler(event, context):
    do_science(FITS_LOCATION)

if __name__ == "__main__":
    handler('', '')

```

## Extra Packages

To add extra packages to the build, add them to the `requirements.txt` file alongside the `build.sh` in this repo. All packages listed there will be installed in addition to those already described in [`build.sh`](https://github.com/arfon/astropy-build-lambda/blob/81b12db7d29c5fc90bc1f4c0f6773eb6a38aa24a/build.sh#L19-L21)

## Sizing and Future Work

In its current form, this set of packages weighs in at 110MB and could probably be reduced further by:

1. Pre-compiling all .pyc files and deleting their source
1. Removing test files
1. Removing documentation

According to [this article](https://docs.aws.amazon.com/lambda/latest/dg/limits.html) the size limit for a zipped Lambda package (the `venv.zip` file) is 50MB, however, reading around it seems like Lambda is tolerant of significantly larger packages when the zipped package is posted to S3.
 
