#!/bin/bash

# Usage: ./download.sh
# note:  This file downloads the pre-generated random state files in the cache directory.
#        The main script should be able to regenerate these files without any download.
#        However, we provide them for maximum reproducibility and possibly a faster setup.

SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPTDIR

# importing the gdluntar helper function from the utils directory
source ../utils/bashfuncs.sh

FILEID="1w9_YOLgIVhtN8YwdiyAkdLLeIVB5UVuU"
FILENAME="cache.tar"
GDRIVEURL="https://drive.google.com/file/d/1w9_YOLgIVhtN8YwdiyAkdLLeIVB5UVuU/view?usp=sharing"
PTHMD5FILE="cache.md5"
REMOVETARAFTERDL="1"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL}
