#!/bin/sh

if [ "$HOSTNAME" == login1 ] || [ "$HOSTNAME" == login2 ]; then

qsub /projects/huanlab/wei/submit.sh

else
    echo 'This script may only be run from an entry point to the cluster(i.e. login1 or login2)'
fi

