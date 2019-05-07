#!/bin/bash

DOCKER_IMAGE=quay.io/pypa/manylinux1_x86_64
PLAT=manylinux1_x86_64

sudo docker run --rm -e PLAT=$PLAT -v `pwd`:/io $DOCKER_IMAGE $PRE_CMD /io/scripts/build-wheels.sh
