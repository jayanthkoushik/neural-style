#!/usr/bin/env bash

set -e

VENV_DIR="env"
if ! [ -d ${VENV_DIR} ]; then
    python2 -m virtualenv ${VENV_DIR}
fi
source ${VENV_DIR}/bin/activate

pip install --upgrade setuptools
pip install --upgrade pip
pip install --upgrade six
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade cython
pip install --upgrade h5py
pip install --upgrade matplotlib
pip install --upgrade git+https://github.com/Theano/Theano.git
pip install --upgrade git+https://github.com/fchollet/keras.git
