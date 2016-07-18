#!/usr/bin/env bash

set -e

mkdir -p data/vgg19
cd data/vgg19
wget -O vgg19.prototxt https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt
wget -O vgg19_normalized.caffemodel http://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel
