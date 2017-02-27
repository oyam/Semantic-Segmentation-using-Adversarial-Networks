if [ ! -e pretrained_model ]; then
    mkdir pretrained_model
fi

if [ ! -e pretrained_model/VGG_ILSVRC_16_layers.caffemodel ]; then
    wget -P pretrained_model http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
fi

if [ ! -e pretrained_model/vgg16.npz ]; then
    python setup/create_vgg_chainermodel.py
fi

if [ ! -e ~/.chainer/dataset/pascal ]; then
    python setup/download_dataset.py
fi
