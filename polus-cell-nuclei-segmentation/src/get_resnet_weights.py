from keras.utils import get_file
import os

if __name__ == '__main__':

    cwd = os.getcwd()

    # weights for resnet 101
    filename = 'ResNet-101-model.keras.h5'
    resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(filename)
    checksum_101 = '05dc86924389e5b401a9ea0348a3213c'
    filepath = os.path.join(cwd, 'dsb2018_topcoders/selim/models/{}'.format(filename))

    get_file(
        filepath,
        resnet_resource,
        cache_subdir=os.path.join(cwd, 'dsb2018_topcoders/selim/models/'),
        md5_hash=checksum_101
    )

    # weights for resnet 152
    filename = 'ResNet-152-model.keras.h5'
    resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(filename)
    checksum_152 = '6ee11ef2b135592f8031058820bb9e71'
    filepath = os.path.join(cwd, 'dsb2018_topcoders/selim/models/{}'.format(filename))

    get_file(
        filepath,
        resnet_resource,
        cache_subdir=os.path.join(cwd, 'dsb2018_topcoders/selim/models/'),
        md5_hash=checksum_152
    )

    