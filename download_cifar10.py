import sys
import os
from globals import ROOT_DIR
from data_io import ImageDataIO

set_name = sys.argv[1]

if set_name=='test':
    print("downloading cifar10 ", set_name)
    x, y = ImageDataIO.download_cifar10(which_set='test')
    ImageDataIO.save_data(x, y, filename_npz=os.path.join(ROOT_DIR, 'data/cifar10-{}.npz'.format(set_name)))

if set_name=='valid':
    print("downloading cifar10 ", set_name)
    x, y = ImageDataIO.download_cifar10(which_set='val')
    ImageDataIO.save_data(x, y, filename_npz=os.path.join(ROOT_DIR, 'data/cifar10-{}.npz'.format(set_name)))

if set_name=='train':
    print("downloading cifar10 ", set_name)
    x, y = ImageDataIO.download_cifar10(which_set='train')
    ImageDataIO.save_data(x, y, filename_npz=os.path.join(ROOT_DIR, 'data/cifar10-{}.npz'.format(set_name)))