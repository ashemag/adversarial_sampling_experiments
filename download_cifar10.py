import sys
import os

from globals import ROOT_DIR
from data_io import ImageDataIO

set_name = sys.argv[1]

names = []
if set_name== 'valid':
    names.append('valid')
if set_name== 'train':
    names.append('train')
if set_name== 'test':
    names.append('test')
if set_name== 'all':
    names = ['train','valid','test']

for name in names:
    print("downloading cifar10 ",name)
    x, y = ImageDataIO.download_cifar10(which_set=name)
    ImageDataIO.save_data(x,y,filename_npz=os.path.join(ROOT_DIR,'data/cifar10-{}.npz'.format(name)))

