import sys
import os

import adversarial_sampling_experiments
# from adversarial_sampling_experiments.globals import ROOT_DIR
# from adversarial_sampling_experiments.data_io import ImageDataIO

set_name = sys.argv[1]

# names = []
# if set_name== 'valid':
#     names.append('valid')
# if set_name== 'train':
#     names.append('train')
# if set_name== 'test':
#     names.append('test')
# if set_name== 'all':
#     names = ['train','valid','test']
#
# if set_name== 'all':
#     for name in names:
#         print("downloading cifar10 ",name)
#         x, y = ImageDataIO.download_cifar10(which_set=name)
#         ImageDataIO.save_data(x,y,filename_npz=os.path.join(ROOT_DIR,'data/cifar10-{}.npz'.format(name)))
