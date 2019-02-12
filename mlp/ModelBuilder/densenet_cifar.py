import torch.nn as nn
import torch
import numpy as np
from torchvision.models.densenet import _DenseBlock
from torchvision.models.densenet import _Transition
from ModelBuilder.base import Network

class DenseNetCifar10(Network):
    '''
    implementation of DenseNet-BC on cifar10.
    - three DenseBlocks.
    - each DenseBlock has the same number of DenseLayers.

    DenseNet-BC for cifar10 architecture: authors of original paper in "implementation details" list following
    specifications:
    - network has 1 Conv layer, 3 DenseBlocks, 2 Transitions, Output (AvgPooling -> FC layer).
    - authors use 2 parameters to define a DenseNet: L, and k. L is total number of layers of a network, k is the
    "growth factor" (the number of kernels of a dense layer). authors experiment with several different settings
    such as L=100, and k=12.
    - the number of dense layers are the same in all dense blocks.

    remarks:
    - DenseNet-BC is combination of DenseNet-B and Densenet-C. DenseNet-B is a DenseNet with 1x1 CONV feature
    reduction in the DenseBlocks. DenseNet-C reduces feature maps of DenseBlocks (compression factor theta).

    useful sources:
    - densnet paper: https://arxiv.org/pdf/1608.06993.pdf
    - blog post: https://towardsdatascience.com/densenet-on-cifar10-d5651294a1a8
    '''

    def __init__(self,num_layers=100,growth_rate=12,theta=0.5,bn_size=4, drop_rate=0):
        super(DenseNetCifar10,self).__init__()

        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.theta = theta # compression factor
        self.bn_size = bn_size
        self.drop_rate = drop_rate

        self.in_channels = 3 # imgages are RGB
        self.num_classes = 10
        self.num_dense_blocks = 3
        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def forward(self,x):
        pred = x
        for k in self.layer_dict.keys(): # dict is ordered
            if k=='fc':
                pred = pred.view(pred.shape[0], -1)  # flatten
            pred = self.layer_dict[k](pred)

        return pred

    def build_module(self):
        '''
        defining layers requires knowing the shape of the input going into the layers. this module automatically
        infers these shapes and builds the network accordingly.
        '''
        print("building densenet module")
        x = torch.zeros((2,self.in_channels,32,32)) # dummy data
        out = x
        self.num_dens_layers = int(np.abs((self.num_layers - 4) / 3) * self.theta) # assumes 3 dense blocks.
        next_input_depth = 2 * self.growth_rate # from paper.

        self.layer_dict['conv1'] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=next_input_depth,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        out = self.layer_dict['conv1'](out)
        next_input_depth = out.shape[1]

        for i in range(self.num_dense_blocks):
            self.layer_dict['denseblock_{}'.format(i)] = _DenseBlock(
                num_layers=self.num_dens_layers, # stays fixed.
                num_input_features=next_input_depth,
                growth_rate=self.growth_rate,
                bn_size=self.bn_size,
                drop_rate=self.drop_rate
            )
            out = self.layer_dict['denseblock_{}'.format(i)](out)
            next_input_depth = out.shape[1]
            print('denseblock_{}: {}'.format(i+1, out.shape))

            if i != self.num_dense_blocks-1:
                self.layer_dict['transition_{}'.format(i)] = _Transition(
                    num_input_features=next_input_depth,
                    num_output_features=int(next_input_depth * self.theta)
                )
                out = self.layer_dict['transition_{}'.format(i)](out)
                next_input_depth = out.shape[1]
                print('transition_{}: {}'.format(i+1,out.shape))

        # final layers
        self.layer_dict['bn1'] = nn.BatchNorm2d(next_input_depth) # doesn't change shape.
        self.layer_dict['avg_pool'] = nn.AvgPool2d(kernel_size=out.shape[2])  # kernel_size = width
        self.layer_dict['fc'] = nn.Linear(next_input_depth, self.num_classes)

        for k in ['bn1','avg_pool','fc']:
            if k=='fc':
                out = out.view(out.shape[0],-1) # flatten
            out = self.layer_dict[k](out)
            print(out.shape,k)
        print(self.layer_dict.keys())

def tests():
    # make it so that you can train it!

    pass


if __name__ == '__main__':
    from mlp_resources.data_providers import CIFAR10

    '''
        def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
    '''

    '''
     def __init__(self, root, set_name,
                 transform=None, target_transform=None,
                 download=False):
    '''

    # rng = np.random.RandomState(seed=9112018)
    # valid_data = data_providers.MNISTDataProvider('valid', batch_size=100, rng=rng, max_num_batches=100)
    from globals import ROOT_DIR
    import os

    data_dir = os.path.join(ROOT_DIR, 'data')
    data = CIFAR10(root=data_dir, set_name='test', download=False)





    model = DenseNetCifar10()
    data = CIFAR10()

    '''


    '''



    #model.train_full()

    pass

