import torch.nn as nn
import torch
from torchvision.models.densenet import DenseNet
from torchvision.models.densenet import densenet161
from ModelBuilder.base import Network
import numpy as np
from collections import OrderedDict

from torchvision.models.densenet import _DenseBlock
from torchvision.models.densenet import _Transition
import torch.nn.functional as F

from mlp_resources.data_providers import CIFAR10

class DenseNetWrapper(DenseNet,Network):
    def __init__(self,in_channels=3,growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        Network.__init__(self)
        DenseNet.__init__(self,
                          num_init_features=num_init_features,
                          growth_rate=growth_rate,
                          block_config=block_config)

        self.features.conv0 = nn.Conv2d(in_channels,out_channels=num_init_features, kernel_size=7, stride=2, padding=3, bias=False)

        # TODO: Adapt num_classes to be custom (e.g. 10 not 1000)

class DenseNetCifar10(nn.Module):
    '''
    implementation of DenseNet-BC.
    - three DenseBlocks.
    - each DenseBlock has the same number of DenseLayers.

    '''

    def build_module(self):
        '''
        defining layers requires knowing the shape of the input going into the layers. this module automatically
        infers these shapes and builds the network accordingly.
        '''
        print("building densenet module")
        in_channels = 3

        x = torch.zeros((2,in_channels,32,32))
        out = x

        self.num_dens_layers = int(np.abs((self.num_layers - 4) / 3) * self.theta)
        self.init_num_features = 2 * self.growth_rate

        self.layer_dict['conv1'] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.init_num_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        out = self.layer_dict['conv1'](out)
        next_input_depth = out.shape[1]

        num_dense_blocks = 3

        for i in range(num_dense_blocks):
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

            if i != num_dense_blocks-1:
                self.layer_dict['transition_{}'.format(i)] = _Transition(
                    num_input_features=next_input_depth,
                    num_output_features=int(next_input_depth * self.theta)
                )
                out = self.layer_dict['transition_{}'.format(i)](out)
                next_input_depth = out.shape[1]
                print('transition_{}: {}'.format(i+1,out.shape))


        '''
        TRANSITION BLOCK.
        - reduces width & height of image by 2 (by pooling).
        - reduces number of feature maps of denseblock by theta
        '''

        # Transition Block 3
        self.bn1 = nn.BatchNorm2d(next_input_depth) # doesn't change shape.
        self.avgpool = nn.AvgPool2d(kernel_size=8)

        out = self.avgpool(out)
        print(out.shape, "pooled")
        out = out.view(out.shape[0],-1) # flatten (must be added to fprop!)

        print(out.shape, "flattened")


        num_classes = 10

        self.fc = nn.Linear(next_input_depth, num_classes)

        out = self.fc(out)

        print(out.shape)



        pass


    def __init__(self,num_layers=100,growth_rate=12,bn_size=4, drop_rate=0):
        super(DenseNetCifar10,self).__init__()
        self.theta = 0.5
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.init_num_features = 2 * growth_rate
        self.bn_size = bn_size
        self.drop_rate = drop_rate

        self.layer_dict = nn.ModuleDict()

        self.build_module()

        # https://towardsdatascience.com/densenet-on-cifar10-d5651294a1a8

        '''
                Hyperparameter for DenseLayer (doesn't affect shape of DenseBlock output):
                The number 4 is given by the authors in the paper and most of the repositories call it bn_size
                given in __init__ of DenseNet
                '''

        # self.num_dens_layers = np.abs((num_layers-4)/3)*self.theta
        #
        # self.conv1 = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=self.init_num_features,
        #     kernel_size=3, stride=1, padding=1, bias=False
        # )
        #
        # # num_input_features = the depth of the volume that goes into a DenseBlock.
        # curr_num_features = self.init_num_features # depth of conv1
        #
        # self.densblock1 = _DenseBlock(
        #     num_layers=self.num_dens_layers,
        #     num_input_features=curr_num_features,
        #     growth_rate=growth_rate,
        #     bn_size=bn_size,
        #     drop_rate = drop_rate
        # )

        '''
        _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        '''

        # recall: final depth of DenseBlock is k0 + (l-1)*k
        # to figure out: what is drop_rate, and bn_size.
        # how to automatically infer depth - build dummy data.

        '''
        Description of cifar 10:
        (1) 45k training, 5k validation, 10k testing images.
        (2) 10 classes.
        '''

        pass

    pass


class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()

        num_init_features = 64
        in_channels = 3
        self.conv0 = nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers_dict = {}
        block_config = (6, 12)
        bn_size = 4
        growth_rate = 32
        drop_rate = 0
        num_classes = 10

        num_features = num_init_features
        for i, num_layers in enumerate(block_config): # 0,1,2,3
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.layers_dict['denseblock_{}'.format(i+1)] = block
            num_features = num_features + num_layers * growth_rate

            if i!= len(block_config)-1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.layers_dict['transition_{}'.format(i+1)] = trans
                num_features = num_features // 2

        self.layers_dict['norm5'] = nn.BatchNorm2d(num_features)

        self.classifier = nn.Linear(num_features,num_classes)




        '''
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
        '''




    def forward(self,x):

        pred = self.conv0(x)
        pred = self.norm0(pred)
        pred = self.relu0(pred)
        pred = self.pool0(pred)

        print(self.layers_dict.keys())

        for k in self.layers_dict.keys():
            pred = self.layers_dict[k](pred)
            print(pred.size())

        pred = F.relu(pred,inplace=True)

        print(pred.size())
        pred = F.avg_pool2d(pred, kernel_size=1, stride=1).view(pred.size(0), -1)
        pred = self.classifier(pred)

        return pred




def test_forward_pass():
    import numpy as np
    from mlp_resources.data_providers import CIFAR10

    #rng = np.random.RandomState(seed=9112018)
    #valid_data = data_providers.MNISTDataProvider('valid', batch_size=100, rng=rng, max_num_batches=100)
    from globals import ROOT_DIR
    import os

    data_dir = os.path.join(ROOT_DIR,'data')
    data = CIFAR10(root=data_dir,set_name='test',download=True) # if not train then test

    # x_img, y = data[0:1] # img, target = self.data[index], self.labels[index]

    x_batch_list = []
    for i in range(10):
        x_img, _ = data[i]
        x_img = np.array(x_img)
        x = np.transpose(x_img, (2, 1, 0)) # (3,32,32)
        x_batch_list.append(x)

    x_batch = np.array(x_batch_list)

    print(x_batch.shape)

    model = TestNetwork()
    x_batch_tens = torch.Tensor(x_batch).float()
    pred = model(x_batch_tens)

    print(pred.size())



    pass

def test_forward2():
    num_init_features = 64
    features = nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))
    ]))
    print(features)
    features.conv0 = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
    print(features)
    # print(features)


    pass

def test_densenet():
    model = DenseNetCifar10()



    pass

def main():
    test_densenet()
    # test_forward2()

    pass

if __name__ == '__main__':
    main()