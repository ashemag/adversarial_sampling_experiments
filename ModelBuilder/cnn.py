import torch.nn as nn
import torch
from collections import OrderedDict
from adversarial_sampling_experiments.ModelBuilder.base import Network

class ConvNetBuilder(Network):
    '''
    this class can be used to create CNNs that have roughly same format as VGG networks.
    '''

    def __init__(self,num_classes=10,img_num_channels=3,img_size=(32,32),config=None):
        super(ConvNetBuilder, self).__init__()
        self.num_classes = num_classes
        self.img_num_channels = img_num_channels
        self.img_height = img_size[0]
        self.img_width = img_size[1]

        print("here")

        # specifies the format of keys of the config dicts. changing this requires changing keys of config_list
        self._config_keys = {
            'stride':'s',
            'kernel_size':'k',
            'padding':'p',
            'out_channels':'d',
            'out_features':'hdim',
            'bias':'bias'
        }

        if config is None: # default architecture
            '''
            the config_list is used to define the network. if no config is specified the default network is used.
            default architecture: CONV > MPOOL (4x) > global avg. pool > FC. CONV layers are 3x3 (kernel_size = 3),
            and MPOOL are 2x2. after each CONV layer ReLU tranform is added.
            '''
            self.config_list = [
                {'type': 'conv', 'd': 32, 'k': 3, 's': 1, 'p': 1,'nl':'relu','repeat':1},
                {'type': 'mpool','k':2,'repeat':1},
                {'type': 'conv', 'd': 32, 'k': 3, 's': 1, 'p': 1,'nl':'relu', 'repeat': 1},
                {'type': 'mpool', 'k': 2, 'repeat': 1},
                {'type': 'conv', 'd': 32, 'k': 3, 's': 1, 'p': 1, 'nl': 'relu', 'repeat': 1},
                {'type': 'mpool', 'k': 2, 'repeat': 1}
            ]
            classifier_pattern = [
                {'type':'global_apool'},
                {'type': 'fc','hdim':num_classes,'bias':False}
            ]
            self.config_list += classifier_pattern
        else:
            self.config_list = config

            # if param not included in dict then use default

            for i,config_dict in enumerate(self.config_list):
                if config_dict['type'] == 'conv':
                    if 'bias' not in config_dict.keys():
                        config_dict['bias'] = False
                    if 'repeat' not in config_dict.keys():
                        config_dict['repeat'] = 1

                if config_dict['type'] == 'fc':
                    if 'bias' not in config_dict.keys():
                        config_dict['bias'] = False
                    if 'repeat' not in config_dict.keys():
                        config_dict['repeat'] = 1

            # nl, batch-norm, dropout don't need defaults. if not included in dict then won't be used.




        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def forward(self,x):
        pred = x
        for k in self.layer_dict.keys(): # dict is ordered
            if k[0:2]=='fc': # can be e.g. fc_2
                pred = pred.view(pred.shape[0], -1)  # flatten
            pred = self.layer_dict[k](pred)

        return pred

    def build_module(self):
        '''
        defining layers requires knowing the shape of the input going into the layers. this module automatically
        infers these shapes and builds the network accordingly.
        '''

        def add_conv_layer(out, config_dict, conv_idx):
            repeat = config_dict['repeat']

            for _ in range(repeat):
                modules = []
                conv = nn.Conv2d(
                    in_channels=out.shape[1],
                    out_channels=config_dict[self._config_keys['out_channels']],
                    kernel_size=config_dict[self._config_keys['kernel_size']],
                    stride=config_dict[self._config_keys['stride']],
                    padding=config_dict[self._config_keys['padding']],
                    bias=config_dict[self._config_keys['bias']]
                )
                modules.append(conv)

                if 'bn' in config_dict.keys() and config_dict['bn']:
                    out_temp = conv(out)
                    modules.append(nn.BatchNorm2d(out_temp.shape[1]))  # comes before non-linearity (but sometimes after?) (https://github.com/keras-team/keras/issues/5465)

                if 'nl' in config_dict.keys() and config_dict['nl'] == 'relu':
                    modules.append(nn.ReLU(inplace=True)) # non-linearity after CONV (see mlpractical repo)

                self.layer_dict['conv_{}'.format(conv_idx)] = nn.Sequential(*modules) # combine CONV with non-linearity

                # update the depth of the current volume (used for creating subsequent layers)
                out = self.layer_dict['conv_{}'.format(conv_idx)](out)

                # update next idx of conv layer (used for naming the layers)
                conv_idx += 1
                print(out.shape,"conv")
            return out, conv_idx

        def add_pool_layer(out,config_dict,pool_idx,type):
            repeat = config_dict['repeat']

            for _ in range(repeat):
                if type == 'mpool':
                    self.layer_dict[type+'_{}'.format(pool_idx)] = nn.MaxPool2d(
                        kernel_size=config_dict[self._config_keys['kernel_size']]
                    )
                if type == 'apool':
                    self.layer_dict[type+'_{}'.format(pool_idx)] = nn.AvgPool2d(
                        kernel_size=config_dict[self._config_keys['kernel_size']]
                    )

                # update the depth of the current volume (used for creating subsequent layers)
                out = self.layer_dict[type+'_{}'.format(pool_idx)](out)

                # update next idx of pool layer (used for naming the layers)
                pool_idx += 1
                print(out.shape, "pool")

            return out, pool_idx

        def add_fc_layer(out,config_dict,fc_idx):
            if len(out.shape) > 2:
                out = out.view(out.shape[0], -1)  # flatten into (batch_size, -1)

            label = ''
            modules = []
            if 'dropout' in config_dict.keys():
                modules.append(nn.Dropout(config_dict['dropout'],inplace=True))
                label += 'dropout'

            fc =  nn.Linear(
                in_features=out.shape[1],
                out_features=config_dict[self._config_keys['out_features']],
                bias=config_dict[self._config_keys['bias']]
            )
            modules.append(fc)
            label += '-fc'

            if 'nl' in config_dict.keys() and config_dict['nl']=='relu':
                modules.append(nn.ReLU(inplace=True))
                label += '-relu'

            self.layer_dict['fc_{}'.format(fc_idx)] = nn.Sequential(*modules)

            # update the depth of the current volume (used for creating subsequent layers)
            out = self.layer_dict['fc_{}'.format(fc_idx)](out)
            print(out.shape, label)

            # update next idx of fc layer (used for naming the layers)
            fc_idx += 1

            return out, fc_idx

        def add_global_avg_pool(out):
            '''
            E.g. if you have a tensor (n,10,8,8) yoou apply global pooling it reduces it to (n,10,1,1) i.e. you
            summarize the spatial dimension of the input volume.
            '''
            self.layer_dict['global_avg_pool'] = nn.AvgPool2d(kernel_size=out.shape[2])
            out = self.layer_dict['global_avg_pool'](out)
            print(out.shape, "global avg. pool")

            return out

        print("building cnn module")
        x = torch.zeros((2,self.img_num_channels,self.img_height,self.img_width)) # dummy batch to infer layer shapes.
        out = x
        print(out.shape, "input")

        conv_idx = 0; mpool_idx = 0; apool_idx = 0; fc_idx = 0
        for layer_config_dict in self.config_list:
            if layer_config_dict['type'] == 'conv':
                out, conv_idx = add_conv_layer(out,layer_config_dict,conv_idx)

            if layer_config_dict['type'] == 'apool':
                out, apool_idx = add_pool_layer(out,layer_config_dict,apool_idx,'apool')

            if layer_config_dict['type'] == 'mpool':
                out, mpool_idx = add_pool_layer(out, layer_config_dict, mpool_idx, 'apool')

            if layer_config_dict['type'] == 'fc':
                out, fc_idx = add_fc_layer(out,layer_config_dict,fc_idx)

            if layer_config_dict['type'] == 'global_apool':
                out = add_global_avg_pool(out)


def vgg_cifar10():
    '''
    returns a VGG networks for cifar10 as built in paper here: https://arxiv.org/pdf/1710.10766.pdf
    - conv0: filter size 3 by 3 (k=3), feature maps 16 (d=16), stride 1 (s=1), batch norm, relu (nl:relu).
    0 padding is 1 see: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    '''
    num_classes = 10
    config = [
        {'type': 'conv', 'd': 16, 'k': 3, 's': 1, 'p': 1, 'nl': 'relu','bn':True,'bias':False,'repeat': 2},
        {'type': 'mpool', 'k': 2, 'repeat': 1},
        {'type': 'conv', 'd': 128, 'k': 3, 's': 1, 'p': 1, 'nl': 'relu', 'bn': True,'bias':False, 'repeat': 2},
        {'type': 'mpool', 'k': 2, 'repeat': 1},
        {'type': 'conv', 'd': 512, 'k': 3, 's': 1, 'p': 1, 'nl': 'relu', 'bn': True,'bias':False, 'repeat': 3},
        {'type': 'mpool', 'k': 2, 'repeat': 1},
        {'type': 'conv', 'd': 512, 'k': 3, 's': 1, 'p': 1, 'nl': 'relu', 'bn': True, 'repeat': 3},
        {'type': 'mpool', 'k': 2, 'repeat': 1},
        {'type': 'fc', 'hdim': 512,'nl':'relu','dropout':0.5,'bias':False},
        {'type': 'fc','hdim':num_classes,'dropout':0.5,'bias':False} # no non-linearity
    ]

    model = ConvNetBuilder(num_classes=10,img_num_channels=3,img_size=(32,32),config=config)
    x = torch.zeros((2,3,32,32)) # dummy batch to infer layer shapes.

    pred = model(x)

    #print(pred)

    return model


def test_module():
    # model = ConvNetBuilder()

    # print("testing forward pass:")
    # pred = model(x)
    # print(pred)

    x = torch.zeros((2,3,32,32)) # dummy batch to infer layer shapes.
    # model = vgg_cifar10(x)
    # pred = model(x)

    model = vgg_cifar10()
    print("PRED")
    pred = model(x)

    print(pred)

    pass

if __name__ == '__main__':
    test_module()
