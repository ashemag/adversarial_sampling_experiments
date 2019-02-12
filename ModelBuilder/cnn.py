import torch.nn as nn
import torch
from collections import OrderedDict
from ModelBuilder.base import Network


class ConvNetCifar10(Network):
    '''
    this class can be used to create CNNs that have roughly same format as VGG networks.
    '''

    def __init__(self,num_classes=10,img_num_channels=3,img_size=(32,32),config=None):
        super(ConvNetCifar10,self).__init__()
        self.num_classes = num_classes
        self.img_num_channels = img_num_channels
        self.img_height = img_size[0]
        self.img_width = img_size[1]

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
                modules.append(
                    nn.Conv2d(
                        in_channels=out.shape[1],
                        out_channels=config_dict[self._config_keys['out_channels']],
                        kernel_size=config_dict[self._config_keys['kernel_size']],
                        stride=config_dict[self._config_keys['stride']],
                        padding=config_dict[self._config_keys['padding']],
                        bias=False
                    )
                )
                if config_dict['nl'] == 'relu':
                    modules.append(nn.ReLU(inplace=True))

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

            self.layer_dict['fc_{}'.format(fc_idx)] = nn.Linear(
                in_features=out.shape[1],
                out_features=config_dict[self._config_keys['out_features']],
                bias=config_dict[self._config_keys['bias']]
            )

            # update the depth of the current volume (used for creating subsequent layers)
            out = self.layer_dict['fc_{}'.format(fc_idx)](out)
            print(out.shape, "fc")

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


def test_module():
    model = ConvNetCifar10()
    x = torch.zeros((2,3,32,32)) # dummy batch to infer layer shapes.
    print("testing forward pass:")
    pred = model(x)
    print(pred)

    pass

if __name__ == '__main__':
    test_module()
