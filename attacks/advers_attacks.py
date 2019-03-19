import data_providers

print("import works")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate, fourier_gaussian, shift, zoom
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as img_transform


class LInfProjectedGradientAttackPenalty():
    '''
    performs max-norm attack projected gradient descent (gradient based iterative local optimizer)
    paper: https://arxiv.org/pdf/1611.01236.pdf

    design note: was easier to implement advers training when attacks were made classes with __call__
    '''

    def __init__(self, model, steps, alpha, epsilon, gamma, rand=False, targeted=False):
        self.model = model
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma  # penalty factor
        self.rand = rand
        self.targeted = targeted

    def __call__(self, x, y_true):
        '''
        :param x: numpy array size (1,img_height*img_width). single observation.
        :param y_true: numpy array size (1,num_classes). one-hot-encoded true label of x.
        :return: x_adv: numpy array size (1,img_height*img_width). adversarial example
        based on x_obs

        NOTE: EVERYTHING NEEDS TO BE ON GPU.
        '''

        if x.shape[0] != 1 or len(x.shape) > 2:
            raise ValueError('x incorrect shape, expected (1,-1) got {}'.format(x.shape))

        y_true_int = np.argmax(y_true, axis=1)  # one-hot to integer encoding.
        y_true_int_tens = torch.Tensor(y_true_int).long()

        if self.rand:
            delta = self.epsilon * np.random.uniform(-1, 1, size=x.shape)  # init random
        else:
            delta = np.zeros_like(x)  # init zeros

        x_adv = x + delta

        for _ in range(self.steps):
            x_adv_tens = torch.Tensor(x_adv).float()
            x_adv_tens.requires_grad = True
            y_pred = self.model(x_adv_tens)
            y_pred = torch.reshape(y_pred, (1, -1))
            loss = F.cross_entropy(input=y_pred, target=y_true_int_tens)
            loss.backward()
            loss_grad_wrt_x_at_x_adv = x_adv_tens.grad.data.numpy();
            loss_grad_wrt_x_at_x_adv = np.reshape(loss_grad_wrt_x_at_x_adv, (1, -1))

            var_grad_wrt_delta = (2 * self.gamma / delta.shape[1]) * delta - \
                                 (2 * self.gamma / (delta.shape[1]) ** 2) * \
                                 np.mean(delta) * np.ones_like(delta)

            obj_grad_wrt_delta = loss_grad_wrt_x_at_x_adv + var_grad_wrt_delta
            delta = delta + self.alpha * np.sign(obj_grad_wrt_delta)
            x_adv = x_adv + self.alpha * np.sign(obj_grad_wrt_delta)

            if self.targeted:
                x_adv = x_adv - self.alpha * np.sign(obj_grad_wrt_delta)
            else:
                x_adv = x_adv + self.alpha * np.sign(obj_grad_wrt_delta)

            x_adv = np.clip(x_adv, x - self.epsilon, x + self.epsilon)  # project onto max-norm (cube)

        return x_adv


class FastGradientSignAttack():
    def __init__(self, model, alpha, targeted=False):
        self.model = model
        self.alpha = alpha
        self.targeted = targeted

    def __call__(self, x, y_true):
        '''
        :param x: numpy array size (1,img_height*img_width). single observation.
        :param y_true: numpy array size (1,num_classes). one-hot-encoded true label of x.
        :return: x_adv: numpy array size (img_height, img_width). adversarial example
        based on x_obs
        '''
        if x.shape[0] != 1 or len(x.shape) > 2:
            raise ValueError('x incorrect shape, expected (1,-1) got {}'.format(x.shape))

        y_true_int = np.argmax(y_true, axis=1)  # F.cross_entropy requires target to be integer encoded
        y_true_int_tens = torch.Tensor(y_true_int).long()
        x_tens = torch.Tensor(x).float()  # input to model must be tensor of type float
        x_tens.requires_grad = True
        y_pred_tens = self.model(x_tens)  # returns tensor shape (-1,) of predicted class probabilities
        y_pred_tens = torch.reshape(y_pred_tens, (1, -1))  # required shape for cross_entropy
        loss = F.cross_entropy(input=y_pred_tens, target=y_true_int_tens)

        loss.backward()  # calculates (does not update) gradient delta_loss/delta_x for ever x that has requires_grad = true
        grad_wrt_x = x_tens.grad.data.numpy()  # returns array of size (-1,)
        grad_wrt_x = np.reshape(grad_wrt_x, (1, -1))  # row vector format (same format as input)

        if self.targeted:
            x_adv = x - self.alpha * np.sign(grad_wrt_x)
        else:
            x_adv = x + self.alpha * np.sign(grad_wrt_x)

        return x_adv


class ShrinkAttack():
    pass


class LInfProjectedGradientAttack2():
    def __init__(self, model, steps, alpha, epsilon, rand=False, targeted=False):
        self.model = model
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.rand = rand
        self.targeted = targeted

    def __call__(self, x, y_true_int, use_gpu=False, plot=False):
        '''
        :param x: numpy array size (1,num_channels, height, width). single observation.
        :param y_true_int: numpy array size (1,). integer encoded label.
        '''
        if len(x.shape) != 4:
            raise ValueError('Expected (1, num_channels, height, width); got {}'.format(x.shape))

        # convert numpy arrays to tensors.
        x = torch.Tensor(x).float().to(self.model.device)
        y_true_int = np.int64(y_true_int).reshape(-1, )
        y_true_int_tens = torch.Tensor(y_true_int).long().to(device=self.model.device)

        x_adv_tens = x

        for _ in range(self.steps):
            x_adv_tens.requires_grad = True
            y_pred = self.model(x_adv_tens)
            if y_pred.shape[0] == 1: y_pred = torch.reshape(y_pred, (1, -1))
            loss = F.cross_entropy(input=y_pred, target=y_true_int_tens)
            loss.backward()

            grad_x_adv = x_adv_tens.grad.clone()

            if self.targeted:
                x_adv_tens = x_adv_tens - self.alpha * torch.sign(x_adv_tens.grad)
            else:
                x_adv_tens = x_adv_tens + torch.clamp(self.alpha * torch.sign(x_adv_tens.grad),-self.epsilon,self.epsilon)

        out = x_adv_tens.cpu().detach().numpy()

        return out


class LInfProjectedGradientAttack():
    '''
    performs max-norm attack projected gradient descent (gradient based iterative local optimizer)
    paper: https://arxiv.org/pdf/1611.01236.pdf

    design note: was easier to implement advers training when attacks were made classes with __call__
    '''

    def __init__(self, model, steps, alpha, epsilon, rand=False, targeted=False):
        self.model = model
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.rand = rand
        self.targeted = targeted

    def __call__(self, x, y_true_int, use_gpu=False, plot=False):
        '''
        :param x: numpy array size (1,num_channels, height, width). single observation.
        :param y_true: numpy array size (1,). integer encoded label.
        :return: x_adv: numpy array size (img_height, img_width). adversarial example
        based on x_obs

        BOTTLENECK: Everything must be Tensors - don't use arrays!
        '''

        # if np.max(x) > 1.05 or np.min(x) < -1.05:
        #     raise Exception('image pixels must be between -1 and 1')

        if len(x.shape) != 4:
            raise ValueError('Expected (1, num_channels, height, width); got {}'.format(x.shape))
        x = torch.Tensor(x).float().to(self.model.device)
        y_true_int = np.int64(y_true_int).reshape(-1, )
        y_true_int_tens = torch.Tensor(y_true_int).long().to(device=self.model.device)

        # y_true_int = np.argmax(y_true, axis=1) # one-hot to integer encoding.

        if self.rand:
            # delta0 = self.epsilon * torch.uniform(-1, 1, size=x.shape)
            delta0 = self.epsilon * torch.empty(x.shape).uniform_(0,1)
            # init random
        else:
            # delta0 = np.zeros_like(x)  # init zeros
            delta0 = torch.zeros(x.shape)

        # x_adv = x + delta0
        x_adv_tens = x
        # x_adv_tens = torch.Tensor(x_adv).float().to(device=self.model.device)

        for _ in range(self.steps):
            # x_adv_tens = torch.Tensor(x_adv).float().to(device=self.model.device)
            x_adv_tens.requires_grad = True
            y_pred = self.model(x_adv_tens)
            if y_pred.shape[0] == 1: y_pred = torch.reshape(y_pred, (1, -1))
            loss = F.cross_entropy(input=y_pred, target=y_true_int_tens)
            loss.backward()


            #grad_x_adv = np.array(x_adv_tens.grad.data.cpu())  # numpy()
            #grad_x_adv = np.reshape(grad_x_adv, x_adv.shape)

            if plot:
                grad_x_adv = x_adv_tens.grad.clone()

            if self.targeted:
                # x_adv = x_adv - self.alpha * np.sign(obj_grad_wrt_delta)
                x_adv_tens = x_adv_tens - self.alpha * torch.sign(x_adv_tens.grad)
            else:
                x_adv_tens = x_adv_tens + torch.clamp(self.alpha * torch.sign(x_adv_tens.grad),-self.epsilon,self.epsilon)

            # x_adv = np.clip(x_adv, x - self.epsilon, x + self.epsilon)  # project onto max-norm (cube)
            # x_adv_arr = np.clip(x_adv_tens.data.numpy(), x.data.numpy() - self.epsilon, x.data.numpy() + self.epsilon)  # project onto max-norm (cube)
            # x_adv_tens = torch.clamp(x_adv_tens,x-self.epsilon,x+self.epsilon)
            # x_adv_tens = torch.clamp(x_adv_tens-x,-self.epsilon,self.epsilon) + x
            # x_adv_tens = torch.Tensor(x_adv_arr).float().to(device=self.model.device)

        if plot:
            import sys
            grad_normalized = (grad_x_adv - torch.mean(grad_x_adv,dim=0)) / torch.std(grad_x_adv,dim=0)



            grad_normalized = grad_normalized / (torch.max(grad_normalized,dim=0,keepdim=True)[0] \
                                                 - torch.min(grad_normalized,dim=0,keepdim=True)[0])
            x_mixed = torch.cat([x,grad_normalized , x_adv_tens], dim=2)
            x_mixed = torch.unbind(x_mixed, dim=0)
            x_mixed = torch.cat(x_mixed, dim=2)
            x_mixed = x_mixed.cpu().detach().numpy()
            x_mixed = np.transpose(x_mixed,(1, 2, 0))
            x_mixed = (127.5*(x_mixed + 1))/255

            print("x mixed shape: ", x_mixed.shape)

            plt.imshow(x_mixed)
            plt.show()


            # x_mixed = transforms.ToPILImage()(x_mixed)
            #x_mixed.show()
            sys.exit("Finished showing images")


        # return x_adv
        zz = x_adv_tens.cpu().detach().numpy()

        return zz

def l_two_pgd_attack(model, steps, alpha, epsilon):
    # projection just on sphere.
    # recall from thesis how to generate uniform points on sphere

    pass


def get_fooling_targets(true_target_int, num_classes):
    # returns one-hot-encoded fooling targets (targets not equal to true_target)

    fooling_classes = []  # all integers except x_batch[0] (7)
    for k in range(num_classes):
        if k != true_target_int[0]:
            fooling_classes.append(k)

    # next he one-hot-encodes them

    foolingtargets = np.zeros((len(fooling_classes), num_classes))
    for n in range(len(fooling_classes)):
        foolingtargets[n, fooling_classes[n]] = 1

    return foolingtargets


def plot_things(plot_dict, epsilon):
    '''
    :param x_batch: array size (batch_size,-1)
    :return:
    '''

    x_advers_batch = plot_dict['x_advers_batch']
    y_desired_ints = plot_dict['desired_targets']
    y_desired_probs = plot_dict['desired_targets_prob']
    y_predicted_ints = plot_dict['predicted_targets']
    y_predicted_probs = plot_dict['predicted_targets_prob']
    # {:.4f}
    plt.figure()

    for i in range(len(x_advers_batch)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_advers_batch[i].reshape((28, 28)), cmap='Greens')
        title = "desired: {} prob: {:.3f}".format(y_desired_ints[i], y_desired_probs[i])
        xlabel = "predicted: {} prob: {:.3f}".format(y_predicted_ints[i], y_predicted_probs[i])
        plt.title(title)
        plt.xlabel(xlabel)

    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("epsilon: {0}".format(epsilon))
    plt.show()


def main():
    from globals import ROOT_DIR;
    import os
    model_path = os.path.join(ROOT_DIR, 'martins_stuff/SavedModels/SimpleFNN/model_49')
    model = SimpleFNN(input_shape=(28, 28), h_out=100, num_classes=10)
    model.load_model(model_path)  # pre-trained model (acc 85%)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    test_data = data_providers.MNISTDataProvider('test', batch_size=100, rng=None, max_num_batches=1,
                                                 shuffle_order=False)

    x_batch, y_batch = test_data.next()
    y_batch_int = np.argmax(y_batch, axis=1)

    y_obs = np.reshape(y_batch_int[0], (1, 1))
    x_obs = np.reshape(x_batch[0], (1, -1))  # numpy row vector

    y_targets = get_fooling_targets(y_obs, num_classes=10)  # one-hot-encoded

    plot_dict = {}
    plot_dict['desired_targets'] = []
    plot_dict['desired_targets_prob'] = []
    plot_dict['predicted_targets'] = []
    plot_dict['predicted_targets_prob'] = []
    plot_dict['x_advers_batch'] = []

    eps = 0.15

    for y_target in y_targets:
        # get pred of advers example, and accuracy:
        y_target = np.reshape(y_target, (1, -1))
        x_advers = targeted_fast_gradient_sign_attack(x_obs, y_target, model, epsilon=eps)  # array (1,-1)
        x_advers_tens = torch.Tensor(x_advers).float()
        y_advers_pred_tens = F.softmax(model(x_advers_tens))
        y_advers_pred = y_advers_pred_tens.data.numpy()  # array (1,-1)

        desired_target = np.argmax(y_target)
        plot_dict['desired_targets'].append(desired_target)

        print(y_advers_pred, " sum: ", np.sum(y_advers_pred))
        plot_dict['desired_targets_prob'].append(y_advers_pred[0, desired_target])
        plot_dict['predicted_targets'].append(np.argmax(y_advers_pred))
        plot_dict['predicted_targets_prob'].append(np.max(y_advers_pred))
        plot_dict['x_advers_batch'].append(x_advers)

    temp = plot_dict['x_advers_batch']
    plot_dict['x_advers_batch'] = np.array(temp).reshape(len(temp), -1)

    plot_things(plot_dict, epsilon=eps)


class TranslateAttack():
    def __init__(self, possible_shifts=None):
        # NOTE: these shifts assume cifar10 data where (height,width) is (32,32)
        self.shifts_used = None
        self.possible_shifts = possible_shifts
        if possible_shifts is None:
            self.possible_shifts = [-12, -10, -8, -6, -4, 4, 6, 8, 10, 12]

    def __call__(self, x):
        choices = np.random.randint(0, len(self.possible_shifts), size=(len(x),))
        self.shifts_used = [self.possible_shifts[i] for i in choices]
        x_adv = np.zeros_like(x)
        for i in range(len(x)):
            x_adv[i] = shift(x[i], [0, 0, self.shifts_used[i]])
        return x_adv


class RotateAttack():
    def __init__(self, possible_rotations=None):
        self.possible_rotations = possible_rotations
        if possible_rotations is None:
            self.possible_rotations = [-360 + 10 * (i + 1) for i in range(71)]

        self.angles = None

    def __call__(self, x):
        '''
        :param x: array, shape: (batch_size, num_channels, height, width)
        '''
        from scipy.ndimage import rotate
        angles_idxs = np.random.randint(0, len(self.possible_rotations), size=(len(x),))
        self.angles = [self.possible_rotations[i] for i in angles_idxs]
        x_adv = np.zeros_like(x)
        for i in range(len(x)):
            x_adv[i] = rotate(x[i], self.angles[i], axes=(1, 2), reshape=False)
        return x_adv


def test():
    from data_viewer import ImageDataIO
    from models.densenet import DenseNet121
    model = DenseNet121()
    attack = LInfProjectedGradientAttack(
        model=model,
        steps=1,
        alpha=1,
        epsilon=10*4/255,
        rand=True,
        targeted=False
    )

    x, y = ImageDataIO.cifar10('train',normalize=True)
    x = x[:2]
    y = y[:2]

    print("max x: ",np.max(x)," min x: ",np.min(x))

    x_adv = attack(x,y,use_gpu=False,plot=True)




if __name__ == '__main__':
    # from data_io import ImageDataIO
    # from data_subsetter import DataSubsetter
    # from data_viewer import ImageDataViewer
    #
    # x, y = ImageDataIO.cifar10('test')
    # target = 3  # cat
    # x, y = DataSubsetter.condition_on_label(x, y, labels=[target], shuffle=False, rng=None)
    #
    # nrows = 3
    # ncols = 3
    #
    # x = x[:nrows * ncols]  # first image.
    # attack = RotateAttack()
    # attack = TranslateAttack()
    #
    # x_adv = attack(x)
    #
    # x_adv = x_adv / 255
    # x = x / 255
    #
    # labels = attack.shifts_used
    # ImageDataViewer.batch_view(x_adv, nrows=nrows, ncols=ncols, labels=labels, cmap='Greens', hspace=0, wspace=0)

    test()
    pass