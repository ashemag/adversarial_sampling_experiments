
def fast_gradient_sign_attack(x_obs, y_obs_true, model, epsilon=100):
    '''
    gradient wrt input x shows how much the loss change wrt to small change in x. this fact is
    used to construct an example that worsens the loss of an observation.

    :param: x_obs is a numpy array of size (1,-1). represents a single observation of an image.
    :param: y_obs_true numpy array of size (1,-1). is one-hot-encoded. gives true label of x_obs
    :return: adversarial example. is a numpy array of size (1,-1)
    '''

    loss, x_obs_temp = cross_entropy_loss_obs(x_obs,y_obs_true,model)
    loss.backward() # calculates (does not update) gradient delta_loss/delta_x for ever x that has requires_grad = true
    grad_wrt_x_obs = x_obs_temp.grad.data.numpy() # returns array of size (-1,)
    grad_wrt_x_obs = np.reshape(grad_wrt_x_obs,(1,-1)) # row vector format (same format as input)
    advers_x_obs = x_obs + epsilon*np.sign(grad_wrt_x_obs)

    return advers_x_obs

def cross_entropy_loss_obs(x_obs, y_obs_target, model):
    '''
    :param: x_obs is a numpy array of size (1,-1). represents a single observation of an image.
    :param: y_obs_target numpy array of size (1,-1). is one-hot-encoded. gives true label of x_obs
    :return: adversarial example. is a numpy array of size (1,-1)
    '''

    # F.cross_entropy requires target to be integer encoded
    y_obs_desired_int = np.argmax(y_obs_target, axis=1)
    y_obs_desired_int = torch.Tensor(y_obs_desired_int).long()

    # input to model must be tensor of type float
    x_obs_temp = torch.Tensor(x_obs).float()
    x_obs_temp.requires_grad = True
    y_obs_pred = model(x_obs_temp)  # returns tensor shape (-1,) of predicted class probabilities
    y_obs_pred = torch.reshape(y_obs_pred, (1, -1))  # required shape for cross_entropy
    loss = F.cross_entropy(input=y_obs_pred, target=y_obs_desired_int)

    return loss, x_obs_temp

def targeted_fast_gradient_sign_attack(x_obs,y_obs_desired,model,epsilon=100):
    '''
    instead of moving in direction that worsens the loss we try to improve loss wrt. to desired target.
    source: https://medium.com/onfido-tech/adversarial-attacks-and-defences-for-convolutional-neural-networks-66915ece52e7
    :return:
    '''

    loss, x_obs_temp = cross_entropy_loss_obs(x_obs,y_obs_desired,model)
    loss.backward() # calculates (does not update) gradient delta_loss/delta_x for ever x that has requires_grad = true
    grad_wrt_x_obs = x_obs_temp.grad.numpy()  # returns array of size (-1,)
    grad_wrt_x_obs = np.reshape(grad_wrt_x_obs, (1, -1))  # row vector format (same format as input)
    advers_x_obs = x_obs - epsilon * np.sign(grad_wrt_x_obs)

    return advers_x_obs

def fast_gradient_sign_attack(x_obs, y_obs_true, model, epsilon=100):
    '''
    gradient wrt input x shows how much the loss change wrt to small change in x. this fact is
    used to construct an example that worsens the loss of an observation.

    :param: x_obs is a numpy array of size (1,-1). represents a single observation of an image.
    :param: y_obs_true numpy array of size (1,-1). is one-hot-encoded. gives true label of x_obs
    :return: adversarial example. is a numpy array of size (1,-1)
    '''

    loss, x_obs_temp = cross_entropy_loss_obs(x_obs,y_obs_true,model)
    loss.backward() # calculates (does not update) gradient delta_loss/delta_x for ever x that has requires_grad = true
    grad_wrt_x_obs = x_obs_temp.grad.data.numpy() # returns array of size (-1,)
    grad_wrt_x_obs = np.reshape(grad_wrt_x_obs,(1,-1)) # row vector format (same format as input)
    advers_x_obs = x_obs + epsilon*np.sign(grad_wrt_x_obs)

    return advers_x_obs