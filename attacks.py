import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = torch.Tensor(tensor)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def reverse_augmentations(images):
    transform_train = UnNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    save_ready_images = []
    for image in images:
        save_ready_images.append(transform_train(image))

    return torch.stack(save_ready_images, dim=0)


def NoAttack(inputs, model, labels, save_images, filepath):
    inputs = nn.Parameter(inputs, requires_grad=True)
    per_step_preds = model.forward(inputs)

    if save_images:
        images = inputs.data.detach().cpu()
        images = reverse_augmentations(images).unbind(0)
        images = torch.cat(images, dim=1)
        images = transforms.ToPILImage()(images)
        images.save(fp=filepath)

    return per_step_preds, labels


def PGDAttack(inputs, model, labels, save_images, filepath):

    inputs = nn.Parameter(inputs, requires_grad=True).to(inputs.device)
    attacked_inputs = inputs.clone()
    for i in range(10):

        per_step_preds = model.forward(attacked_inputs)
        loss = F.cross_entropy(input=per_step_preds, target=labels)

        grads = torch.autograd.grad(inputs=[attacked_inputs], outputs=-loss)  # try variant where the model anticipates the future

        attacked_inputs = attacked_inputs.clone() - 1000 * grads[0]

    combined_inputs = torch.cat([inputs, attacked_inputs.detach()], dim=0)
    combined_labels = torch.cat([labels, labels], dim=0)

    if save_images:
        images = inputs.data.detach().cpu()
        images = reverse_augmentations(images).unbind(0)
        images = torch.cat(images, dim=1)
        images = normalize(images)

        attacked_images = attacked_inputs.data.detach().cpu()
        attacked_images = reverse_augmentations(attacked_images).unbind(0)
        attacked_images = torch.cat(attacked_images, dim=1)
        attacked_images = normalize(attacked_images)

        attacked_grads = torch.sign(grads[0]).data.detach().cpu()
        attacked_grads = reverse_augmentations(attacked_grads).unbind(0)
        attacked_grads = torch.cat(attacked_grads, dim=1)
        attacked_grads = normalize(attacked_grads)

        images = torch.cat([images, attacked_images, attacked_grads], dim=2)

        print(grads[0].shape, grads[0].abs().sum(), images.max(), images.min())
        images = transforms.ToPILImage()(images)
        images.save(fp=filepath)

    per_step_preds = model.forward(combined_inputs)

    return per_step_preds, combined_labels


def FGSMAttack(inputs, model, labels, save_images, filepath):
    inputs = nn.Parameter(inputs, requires_grad=True).to(inputs.device)
    per_step_preds = model.forward(inputs)
    loss = F.cross_entropy(input=per_step_preds[-1], target=labels)

    grads = torch.autograd.grad(inputs=[inputs], outputs=-loss)  # try variant where the model anticipates the future

    attacked_inputs = inputs.clone() - 0.5 * torch.sign(grads[0])

    combined_inputs = torch.cat([inputs, attacked_inputs.detach()], dim=0)
    combined_labels = torch.cat([labels, labels], dim=0)

    if save_images:
        images = inputs.data.detach().cpu()
        images = reverse_augmentations(images).unbind(0)
        images = torch.cat(images, dim=1)
        images = normalize(images)

        attacked_images = attacked_inputs.data.detach().cpu()
        attacked_images = reverse_augmentations(attacked_images).unbind(0)
        attacked_images = torch.cat(attacked_images, dim=1)
        attacked_images = normalize(attacked_images)

        attacked_grads = torch.sign(grads[0]).data.detach().cpu()
        attacked_grads = reverse_augmentations(attacked_grads).unbind(0)
        attacked_grads = torch.cat(attacked_grads, dim=1)
        attacked_grads = normalize(attacked_grads)

        images = torch.cat([images, attacked_images, attacked_grads], dim=2)

        print(grads[0].shape, grads[0].abs().sum(), images.max(), images.min())
        images = transforms.ToPILImage()(images)
        images.save(fp=filepath)

    per_step_preds = model.forward(combined_inputs)

    return per_step_preds, combined_labels


def FGMAttack(inputs, model, labels, save_images, filepath):
    inputs = nn.Parameter(inputs, requires_grad=True).to(inputs.device)
    per_step_preds = model.forward(inputs)
    loss = F.cross_entropy(input=per_step_preds[-1], target=labels)

    grads = torch.autograd.grad(inputs=[inputs], outputs=-loss)  # try variant where the model anticipates the future

    attacked_inputs = inputs.clone() - 1000 * grads[0]

    combined_inputs = torch.cat([inputs, attacked_inputs.detach()], dim=0)
    combined_labels = torch.cat([labels, labels], dim=0)

    if save_images:
        images = inputs.data.detach().cpu()
        images = reverse_augmentations(images).unbind(0)
        images = torch.cat(images, dim=1)
        images = normalize(images)

        attacked_images = attacked_inputs.data.detach().cpu()
        attacked_images = reverse_augmentations(attacked_images).unbind(0)
        attacked_images = torch.cat(attacked_images, dim=1)
        attacked_images = normalize(attacked_images)

        attacked_grads = torch.sign(grads[0]).data.detach().cpu()
        attacked_grads = reverse_augmentations(attacked_grads).unbind(0)
        attacked_grads = torch.cat(attacked_grads, dim=1)
        attacked_grads = normalize(attacked_grads)

        images = torch.cat([images, attacked_images, attacked_grads], dim=2)

        print(grads[0].shape, grads[0].abs().sum(), images.max(), images.min())
        images = transforms.ToPILImage()(images)
        images.save(fp=filepath)

    per_step_preds = model.forward(combined_inputs)

    return per_step_preds, combined_labels


def normalize(x):
    # x = (x - x.mean(dim=0).expand_as(x)) / x.std(dim=0).expand_as(x)
    x = (x - x.min()) / (x.max() - x.min())
    return x