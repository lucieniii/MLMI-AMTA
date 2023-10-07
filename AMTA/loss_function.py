import torch

## loss function
def one_hot(y_true, target_label):
    last_channel = torch.zeros_like(y_true) + 1
    t = ()
    for i in target_label:
        label = torch.zeros_like(y_true)
        label[y_true==i] = 1
        last_channel[y_true==i] = 0
        t += (label,)
    t += (last_channel,)
    return torch.cat(t, dim=1)

## loss function
def loss_dice(y_pred, y_true, target_label, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    y_label = one_hot(y_true, target_label)
    # print(y_pred.shape, y_label.shape)
    numerator = torch.sum(y_pred*y_label, dim=(-3,-2,-1)) * 2
    denominator = torch.sum(y_pred, dim=(-3,-2,-1)) + torch.sum(y_label, dim=(-3,-2,-1)) + eps
    return torch.mean(1. - (numerator / denominator))
