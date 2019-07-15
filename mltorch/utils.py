import torch


def one_hot(label, n_category=None):
    batch_size = label.shape[0]
    n_category = n_category or (label.max().item() + 1)

    y_onehot = torch.zeros((batch_size, n_category), dtype=torch.long)
    y_onehot.scatter_(1, label.view(-1, 1), 1)
    return y_onehot
