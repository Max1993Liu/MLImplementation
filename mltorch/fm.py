import torch
import numpy as np
try:
    from tqdm import trange
except:
    trange = range


from .metrics import mean_square_error, binary_crossentropy



class FMBase:

    loss_fn = None

    def __init__(self, k=10, max_iter=100, lr=1e-3, reg_v=0.1, reg_w=0.5, tolerance=1e-5):
        self.k = k
        self.max_iter = max_iter
        self.lr = lr
        self.reg_v = reg_v
        self.reg_w = reg_w
        self.tolerance = tolerance

    def fit(self, X, y):
        # initialization
        n_sample, n_feature = X.shape

        self.bias = torch.tensor(0.0, requires_grad=True)
        self.w = torch.zeros(n_feature, dtype=torch.float32, requires_grad=True)
        self.v = torch.randn((n_feature, k)) / np.sqrt(n_sample)
        v.requires_grad_()

        if self.loss_fn is None:
            raise ValueError('Specify the loss function in the subclass.')

        prev_loss = None
        for step in trange(self.max_iter):
            loss = loss_fn(y, predict(X))

            if prev_loss is not None and np.abs(loss.item() - prev_loss) < self.tolerance:
                print('Early stopped at step {}'.format(step + 1))
                break 
            prev_loss = loss

            # update the bias
            loss.backward()
            self.bias.data -= lr * self.bias.grad

            # update the linear weights
            (torch.norm(self.w, 1) * reg_w).backward()
            self.w.data -= lr * self.w.grad

            # update the factor weights
            (loss + torch.norm(self.v, 1) * reg_v).backward()
            self.v.data -= lr * self.v.grad

            self.w.grad.zero_()
            self.v.grad.zero_()
            self.bias.grad.zero_()
        
        print('Optimization complete')
        self.w = w
        self.v = v
        self.bias = bias

    def predict(self, X):
        linear_output = X @ self.w
        factor_output = ((X @ self.v) ** 2 - (X ** 2) @ self.v ** 2).sum(dim=1) / 2
        prediction = linear_output + factor_output + self.bias
        return prediction


class FMClassifier(FMBase):
    
    loss_fn = binary_crossentropy


class FMRegressor(FMBase):

    loss_fn = mean_square_error