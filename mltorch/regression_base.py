import torch
import numpy as np
import tqdm


class RegressionBase():
    
    def __init__(self, C=0.01, penalty='l1', lr=1e-6, max_iters=1000, tolerance=1e-5):
        self.C = C
        self.penalty = penalty
        self.lr = lr
        self.max_iters = max_iters
        self.weight = None
        self.tolerance = tolerance

    @staticmethod
    def add_intercept(X):
        nrow, _ = X.shape
        intercept = np.ones(nrow, dtype=X.dtype).reshape((nrow, 1))
        return np.concatenate([intercept, X], axis=1)

    def add_penalty(self, loss, weight):
        if self.penalty == 'l1':
            return loss + self.C * weight[1:].abs().sum()
        elif self.penalty == 'l2':
            return loss + self.C * torch.norm(weight[1:])
        else:
            return loss

    def _pre_fit(self, X, y):
        X = self.add_intercept(X)
        X, y = torch.tensor(X, dtype=torch.double), torch.tensor(y, dtype=torch.double)
        # self.weight = torch.randn(X.shape[1], dtype=torch.double, requires_grad=True)
        self.weight = torch.ones(X.shape[1], dtype=torch.double, requires_grad=True)
        return X, y

    def _loss(self, X, y):
        raise NotImplementedError

    def fit(self, X, y, progress_bar=True, show_loss=True):
        X, y = self._pre_fit(X, y)
        loss_history = list()
        
        steps = tqdm.trange(self.max_iters) if progress_bar else range(self.max_iters)
        for step in steps:
            loss = self._loss(X, y)

            if show_loss and step % 100 == 0:
                print('Loss at step {}: {:.2f}'.format(step, loss))
            
            if loss_history and (loss - loss_history[-1]) ** 2 < self.tolerance:
                return self
            
            loss_history.append(loss)
            loss.backward()
            self.weight.data -= self.lr * self.weight.grad
            self.weight.grad.data.zero_()
        
        if show_loss:    
            print('Loss after training: {:.2f}'.format(loss))
        self.weight = self.weight.detach().numpy()

    def _predict(self, X):
        raise NotImplementedError
        
    def predict(self, X):
        if self.weight is None:
            raise ValueError('The model is not fitted yet.')
        return self._predict(X)