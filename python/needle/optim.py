"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            if self.momentum != 0.0:
                v_prev = self.u.get(p, 0)
                v = self.momentum * v_prev + (1 - self.momentum) * g
                self.u[p] = v
                update = v
            else:
                update = g

            p.data = p.data - self.lr * update

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias_c1 = 1.0 - (b1 ** self.t)
        bias_c2 = 1.0 - (b2 ** self.t)

        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            m_prev = self.m.get(p, 0)
            v_prev = self.v.get(p, 0)

            m = b1 * m_prev + (1.0 - b1) * g
            v = b2 * v_prev + (1.0 - b2) * (g * g)

            self.m[p] = m
            self.v[p] = v

            m_hat = m / bias_c1
            v_hat = v / bias_c2

            inv_rms = np.power(v_hat, 0.5) + self.eps
            step = self.lr * (m_hat / inv_rms)

            p.data = p.data - step
