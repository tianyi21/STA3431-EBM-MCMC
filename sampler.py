#   -*- coding: utf-8 -*-
#
#   sampler.py
#
#   Created by Tianyi Liu on 2021-11-27 as tianyi
#   Copyright (c) 2021. All Rights Reserved.
#
#   Distributed under terms of the MIT license.


from tqdm import tqdm
import torch
import torchvision as tv
import numpy as np

from params import *


class Sampler:
    def __init__(self, buf_size, batch_size, img_size, device):
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.img_size = img_size
        self.buffer = torch.rand((buf_size,) + img_size) * 2 - 1
        self.device = device
    
    def _fetch(self, reinit_freq=0.05):
        idx = np.random.choice(self.buf_size, (self.batch_size, ), replace=False)
        prev_samples = self.buffer[idx]
        reinit_idx = (torch.rand(self.batch_size) < reinit_freq)[:,None,None,None]
        reinit_samples = torch.rand_like(prev_samples) * 2 - 1
        return torch.where(reinit_idx, reinit_samples, prev_samples), idx
    
    def _restore(self, sample, idx):
        self.buffer[idx] = sample.detach().cpu()
        
    def visualize(self, title, path):
        grid = tv.utils.make_grid(self.buffer[:100], nrow=10)
        tv.utils.save_image(grid, f"{SAVE_PATH}/{path}/buffer_{title}.pdf")
    
    def sample(self):
        raise NotImplementedError
        
        
class SGLD(Sampler):
    """Stochastic Gradient Langevin Dynamics
        Welling, Max; Teh, Yee Whye (2011). 
        Bayesian Learning via Stochastic Gradient Langevin Dynamics"""
    def __init__(self, n_step, lr, sd, buf_size, batch_size, img_size, device):
        super(SGLD, self).__init__(buf_size, batch_size, img_size, device)
        self.n_step = n_step
        self.lr = lr
        self.sd = sd
        self.buf_size = buf_size
        self.batch_size = batch_size
    
    def sample(self, model, reinit_freq):
        model.eval()
        samples, idx = self._fetch(reinit_freq)
        samples = torch.autograd.Variable(samples, requires_grad=True).to(self.device)
        for _ in tqdm(range(self.n_step), leave=False, ncols=80):
            grad_x = torch.autograd.grad(model(samples).sum(), [samples], retain_graph=True)[0]
            grad_x = grad_x.data.clamp_(-0.03, 0.03)
            samples += self.lr * grad_x + self.sd * torch.randn_like(samples)
            samples.data.clamp_(-1, 1)
        self._restore(samples, idx)
        model.train()
        return samples
    
    
class HMC(Sampler):
    """Hamiltonian Monte Carlo
        Neal, Radford (2011).
        MCMC using Hamiltonian dynamics."""
    def __init__(self, n_step, path_n, step_size, buf_size, batch_size, img_size, device, sigma=1):
        super(HMC, self).__init__(buf_size, batch_size, img_size, device)
        self.n_step = n_step
        self.path_n = path_n
        self.step_size = step_size
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.img_size = img_size
        self.sigma = sigma
    
    def sample(self, model, reinit_freq):
        model.eval()
        samples, idx = self._fetch(reinit_freq)
        samples = torch.autograd.Variable(samples, requires_grad=True).to(self.device)
        dVdq = lambda q: - torch.autograd.grad(model(q).sum(), [q], retain_graph=True)[0]
        masks = None
        for _ in tqdm(range(int(self.n_step.val)), leave=False, ncols=80):
            p_init = torch.rand_like(samples) * 2 - 1
            q_new, p_new = self._leapfrog(self, samples, p_init, dVdq)
            q_new.data.clamp_(-1.1)
            init_log_p = - model(samples) - self._gaussian_pdf(p_init).sum((-1,-2))
            new_log_p = - model(q_new) - self._gaussian_pdf(p_new).sum((-1,-2))
            mask = (torch.log(torch.rand(self.batch_size, 1)) < (init_log_p - new_log_p).cpu())[:,None,None].to(self.device)
            samples = torch.where(mask, q_new, samples)
            masks = mask if masks is None else torch.cat((masks, mask), dim=0)
            print(f"> HMC: acceptance rate={mask.float().mean().cpu():.2%}")
        print(f"> Overal: acceptance rate={masks.float().mean().cpu():.2%}")
        self._restore(samples, idx)
        model.train()
        return samples
    
    @staticmethod
    def _leapfrog(self, q, p, dVdq):
        p = p - self.step_size * dVdq(q) / 2
        q = q + self.step_size * p / self.sigma
        for _ in range(self.path_n - 1):
            p = p - self.step_size * dVdq(q)
            q = q + self.step_size * p / self.sigma
        p = p - self.step_size * dVdq(q) / 2
        return q, p
    
    @staticmethod
    def _gaussian_pdf(x, mu=0, sigma=1):
        return - 0.5 * ((x - mu) / sigma) ** 2


class HASGLD(Sampler):
    def __init__(self, n_step, lr, sd, path_n, step_size, buf_size, batch_size, img_size, device, sigma=1):
        super(HASGLD, self).__init__(buf_size, batch_size, img_size, device)
        self.n_step = n_step
        self.lr = lr
        self.sd = sd
        self.path_n = path_n
        self.step_size = step_size
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.img_size = img_size
        self.sigma = sigma
    
    def sample(self, model, reinit_freq):
        model.eval()
        samples, idx = self._fetch(reinit_freq)
        samples = torch.autograd.Variable(samples, requires_grad=True).to(self.device)
        for _ in tqdm(range(self.n_step), leave=False, ncols=80):
            grad_x = torch.autograd.grad(model(samples).sum(), [samples], retain_graph=True)[0]
            grad_x = grad_x.data.clamp_(-0.03, 0.03)
            samples += self.lr * grad_x + self.sd * torch.randn_like(samples)
            samples.data.clamp_(-1, 1)
        samples, idx = self._fetch(reinit_freq)
        samples = torch.autograd.Variable(samples, requires_grad=True).to(self.device)
        dVdq = lambda q: - torch.autograd.grad(model(q).sum(), [q], retain_graph=True)[0]
        masks = None
        for _ in tqdm(range(self.n_step), leave=False, ncols=80):
            p_init = torch.rand_like(samples) * 2 - 1
            q_new, p_new = self._leapfrog(self, samples, p_init, dVdq)
            q_new.data.clamp_(-1.1)
            init_log_p = - model(samples) - self._gaussian_pdf(p_init).sum((-1,-2))
            new_log_p = - model(q_new) - self._gaussian_pdf(p_new).sum((-1,-2))
            mask = (torch.log(torch.rand(self.batch_size, 1)) < (init_log_p - new_log_p).cpu())[:,None,None].to(self.device)
            samples = torch.where(mask, q_new, samples)
            masks = mask if masks is None else torch.cat((masks, mask), dim=0)
        print(f"> HMC: acceptance rate={masks.float().mean().cpu():.2%}")
        self._restore(samples, idx)
        model.train()
        return samples


class DualAveragingStepSizeScheduler:
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75, tune_step=20):
        self.initial_step_size = initial_step_size
        self.mu = np.log(10 * initial_step_size)
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.t = t0
        self.kappa = kappa
        self.tune_step = tune_step
        
        self.count = 0
        self.error_sum = 0
        self.log_step = 0
        self.log_averaged_step = 0
        self.disable = True
        
        self.step_size = initial_step_size

    def __call__(self, p_accept):
        if self.disabled or self.count > self.tune_step:
            return None

        self.count += 1
        self.error_sum += self.target_accept - p_accept
        self.log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        eta = self.t ** -self.kappa
        self.log_averaged_step = eta * self.log_step + (1 - eta) * self.log_averaged_step
        self.t += 1
        self.step_size = np.exp(self.log_step) if self.count < self.tune_step else np.exp(self.log_averaged_step)
        
    def reinit(self):
        self.disable = False
        self.count = 0
        self.error_sum = 0
        self.log_averaged_step = 0
        self.t = self.t0
    
    
class AdaptiveHMC(HMC):
    def __init__(self, n_step, path_n, step_size, buf_size, batch_size, img_size, device, sigma=1):
        super(AdaptiveHMC, self).__init__(n_step, path_n, step_size, buf_size, batch_size, img_size, device, sigma=sigma)
        self.step_size_scheduler = DualAveragingStepSizeScheduler(step_size)
        self.step_size_scheduler.disabled = False

    def sample(self, model, reinit_freq):
        model.eval()
        samples, idx = self._fetch(reinit_freq)
        samples = torch.autograd.Variable(samples, requires_grad=True).to(self.device)
        dVdq = lambda q: - torch.autograd.grad(model(q).sum(), [q], retain_graph=True)[0]
        masks = None
        self.step_size_scheduler.reinit()
        for _ in tqdm(range(int(self.n_step.val)), leave=False, ncols=80):
            p_init = torch.rand_like(samples) * 2 - 1
            q_new, p_new = self._leapfrog(self, samples, p_init, dVdq)
            q_new.data.clamp_(-1.1)
            init_log_p = - model(samples) - self._gaussian_pdf(p_init).sum((-1,-2))
            new_log_p = - model(q_new) - self._gaussian_pdf(p_new).sum((-1,-2))
            mask = (torch.log(torch.rand(self.batch_size, 1)) < (init_log_p - new_log_p).cpu())[:,None,None].to(self.device)
            samples = torch.where(mask, q_new, samples)
            masks = mask if masks is None else torch.cat((masks, mask), dim=0)
            if self.n_step.val > 20:
                self.step_size_scheduler(masks.float().mean().cpu())
                self.step_size = self.step_size_scheduler.step_size
                print(f"> Current step size={self.step_size:.4f} acceptance rate={mask.float().mean().cpu():.2%}")
        print(f"> Overal: acceptance rate={masks.float().mean().cpu():.2%}")
        self._restore(samples, idx)
        model.train()
        return samples
