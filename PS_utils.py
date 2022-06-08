"""
The following code is under the MIT License.

Author: Isaac Sheidlower, Github: IsaacSheidlower, Email: Isaac.Sheidlower@tufts.edu
Affiliation: Tufts University, AABL Lab
"""

from operator import truediv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from scipy.special import expit

def sample_normal(agent, actor, observation, with_noise=False, max_action=2, env_only=False, 
                    with_grad_env=True, with_grad_agent=True, kappa=.9):
    def get_dist(agent, actor, observation):
        observation = torch.Tensor([observation]).to('cpu')
        mu1, sigma1 = agent.actor.get_dist(observation, with_grad=with_grad_agent)
        mu2, sigma2 = actor.actor.get_dist(observation, with_grad = with_grad_env)
        mu1 = mu1[0].detach().numpy()
        sigma1 = sigma1[0].detach().numpy()
        mu2 = mu2[0].detach().numpy()
        sigma2 = sigma2[0].detach().numpy()
        kl = np.tanh(np.log(np.sqrt(sigma2)/np.sqrt(sigma2)) + (sigma2+(mu1-mu2)**2)/(2*sigma2) - .5)
        for i in range(len(kl)):
            if kl[i] > kappa:
                kl[i] = kappa
        mu = mu1*(kl) + mu2*(1-(kl))
        sigma = sigma2
        mu = torch.from_numpy(mu)
        sigma = torch.from_numpy(sigma)
        #print(mu, sigma)
        return Normal(mu, sigma), mu.numpy(), sigma.numpy(), np.mean(kl)

    def get_dist_env(actor, observation):
        observation = torch.Tensor([observation]).to('cpu')
        mu1, sigma1 = actor.actor.get_dist(observation, with_grad=with_grad_env)
        mu1 = mu1[0].detach().numpy()
        sigma1 = sigma1[0].detach().numpy()
        mu = mu1
        sigma = np.zeros(4) 
        sigma[0] = sigma1[0] 
        sigma[1] = sigma1[1] 
        sigma[2] = sigma1[2] 
        sigma[3] = sigma1[3] 
        mu = torch.from_numpy(mu)
        sigma = torch.from_numpy(sigma)
        #print(mu, sigma)
        return Normal(mu, sigma), mu, sigma

    if env_only is False:
        dist, mu, sigma, kl = get_dist(agent, actor, observation)
        if with_noise:
            sample = dist.rsample().numpy()
        else:
            sample = dist.sample().numpy()
        #print(sample)
        sample = max_action * np.tanh(sample)
        return sample, dist, mu, sigma, kl
    else:
        dist, mu, sigma = get_dist_env(actor, observation)
        if with_noise:
            sample = dist.rsample().numpy()
        else:
            sample = dist.sample().numpy()
        #print(sample)
        sample = max_action * np.tanh(sample)
        return sample, dist, mu, sigma
