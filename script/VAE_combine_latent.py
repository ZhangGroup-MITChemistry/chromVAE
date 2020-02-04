__author__ = "Wenjun Xie <xwj123@gmail.com>"

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from utils import *
from models import *

# hyper-parameters
h_dim = 200
z_dim = 2
learning_rate = 1e-3


# load contact matrix
seqs_WT = np.loadtxt("../data/HCT116/config_90kb_interpolated_450nm.txt", dtype = np.int)
seqs_depletion = np.loadtxt("../data/HCT116_auxin/config_90kb_interpolated_450nm.txt", dtype = np.int)
seqs = np.concatenate((seqs_WT, seqs_depletion), axis=0)
seqs = seqs.reshape(len(seqs), -1)
num_sample, num_site = seqs.shape

torch_seqs = torch.from_numpy(seqs).type(torch.FloatTensor)
train_data = Binary_Dataset(torch_seqs)
train_data_loader = DataLoader(train_data, batch_size = num_sample, shuffle = False)


# load VAE model
vae = VAE(num_site, h_dim, z_dim).cuda()
state_dict = torch.load("../output/VAE_combine_num_samples_{}_num_sites_{}_h_dim_{}_z_dim_{}.pt".format(num_sample, num_site, h_dim, z_dim))
vae.load_state_dict(state_dict['state_dict'])
vae.eval()


# get latent space
start_time = time.time()

for idx, x in enumerate(train_data_loader):
    with torch.no_grad():
        x = x.cuda()
        x_reconst, mu, log_var = vae.forward(x)
mu = mu.cpu().data.numpy()

np.save("../analysis/latent_space_combine_num_samples_{}_num_sites_{}_z_dim_{}".format(num_sample, num_site, z_dim),mu)

print("time used: {:.2f}".format(time.time() - start_time))

exit()
