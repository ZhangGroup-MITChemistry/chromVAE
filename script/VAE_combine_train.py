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
num_epochs = 1000
z_dim = 2
batch_size = 500
learning_rate = 1e-3


# load contact matrix
seqs_WT = np.loadtxt("../data/HCT116/config_90kb_interpolated_450nm.txt", dtype = np.int)
seqs_depletion = np.loadtxt("../data/HCT116_auxin/config_90kb_interpolated_450nm.txt", dtype = np.int)
seqs = np.concatenate((seqs_WT, seqs_depletion), axis=0)
seqs = seqs.reshape(len(seqs), -1)
num_sample, num_site = seqs.shape

torch_seqs = torch.from_numpy(seqs).type(torch.FloatTensor)
train_data = Binary_Dataset(torch_seqs)
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)


# train VAE
vae = VAE(num_site, h_dim, z_dim).cuda()
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

start_time = time.time()

train_loss_epoch = []
for epoch in range(num_epochs):
    running_loss = []    
    for idx, x in enumerate(train_data_loader):
        x = x.cuda()
        loss = vae.loss(x)
        running_loss.append(loss.cpu().data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (idx+1) % 10 == 0:
            print ("Epoch[{:>3d}/{:>3d}], Step [{:>3d}/{:>3d}], Loss: {:.4f}".format(epoch+1, num_epochs, idx+1, len(train_data_loader), loss.cpu().data.numpy()))
    train_loss_epoch.append(np.mean(running_loss))


torch.save({'state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
           "../output/VAE_combine_num_samples_{}_num_sites_{}_h_dim_{}_z_dim_{}.pt".format(num_sample, num_site, h_dim, z_dim))
print("time used: {:.2f}".format(time.time() - start_time))

exit()
