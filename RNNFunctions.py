# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:30:32 2024

@author: jezra
"""
import numpy as np
from optic.torchUtils import slidingWindowDataSet, MLP
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset
from optic.comm.modulation import GrayMapping, modulateGray, demodulateGray
from optic.dsp.core import pulseShape, lowPassFIR, pnorm, signal_power
from optic.utils import parameters, dBm2W
from optic.models.devices import mzm, photodiode, edfa
from optic.comm.modulation import GrayMapping, modulateGray, demodulateGray

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
            # Passar os dados pela RNN
            out, h_n = self.rnn(x, h0)  # h_n é o estado oculto final da última etapa de tempo

            # Passar a saída da RNN pela camada linear
            out = self.fc(out)
            return out
        

def train_loop_rnn(train_dataloader, model, loss, optimizer, cada_print):
    size = len(train_dataloader.dataset)
    
    model.train()
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.float().to(device), y.float().to(device)
        predict = model(x)

        print(predict.shape)
        loss_value = loss(predict, y)
        
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % cada_print == 0:
            loss_value, current = loss_value.item(), (batch+1)*len(x)
            print(f"loss: {loss_value:>7f}[{current:>5d}/{size:>5d}]" )
def test_loop_rnn(progress_dataloader, equalizer, loss):
    size = len(progress_dataloader.dataset)
    num_batches = len(progress_dataloader)
    equalizer.eval()
    eval_loss = 0
    with torch.no_grad():
        for x, y in progress_dataloader:
            x, y = x.float().to(device), y.float().to(device)
            pred = equalizer(x)
            eval_loss += loss(pred, y).item()
    eval_loss /= num_batches
    print(f"Perda média: {eval_loss:>8f} \n")

def SinalEqualizado(modelo_rnn, dataloader):
    modelo_rnn.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.float()
            pred = modelo_rnn(X)

            symbRx_NN = pred

            symbRx_NN = symbRx_NN.numpy().reshape(-1,)
    return symbRx_NN