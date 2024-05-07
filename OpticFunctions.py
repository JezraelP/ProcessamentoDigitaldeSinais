# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:36:26 2024

@author: jezra
"""

import numpy as np
from commpy.utilities  import upsample
from optic.models.devices import mzm, photodiode, edfa
from optic.models.channels import linearFiberChannel
from optic.comm.modulation import GrayMapping, modulateGray, demodulateGray
from optic.comm.metrics import  theoryBER
from optic.dsp.core import pulseShape, lowPassFIR, pnorm, signal_power

try:
    from optic.dsp.coreGPU import firFilter    
except ImportError:
    from optic.dsp.core import firFilter
    
from optic.utils import parameters, dBm2W
from optic.plot import eyediagram, pconst
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm.notebook import tqdm
import scipy as sp


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from optic.torchUtils import slidingWindowDataSet, MLP

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def SimulaSinalOptico(SpS, Rs, M, mod_format = 'pam', num_symbs = 1e6, pulse_format = 'nrz', MZM_Vpi = 2, Pi_dBm = 0, Pi_in = 0.5):
    
    Pi_W = dBm2W(Pi_dBm)
    
    paramMZM = parameters()
    paramMZM.Vpi = MZM_Vpi
    paramMZM.Vb = -paramMZM.Vpi/2
    
    bitsTx = np.random.randint(2, size = int(np.log2(M)*num_symbs))
    
    symbTx = modulateGray(bitsTx, M, mod_format)
    print("SymbTx antes da normalização: ", symbTx)
    symbTx = pnorm(symbTx) # normaliza a potência 
    
    symbUp = upsample(symbTx, SpS)
    
    pulse = pulseShape(pulse_format, SpS)
    pulse = pulse/max(abs(pulse))
    
    sigTx = firFilter(pulse, symbUp)
    
    
    Ai = np.sqrt(Pi_W)
    sigTxo = mzm(Ai, Pi_in*sigTx, paramMZM)
    
    return sigTxo, symbTx
def SimulaCanalOptico(sinal, SpS, Rs, dist_fibra, perda_fibra = 0.2, dispersao = 16, freq_central = 193.1e12, ruido = 4.5):
     
    Fs = SpS*Rs
    paramCh = parameters()
    paramCh.L = dist_fibra         # Distância [km]
    paramCh.α = perda_fibra        # Parâmetro de perdas da fibra [dB/km]
    paramCh.D = dispersao         # Parâmetro de dispersão da fibra [ps/nm/km]
    paramCh.Fc = freq_central  # Frequência óptica central [Hz]
    paramCh.Fs = Fs        # Frequência de amostragem da simulação [samples/second]
    
    sigCh = linearFiberChannel(sinal, paramCh)
    
    # Pré-amplificador do receptor
    paramEDFA = parameters()
    paramEDFA.G = paramCh.α*paramCh.L    # ganho edfa
    paramEDFA.NF = ruido   # edfa noise figure 
    paramEDFA.Fc = paramCh.Fc
    paramEDFA.Fs = Fs

    sigCh = edfa(sigCh, paramEDFA)
    
    paramPD = parameters()
    paramPD.ideal = False
    paramPD.B = Rs
    paramPD.Fs = Fs

    I_Rx = photodiode(sigCh, paramPD)
    
    return I_Rx

def RecuperaBits(sinal, SpS, M, mod_format = 'pam'):
    #Normaliza o sinal e seleciona os valores no intervalo de sinalização
    sinal = sinal/np.std(sinal)
    amostras = sinal[0::SpS]
    

    #Tira o nível DC e normaliza a potência
    amostras = amostras - amostras.mean()
    amostras = pnorm(amostras)
    

    # Demodula os símbolos para bits usando a mínima distância euclidiana 
    const = GrayMapping(M, mod_format) 
    Es = signal_power(const) 

    bits = demodulateGray(np.sqrt(Es)*amostras, M, mod_format)
    
    return bits

def CalculaBER(bits_transmitidos, bits_recebidos):
    discard = 100
    err = np.logical_xor(bits_recebidos[discard:bits_recebidos.size-discard], bits_transmitidos[discard:bits_transmitidos.size-discard])
    BER = np.mean(err)

    #Pb = 0.5*erfc(Q/np.sqrt(2)) # theoretical error probability
    print('Number of counted errors = %d '%(err.sum()))
    print('BER = %.2e '%(BER))
    #print('Pb = %.2e '%(Pb))

    err = err*1.0
    err[err==0] = np.nan

    plt.plot(err,'o', label = 'bit errors')
    plt.vlines(np.where(err>0), 0, 1)
    plt.xlabel('bit position')
    plt.ylabel('counted error')
    plt.legend()
    plt.grid()
    plt.ylim(0, 1.5)
    plt.xlim(0,err.size);



def PlotEyediagram(signal, SpS, label, discard = 100):
    eyediagram(signal[discard:-discard], signal.size-2*discard, SpS, plotlabel=label, ptype='fancy')



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

#FUNÇÕES PYTORCH

def CriaDataSet(symb,signal, SpS_in, train_pct = 0.8, Ntaps = 64, SpS_out = 1, batch_size = 64, shuffle = False):
    
    Nsymbols = len(symb)
    
    signal_full = pnorm(signal[0::SpS_in//SpS_out])   #Recupera o sinal nos intervalos de sinalização
    signal = pnorm(signal[0:Nsymbols*SpS_in:SpS_in//SpS_out])
    symbols = pnorm(symb[0:Nsymbols])   #Símbolos a serem passados como alvo do equalizador
    
    signal = (signal - np.mean(signal))/np.std(signal)
    signal_full = (signal_full - np.mean(signal_full))/np.std(signal_full)
    symbols = (symbols - np.mean(symbols))/np.std(symbols)
    
    symbols = symbols.reshape(-1,1)
    
    idx_train = np.arange(0, int(train_pct * len(symbols)))
    idx_eval = np.arange(int(train_pct * len(symbols)), len(symbols))

    sig_train = signal[idx_train] #80% para treinamento
    sig_test = signal[idx_eval]   #20% para avaliação do progresso
    
    train_dataset = slidingWindowDataSet(sig_train, symbols[idx_train], Ntaps, SpS_out)
    test_dataset = slidingWindowDataSet(sig_test, symbols[idx_eval], Ntaps, SpS_out)
    full_dataset = slidingWindowDataSet(signal_full, symbols, Ntaps, SpS_out)
    
    
    return train_dataset, test_dataset, full_dataset

def train_loop(train_dataloader, equalizer, loss, optimizer, cada_print):
    size = len(train_dataloader.dataset)
    
    equalizer.train()
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.float(), y.float()
        predict = equalizer(x)
        loss_value = loss(predict, y)
        
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % cada_print == 0:
            loss_value, current = loss_value.item(), (batch+1)*len(x)
            print(f"loss: {loss_value:>7f}[{current:>5d}/{size:>5d}]" )

def test_loop(progress_dataloader, equalizer, loss):
    size = len(progress_dataloader.dataset)
    num_batches = len(progress_dataloader)
    equalizer.eval()
    eval_loss = 0
    with torch.no_grad():
        for x, y in progress_dataloader:
            x, y = x.float(), y.float()
            pred = equalizer(x)
            eval_loss += loss(pred, y).item()
    eval_loss /= num_batches
    print(f"Perda média: {eval_loss:>8f} \n")

def GeraSinalEqualizado(model, dataloader):
    symbRx_NN_list = []
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.float()
            pred = model(X)

            # adiciona os símbolos preditos à lista
            symbRx_NN_list.append(pred.numpy().reshape(-1,))

    # concatena todos os símbolos preditos em um único array NumPy
    symbRx_NN = np.concatenate(symbRx_NN_list)
    
    return symbRx_NN
            