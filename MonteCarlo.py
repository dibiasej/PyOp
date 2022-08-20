import pandas as pd
import numpy as np
import numba
import math
from setuptools import setup
import matplotlib.pyplot as plt
import datetime as dt

def MonteCarlo(S0, K, r, sigma, dte, M, I):
    
    T = dte / 252
    
    def MonteCarlo2(p):
        M, I = p
        dt = T / M
        S = np.zeros((M + 1, I))
        S[0] = S0
        rn = np.random.standard_normal(S.shape)
        for t in range(1, M + 1):
            S[t] = S[t - 1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * rn[t])
        return S
    
    numba_mc = numba.jit(MonteCarlo2)
    S = numba_mc((M, I))
    
    data = pd.DataFrame(S)
    
    today = dt.datetime.now()
    
    exp = today + dt.timedelta(dte)
    
    date_range = pd.date_range(start = today, end = exp, periods = M + 1)
    data.index = date_range
    plt.figure(figsize=(10,6))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Monte Carlo')
    plt.xticks(rotation=42)
    #plt.plot(S[:, 1], 'r', lw=5)
    for i in range(0, I):
        plt.plot(data.index, data.iloc[:, i])
    #plt.plot(S[20, :], 'r', lw=3)
        
    C = np.exp(-r * T) * np.maximum(S[-1] - K, 0).mean()
    P = np.exp(-r * T) * np.maximum(K - S[-1], 0).mean()
    return  print('Call Price = ' + str(C),'Put price = ' + str(P))

def npMonteCarlo(S0, K, r, sigma, dte, M, I):
    T = dte / 252

    def MonteCarlo2(p):
        M, I = p
        dt = T / M
        S = np.zeros((M + 1, I))
        S[0] = S0
        rn = np.random.standard_normal(S.shape)
        for t in range(1, M + 1):
            S[t] = S[t - 1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * rn[t])
        return S
    
    #numba_mc = numba.jit(MonteCarlo2)
    S = MonteCarlo2((M, I))
    data = pd.DataFrame(S)

    today = dt.datetime.now()

    exp = today + dt.timedelta(dte)

    date_range = pd.date_range(start = today, end = exp, periods=M + 1)
    data.index = date_range

    plt.figure(figsize=(10,6))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Monte Carlo')
    plt.xticks(rotation=45)
    for i in range(0, I):
        plt.plot(data.index, data.iloc[:, i])
        
    C = np.exp(-r * T) * np.maximum(S[-1] - K, 0).mean()
    return print('Call Price = ' + str(C))


def GetCall(S0, K, r, sigma, dte, M, I):
    T = dte / 252

    def MonteCarlo2(p):
        M, I = p
        dt = T / M
        S = np.zeros((M + 1, I))
        S[0] = S0
        rn = np.random.standard_normal(S.shape)
        for t in range(1, M + 1):
            S[t] = S[t - 1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * rn[t])
        return S
        
    S = MonteCarlo2((M, I))
    data = pd.DataFrame(S)
    C = np.exp(-r * T) * np.maximum(S[-1] - K, 0).mean()
    P = np.exp(-r * T) * np.maximum(K - S[-1], 0).mean()
    return(C, P)