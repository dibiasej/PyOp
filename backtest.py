import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime as dt
import yfinance as yf
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate as spi
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
import scipy
from pandas_datareader import get_data_yahoo as pdr
from PyOp.PyOp import functions as f
from PyOp.PyOp import onlydata as od


def call_price(S, K, r, t, sigma):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    C = np.multiply(S, norm.cdf(d1)) - np.multiply(norm.cdf(d2) * K, np.exp(-r * t))
    
    return C

def put_price(S, K, r, t, sigma):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    P = -np.multiply(S, norm.cdf(-d1)) + np.multiply(norm.cdf(-d2) * K, np.exp(-r * t))
    
    return P

class payoff(object):
    
    def __init__(self, ticker,  r=.03, sigma=.35, sput = 0, lput = 0, scall = 0, lcall = 0):
        
        self.ticker = ticker
        self.r = r
        self.sigma = sigma
        self.sput = sput
        self.lput = lput
        self.scall = scall
        self.lcall = lcall
        
    
    def getstrikes(self, t):
        
        spot = f.getprice(self.ticker)
        
        if self.sput > 0:
            
            sputK = []
            sputP = []
            for sp in range(self.sput):
                
                K = input(f'Enter the strike of your {sp + 1} short put')
                print(K)
                P = f.getpremium(self.ticker, 'puts', t, K)
                
                sputK.append(K)
                sputP.append(P)
                
        elif self.sput == 0:
            sputK = 0
            sputP = 0
                
        if self.lput > 0:
            
            lputK = []
            lputP = []
            for lp in range(self.lput):
                
                K = input(f'Enter the strike of your {lp + 1} long put')
                
                P = f.getpremium(self.ticker, 'puts', t, K)
                
                lputK.append(K)
                lputP.append(P)
                
        elif self.lput == 0:
            lputK = 0
            lputP = 0
            
        if self.scall > 0:
            
            scallK = []
            scallP = []
            
            for sc in range(self.scall):
                
                K = input(f'Enter the strike of your {sc + 1} short call')
                
                P = f.getpremium(self.ticker, 'calls', t, K)
                
                scallK.append(K)
                scallP.append(P)
        elif self.scall == 0:
            scallK = 0
            scallP = 0
                
        if self.lcall > 0:
            
            lcallK = []
            lcallP = []
            for lc in range(self.lcall):
                
                K = int(input(f'Enter the strike of your {lc + 1} long call'))
                
                P = f.getpremium(self.ticker, 'calls', t, K)
                
                lcallK.append(K)
                lcallP.append(P)
                
        elif self.lcall == 0:
            lcallK = 0
            lcallP = 0
        df = pd.DataFrame({})
        
        
        return sputK, lputK, scallK, lcallK, sputP, lputP, scallP, lcallP
    
    def diagram(self):
        
        spot = f.getprice(self.ticker)
        
        vd = od.volsurfacedata(self.ticker, 'calls')
        
        doos = vd['DTE'].unique()
        
        print(f'list of DTEs: {doos}')
        
        DTE = input('Choose a DTE')
        
        t = int(DTE)
        
        spk, lpk, sck, lck, spp, lpp, scp, lcp = self.getstrikes(t)
        
        S = np.linspace(spot - spot * .15, spot * .15 + spot, 100)
        
        df = pd.DataFrame()
        
        df0 = pd.DataFrame()
        
        if spk == 0:
            df['Short Put'] = 0

            del df['Short Put']
            
            df0['Short Put'] = 0

            del df0['Short Put']
            
        else:
        
            for k, g in zip(spk, spp):
                i = int(k)

                shortputprice = -put_price(S, i, self.r, t/260, self.sigma)

                shortputprice0 = -put_price(S, i, self.r, 0, self.sigma)

                df0[f'{k} Short Put'] = shortputprice0

                df[f'{k} Short Put'] = shortputprice

                df[f'{k} Short Put Premium'] = g

                df0[f'{k} Short Put Premium'] = g
        
        if lpk == 0:
            
            df['Long Put'] = 0

            del df['Long Put']
            
            df0['Long Put'] = 0

            del df0['Long Put']
        else:
            
            for k, g in zip(lpk, lpp):
                i = int(k)

                longputprice = put_price(S, i, self.r, t/260, self.sigma)

                longputprice0 = put_price(S, i, self.r, 0, self.sigma)

                df0[f'{k} Long Put'] = longputprice0

                df[f'{k} Long Put'] = longputprice

                df[f'{k} Long Put Premium'] = -g

                df0[f'{k} Long Put Premium'] = -g
        
        if sck == 0:
            
            df['Short Call'] = 0

            del df['Short Call']
            
            df0['Short Call'] = 0

            del df0['Short Call']
        else:
            
            for k, g in zip(sck, scp):
                i = int(k)

                shortcallprice = -call_price(S, i, self.r, t/260, self.sigma)

                shortcallprice0 = -call_price(S, i, self.r, 0, self.sigma)

                df0[f'{k} Short Call'] = shortcallprice0

                df[f'{k} Short Call'] = shortcallprice

                df[f'{k} Short Call Premium'] = g

                df0[f'{k} Short Call Premium'] = g
        
        if lck == 0:
            
            df['Long Call'] = 0

            del df['Long Call']
            
            df0['Long Call'] = 0

            del df0['Long Call']
        else:
            
            for k, g in zip(lck, lcp):
                i = int(k)

                longcallprice = call_price(S, i, self.r, t/260, self.sigma)

                longcallprice0 = call_price(S, i, self.r, 0, self.sigma)

                df0[f'{k} Long Call'] = longcallprice0

                df[f'{k} Long Call'] = longcallprice

                df[f'{k} Long Call Premium'] = -g

                df0[f'{k} Long Call Premium'] = -g
        
        cols = df.columns
        print(lck, sck, spk, lpk)
        cols0 = df0.columns
        
        df['PL'] = 0
        
        df0['PL'] = 0
        
        for col in cols0:
            
            df0['PL'] += df0[col] 
        
        for col in cols:
            
            df['PL'] += df[col]
        plt.figure(figsize=(8,6))
        
        plt.plot(S, df['PL'], 'r--')
        plt.plot(S, df0['PL'], 'b')
        plt.xlabel('Spot Price')
        plt.ylabel('P&L')
        plt.show()
        return df