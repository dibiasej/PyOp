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
from PyOp.PyOp import onlydata as od

def getprice(ticker):
    
    stock = pdr(ticker)
    price = stock['Adj Close'][-1]
    return price

def plot_surface(x, y, z):
		
		fig = plt.figure(figsize=(8, 8))
		ax = plt.gca(projection='3d')
		surf = ax.plot_surface(x, y, z, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)

		ax.set_xlabel('Strike')
		ax.set_ylabel('DTE')
		ax.set_zlabel('IV')
		ax.view_init(30, 30)

		return

def delta1(S, K, T, r, sigma, Otype):
    
    d1 = ((np.log(S / K) + (r + (sigma ** 2) / 2) * T) )/ (sigma * np.sqrt(T))
    
    #delta = np.exp(-r * T) * norm.cdf(d1)
    
    if Otype == 'calls':
        delta = np.exp(-r * T) * norm.cdf(d1)
        return delta
    elif Otype == 'puts':
        delta = np.exp(-r * T) * (norm.cdf(d1) - 1)
        return delta

def call_price(S, K, r, t, sigma):
    d1 = (np.log(K / S) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    C = np.multiply(S, norm.cdf(d1)) - np.multiply(norm.cdf(d2) * K, np.exp(-r * t))
    
    return C

def put_price(S, K, r, t, sigma):
    d1 = (np.log(K / S) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    P = -np.multiply(S, norm.cdf(-d1)) + np.multiply(norm.cdf(-d2) * K, np.exp(-r * t))
    
    return P

def bs_call(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

def bs_put(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def find_vol(target_value, S, K, T, r, *args):
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far

def getdeltaskew(ticker, spot, dte, lb, ub):

	calls = od.optiondata(ticker, 'calls')
	puts = od.optiondata(ticker, 'puts')

	calls = calls.loc[(calls['Strike'] > lb) & (calls['Strike'] < ub)]
	puts = puts.loc[(puts['Strike'] > lb) & (puts['Strike'] < ub)]
	calls = calls.loc[(calls['DTE'] > 1) & (calls['DTE'] < 350)]
	puts = puts.loc[(puts['DTE'] > 1) & (puts['DTE'] < 350)]

	calls['Delta'] = delta(spot, calls['Strike'], calls['DTE']/260, .03, calls['IV'], 'calls')
	puts['Delta'] = delta(spot, puts['Strike'], puts['DTE']/260, .03, puts['IV'], 'puts')

	cskew = calls.groupby('DTE').get_group(dte)
	pskew = puts.groupby('DTE').get_group(dte)

	cskew['Delta'] = cskew['Delta'].round(1)
	pskew['Delta'] = pskew['Delta'].round(1)

	civ = cskew.loc[cskew['Delta'] == 0.2]['IV'].iloc[0]
	piv = pskew.loc[pskew['Delta'] == -0.2]['IV'].iloc[0]

	deltaskew = piv - civ

	patm = pskew.loc[pskew['Delta'] == -0.5]['IV'].iloc[0]
	catm = cskew.loc[cskew['Delta'] == 0.5]['IV'].iloc[0]

	skewavg = (catm + patm) / 2

	skew = deltaskew / skewavg

	return skew


def getpremium(ticker, Otype, dte, strike):
    
    vd = od.volsurfacedata(ticker, Otype)
    
    exp = vd.groupby('DTE').get_group(dte)

    strike1 = int(strike)
    
    premium = exp.loc[exp['Strike'] == strike1]['Premium'].iloc[0]
    
    return premium

def delta(S, K, T, r, sigma, Otype):
    
    d1 = (np.log(K / S) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    
    #delta = np.exp(-r * T) * norm.cdf(d1)
    
    if Otype == 'calls':
        delta = np.exp(-r * T) * norm.cdf(d1)
        return delta
    elif Otype == 'puts':
        delta = np.exp(-r * T) * (norm.cdf(d1) - 1)
        return delta
def volga(S, K, T, r, sigma):
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    volga = (np.sqrt(T) * norm.pdf(d1)) * ((d1 * d2) / sigma)
    
    return volga

def vanna(S, K, T, r, sigma):
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    vanna = (-d2 * norm.pdf(d1)) / sigma
    
    return vanna
def vega(S, K, T, r, sigma):
    
    N_prime = norm.pdf
    
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)
    
    vega = S * np.sqrt(T) * N_prime(d1)
    
    return(vega)

def ultima(S, K, T, r, sigma, vega):
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    ultima = (vega/sigma**2) * (d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2)
    return ultima

def color(S, K, T, r, sigma):
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    #color = (np.exp(-r * T) * (norm.pdf(d1) / (2 * S * T * sigma * np.sqrt(T)))) * (2 * T + 1 + ((2(r) * T - d2 * sigma * np.sqrt(T))/(sigma * np.sqrt(T)))*d1)
    
    g = (norm.pdf(d1) / (2 * S * T * sigma * np.sqrt(T)))
    
    
    v = ((2 * r * T - d2 * sigma * np.sqrt(T)) / (sigma * np.sqrt(T))) * d1
    l = 2 * T + 1 + v
    
    color1 = -np.exp(T) * g * l
    
    return color1

def veta(S, K, T, r, sigma, vega):
    
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    x1 = (r * d1)/(sigma * np.sqrt(T))
    
    x2 = (1 - d1 * d2) / (2 * T)
    
    veta = vega * (r + x1 - x2)
    
    return veta
def charm(S, K, T, r, sigma, Otype):
    
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    if Otype == 'calls':
        
        #x2 = (r / (sigma * np.sqrt(T))) - (d2 / (2 * T))
        #charm = -np.exp(-r * T) * (norm.pdf(d1) * x2 - r * norm.cdf(d1))
        
        charm = np.exp(T) * norm.cdf(d1) - np.exp(T) * norm.pdf(d1) * ((2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))
        
        return charm
    
    elif Otype == 'puts':
        
        #x2 = (r / (sigma * np.sqrt(T))) - (d2 / (2 * T))
        #charm = np.exp(-r * T) * (norm.pdf(d1) * x2 + r * norm.cdf(d1))
        
        charm = -np.exp(T) * norm.cdf(-d1) - np.exp(T) * norm.pdf(d1) * ((2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))
        
        return charm

def theta(S, K, T, r, sigma, Otype):
    
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    if Otype == 'calls':
        
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
        
        return theta
    
    elif Otype == 'puts':
        
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        return theta

def speed(S, K, T, r, sigma):
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    speed = - np.exp(T) * (norm.pdf(d1) / (S ** 2 * sigma * np.sqrt(T))) * ((d1 / (sigma * np.sqrt(T))) + 1)
    
    return speed

def gamma(S, K, T, r, sigma):
    
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return gamma

def zomma(S, K, T, r, sigma, gamma):
    
    d1 = (np.log(S/K) + ((r + sigma ** 2 / 2) * T)) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    zomma = gamma * ((d1 * d2 - 1) / sigma)
    
    return zomma