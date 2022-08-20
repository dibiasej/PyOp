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
from PyOp.PyOp import functions as f
from PyOp.PyOp import rvolcone as rcone

class volatility(object):
    
    def __init__(self, ticker, risk_free=0.3, sigma=0.35, angle1 = 30, angle2 = 60):
        
        self.ticker = ticker
        self.risk_free = risk_free
        self.sigma = sigma
        self.angle1 = angle1
        self.angle2 = angle2


    def surface(self, Otype):
        
        volsurface = od.volsurfacedata(self.ticker, Otype)
        
        spot = f.getprice(self.ticker)
        
        dtes = volsurface['DTE'].unique()

        ivreg = []

        x = 0

        pct = np.linspace(.02, .99, 70)

        for dte in dtes:

        	exp = volsurface.groupby('DTE').get_group(dte)

        	if len(exp) < 15:

        		continue

        	else:

        		ivexp = []

        		for p in pct:

        			iv = exp.iloc[(int(len(exp) * p))]['IV']
        			ivexp.append(iv)

        		x = np.linspace(exp['Strike'].min(), exp['Strike'].max(), len(ivexp))
        		regy = np.polyfit(x, ivexp, deg=2)
        		evy = np.polyval(regy, x)

        		ivreg.append(evy)

        ivarr = np.array(ivreg)

        ttm = np.linspace(0, 350, len(ivarr))
        strike, time = np.meshgrid(x, ttm)

        plot = f.plot_surface(strike, time, ivarr)
        return volsurface
    def greeksurface(self, Otype, greek):
	    spot = f.getprice(self.ticker)

	    volsurface = od.volsurfacedata(self.ticker, Otype)

	    volsurface['Delta'] = f.delta(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], Otype)

	    volsurface['Volga'] = f.volga(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	    volsurface['Vanna'] = f.vanna(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	    volsurface['Vega'] = f.vega(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	    volsurface['Ultima'] = f.ultima(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], volsurface['Vega'])

	    volsurface['Color'] = f.color(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	    volsurface['Veta'] = f.veta(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], volsurface['Vega'])

	    volsurface['Charm'] = f.charm(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], Otype)

	    volsurface['Theta'] = f.theta(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], Otype)

	    volsurface['Speed'] = f.speed(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	    volsurface['Gamma'] = f.gamma(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	    volsurface['Zomma'] = f.zomma(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], volsurface['Gamma'])

	    fig = plt.figure(figsize=(10, 8))

	    axs = plt.axes(projection="3d")

	    axs.plot_trisurf(volsurface['Strike'], volsurface['DTE'], volsurface[greek], cmap=cm.jet)

	    axs.view_init(self.angle1, self.angle2)

	    # add labels
	    plt.xlabel("Strike"),
	    plt.ylabel("DTE"),
	    plt.title(f"{greek} Surface for {self.ticker}"),
	    plt.show()

	    return volsurface

    def skew(self, dte, Otype):
    	volsurface = od.volsurfacedata(self.ticker, Otype)
    	dtes = volsurface['DTE'].unique()
    	skew = volsurface.groupby('DTE').get_group(dtes[dte])

    	otm = skew.iloc[(int(len(skew) * .25))]['IV']
    	atm = skew.iloc[(int(len(skew) * .5))]['IV']
    	itm = skew.iloc[(int(len(skew) * .75))]['IV']

    	ivseries = pd.Series([skew['IV'].iloc[0], skew['IV'].iloc[1], skew['IV'].iloc[2],
    		skew['IV'].iloc[(int(len(skew) * .4))], skew['IV'].iloc[(int(len(skew) * .45))], skew['IV'].iloc[(int(len(skew) * .5))], skew['IV'].iloc[(int(len(skew) * .55))], skew['IV'].iloc[(int(len(skew) * .6))],
    		skew['IV'].iloc[-3], skew['IV'].iloc[-2], skew['IV'].iloc[-1]])

    	x = np.linspace(skew['Strike'].min(), skew['Strike'].max(), len(ivseries))
    	regy = np.polyfit(x, ivseries, deg=3)
    	evy = np.polyval(regy, x)

    	plt.figure(figsize=(8, 5))
    	plt.plot(x, evy)

    	plt.xlabel('Strike')
    	plt.ylabel('IV')
    	plt.title(f"IV Skew {dtes[dte]} Days Till Exp")
    	plt.show()
    	return skew

    def deltaskew(self, dte, lb, ub):

    	spot = f.getprice(self.ticker)

    	skew = f.getdeltaskew(self.ticker, spot, dte, lb, ub)

    	return skew

    def termstructure(self, strike, Otype):
    	spot = f.getprice(self.ticker)

    	volsurface = od.volsurfacedata(self.ticker, Otype)

    	term = volsurface.groupby('Strike').get_group(strike)

    	front = term.iloc[(int(len(term) * .25))]['IV']

    	middle = term.iloc[(int(len(term) * .5))]['IV']

    	back = term.iloc[(int(len(term) * .75))]['IV']

    	termseries = pd.DataFrame([term['IV'].iloc[0], term['IV'].iloc[1], term['IV'].iloc[2], term['IV'].iloc[3], term['IV'].iloc[4],
    		front, middle, back,
    		term['IV'].iloc[-4], term['IV'].iloc[-3], term['IV'].iloc[-2], term['IV'].iloc[-1]])
    	x = np.linspace(term['DTE'].min(), term['DTE'].max(), len(term['IV']))
    	regy = np.polyfit(term['DTE'], term['IV'], deg=2)
    	evy = np.polyval(regy, term['DTE'])

    	plt.figure(figsize=(8, 5))
    	plt.plot(term['DTE'], evy)

    	plt.xlabel('DTE')
    	plt.ylabel('IV')
    	plt.title("IV Term Structure")
    	plt.show()

    	return term

    def rvolcone(self, rvol, start=2021, end='today'):

    	if rvol == 'Close To Close':

    		return rcone.closecone(self.ticker, start, end)

    	elif rvol == 'Parkinson':

    		return rcone.parkinsoncone(self.ticker, start, end)

    	elif rvol == 'Rodgers Satchell':

    		return rcone.rodgersatchellcone(self.ticker, start, end)

    	elif rvol == 'Garman Klass':

    		return rcone.garmanklasscone(self.ticker, start, end)

    	elif rvol == 'Yang Zang':

    		return rcone.yangzangcone(self.ticker, start, end)

    	elif rvol == 'ewma':

    		return rcone.ewmacone(self.ticker, start, end)

    	elif rvol == 'all':

    		rcone.yangzangcone(self.ticker, start, end)

    		rcone.garmanklasscone(self.ticker, start, end)

    		rcone.rodgersatchellcone(self.ticker, start, end)

    		rcone.parkinsoncone(self.ticker, start, end)

    		rcone.closecone(self.ticker, start, end)

    		return

    def rvolplot(self, rvol, start=2021, end='today', window=30, overlay=False):

    	if rvol == 'all':

    		rvoldata = od.rvoldata(self.ticker, start, end, window)

    		rvoldata['Close To Close'].plot()

    		rvoldata['Parkinson'].plot()

    		rvoldata['Rodger Satchell'].plot()

    		rvoldata['Garman Klass'].plot()

    		rvoldata['Yang Zang'].plot()

    		plt.title(f'{window} Day Realized Volatility')

    		plt.legend()

    		return rvoldata

    	else:

    		if overlay == False:

	    		rvoldata = od.rvoldata(self.ticker, start, end, window)

	    		rvoldata[rvol].plot()
	    		plt.title(f'{rvol} {window} Day Volatility')
	    		plt.ylabel

	    		return rvoldata[rvol]
	    	elif overlay == True:

	    		rvoldata = od.rvoldata(self.ticker, start, end, window)

	    		fig, ax1 = plt.subplots(figsize=(10, 5))

	    		ax2 = ax1.twinx()

	    		ax1.set_xticklabels(rvoldata.index, rotation=45)

	    		ax1.plot(rvoldata.index, rvoldata[rvol], 'r--')

	    		ax2.plot(rvoldata.index, rvoldata['Adj Close'])

	    		return
    def varianceratio(self, start=2021, end='today', window=30, overlay=False):

    	rvoldata = od.rvoldata(self.ticker, start, end, window)

    	if overlay == False:
    		n = rvoldata['Yang Zang']
    		d = rvoldata['Close To Close']
    		y = n / d
    		x = rvoldata.index
    		plt.figure(figsize=(9, 4))
    		plt.xticks(rotation=45)

    		return plt.plot(x, y)
    	elif overlay == True:
    		n = rvoldata['Yang Zang']
    		d = rvoldata['Close To Close']
    		y = n / d
    		x = rvoldata.index

    		fig, ax1 = plt.subplots(figsize=(9, 4))
    		ax2 = ax1.twinx()

    		ax1.set_xticklabels(x, rotation=45)

    		ax1.plot(x, y, 'r--')

    		ax2.plot(x, rvoldata['Adj Close'])

    		return rvoldata