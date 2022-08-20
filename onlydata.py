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

def optiondata(ticker, Otype):
    
    stock = yf.Ticker(ticker)

    Maturity = list(stock.options)

            # get current date
    today = datetime.now().date()

            # empty list for days to expiration
    DTE = []

            # empty list to store data for calls
    Data = []

            # loop over maturities
    for maturity in Maturity:
                # maturity date
        maturity_date = datetime.strptime(maturity, '%Y-%m-%d').date()

                # DTE: differrence between maturity date and today
        DTE.append((maturity_date - today).days)

                # store call data
        if Otype == 'calls':

            Data.append(stock.option_chain(maturity).calls)

        elif Otype == 'puts':

            Data.append(stock.option_chain(maturity).puts)
    
    Strike = []
    DTE_extended = []
    ImpVol = []
    Premium = []

    for i in range(0, len(Data)):
            # append strikes to list
        Strike.append(Data[i]['strike'])

                # repeat dte so the list has the same length as the other lists
        DTE_extended.append(np.repeat(DTE[i], len(Data[i])))

                # append IVs to list
        ImpVol.append(Data[i]["impliedVolatility"])

                #Data[i]['Premium'] = (Data[i]['bid'] + Data[i]['ask']) / 2
        Premium.append(Data[i]['lastPrice'])

            # unlist list of lists
    Strike = list(chain(*Strike))
    DTE_extended = list(chain(*DTE_extended))
    ImpVol = list(chain(*ImpVol))       
    Premium = list(chain(*Premium))

    volsurface = pd.DataFrame({'Strike': Strike,
                'DTE': DTE_extended,
                'IV': ImpVol,
                'Premium': Premium})

    return volsurface

def volsurfacedata(ticker, Otype):

	volsurface = optiondata(ticker, Otype)

	volsurface = volsurface.loc[(volsurface['DTE'] > 1)&(volsurface['DTE'] < 356)]

	spot = f.getprice(ticker)

	volsurface = volsurface.loc[(volsurface['Premium'] > 0)]

	volsurface = volsurface.loc[(volsurface['IV'] > .05) & (volsurface['IV'] < 1)]

	if spot >= 300:

		volsurface = volsurface.loc[(volsurface['Strike'] > spot - (spot * .1))&(volsurface['Strike'] < spot + (spot * .1))]
		#volsurface = volsurface.fillna(method='ffill')

		return volsurface

	elif spot < 300 and spot > 50:
		volsurface = volsurface.loc[(volsurface['Strike'] > spot - (spot * .2))&(volsurface['Strike'] < spot + (spot * .2))]

		return volsurface

	elif spot <= 50:

		volsurface = volsurface.loc[(volsurface['Strike'] > spot - (spot * .5))&(volsurface['Strike'] < spot + (spot * .5))]
		#volsurface = volsurface.fillna(method='ffill')

		return volsurface

def greekdata(ticker, Otype):

	volsurface = volsurfacedata(ticker, Otype)

	spot = f.getprice(ticker)

	volsurface['Delta'] = f.deltafunc(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], Otype)

	volsurface['Volga'] = f.volga(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	volsurface['Vanna'] = f.vanna(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	volsurface['Vega'] = f.vega(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	volsurface['ultima'] = f.ultima(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], volsurface['Vega'])

	volsurface['color'] = f.color(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	volsurface['Veta'] = f.veta(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], volsurface['Vega'])

	volsurface['Charm'] = f.charm(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], Otype)

	volsurface['Theta'] = f.theta(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], Otype)

	volsurface['Speed'] = f.speed(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	volsurface['Gamma'] = f.gamma(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'])

	volsurface['Zomma'] = f.zomma(spot, volsurface['Strike'], volsurface['DTE']/260, .03, volsurface['IV'], volsurface['Gamma'])

	return volsurface

def rvoldata(ticker, start, end,  window, trading_days = 252, clean=True):

    #start = dt.datetime(2021,1,1)
    #end = dt.datetime.now()
    #ticker = 'spy'
    def stock_data(ticker, start, end):

        if end == 'today':
            dataframe = pd.DataFrame(pdr(ticker, dt.datetime(start, 1, 1), dt.datetime.now()))
            
        else:
            dataframe = pd.DataFrame(pdr(ticker, dt.datetime(start, 1, 1), dt.datetime(end, 1, 1)))
        
        return(dataframe)
    
    
    df = stock_data(ticker, start, end)

    df['log_return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))**2

    df['logHL'] = np.log(df['High']/df['Low'])**2

    df['logCO'] = np.log(df['Close']/df['Open'])**2

    df['logOC'] = np.log(df['Open']/df['Close'].shift(1))**2

    df['logHC'] = np.log(df['High']/df['Close'])

    df['logHO'] = np.log(df['High']/df['Open'])

    df['logLC'] = np.log(df['Low']/df['Close'])

    df['logLO'] = np.log(df['Low']/df['Open'])
    
    df['pk'] = (1/(4 * np.log(2))) * (np.log(df['High']/df['Low'])) ** 2
    
    df['rs'] = (df['logHC']*df['logHO']) + (df['logLC']*df['logLO'])
    
    df['Rodger Satchell'] = (np.sqrt((1/window) * df['rs'].rolling(window).sum()) * np.sqrt(trading_days)) * 100
    
    df['Parkinson'] = (np.sqrt(252*df['pk'].rolling(window).mean())) * 100
    
    df['Garman Klass'] = (np.sqrt((1/window) * ((1/2) * df['logHL'] - (2 * np.log(2) - 1) * df['logCO']).rolling(window).sum()) * (np.sqrt(trading_days))) * 100
    
    df['Yang Zang'] = (np.sqrt((1/window)*(df['logOC'] + (1/2)*df['logHL'] - (2*np.log(2) - 1)*df['logCO']).rolling(window).sum()) * np.sqrt(trading_days)) * 100
    
    df['Close To Close'] = (np.sqrt(trading_days) * np.sqrt((1/(window - 1)) * df['log_return'].rolling(window).sum())) * 100

    df['Log return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))

    lm = .94
    sig2 = df['Close To Close'] ** 2
    u = df['Log return']
    um = df['Log return'].mean()
    df['ewma'] = np.sqrt(lm * sig2 + (1 - lm) * (u - um) ** 2)

    return df