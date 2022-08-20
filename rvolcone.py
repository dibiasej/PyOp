from pandas_datareader import get_data_yahoo as pdr
from PyOp.PyOp import onlydata as od
from PyOp.PyOp import functions as f
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def closecone(ticker, start, end):

	df30 = od.rvoldata(ticker, start, end, 30).describe()
	df60 = od.rvoldata(ticker, start, end, 60).describe()
	df90 = od.rvoldata(ticker, start, end, 90).describe()
	df120 = od.rvoldata(ticker, start, end, 120).describe()
	ctc30 = pd.Series(df30['Close To Close']).reset_index()
	ctc60 = pd.Series(df60['Close To Close']).reset_index()
	ctc90 = pd.Series(df90['Close To Close']).reset_index()
	ctc120 = pd.Series(df120['Close To Close']).reset_index()
	ctcdf = pd.DataFrame({
	    'ctc30': ctc30['Close To Close'],
	    'ctc60': ctc60['Close To Close'],
	    'ctc90': ctc90['Close To Close'],
	    'ctc120': ctc120['Close To Close']
	})
	ctcdf.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
	ctcdf = ctcdf.iloc[3:]
	ctcdf = ctcdf.T
	x = [30, 60, 90, 120]
	y1 = ctcdf['min']
	y2 = ctcdf['25%']
	y3 = ctcdf['50%']
	y4 = ctcdf['75%']
	y5 = ctcdf['max']
	plt.figure(figsize = (12, 8))
	plt.xlabel('Time(days) ' + f'{start} - {end}')
	plt.ylabel('Annualized Volatility')
	plt.title('Close to Close Volatility Cone')
	Min = plt.plot(x, y1, label='Min')
	quartile25 = plt.plot(x, y2, label='25%')
	quartile50 = plt.plot(x, y3, label='50%')
	quartile75 = plt.plot(x, y4, label='75%')
	Max = plt.plot(x, y5, label='Max')
	plt.legend()
	return(ctcdf)

def parkinsoncone(ticker, start, end):
	parkinson30 = od.rvoldata(ticker, start, end, 30).describe()
	parkinson60 = od.rvoldata(ticker, start, end, 60).describe()
	parkinson90 = od.rvoldata(ticker, start, end, 90).describe()
	parkinson120 = od.rvoldata(ticker, start, end, 120).describe()
	parkinson30 = pd.Series(parkinson30['Parkinson']).reset_index()
	parkinson60 = pd.Series(parkinson60['Parkinson']).reset_index()
	parkinson90 = pd.Series(parkinson90['Parkinson']).reset_index()
	parkinson120 = pd.Series(parkinson120['Parkinson']).reset_index()
	parkinsondf = pd.DataFrame({
		'parkinson30': parkinson30['Parkinson'],
		'parkinson60': parkinson60['Parkinson'],
		'parkinson90': parkinson90['Parkinson'],
		'parkinson120': parkinson120['Parkinson']
		})
	parkinsondf.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
	parkinsondf = parkinsondf.iloc[3:]
	parkinsondf = parkinsondf.T
	x = [30, 60, 90, 120]
	y1 = parkinsondf['min']
	y2 = parkinsondf['25%']
	y3 = parkinsondf['50%']
	y4 = parkinsondf['75%']
	y5 = parkinsondf['max']
	plt.figure(figsize = (12, 8))
	plt.xlabel('Time(days) ' + f'{start} - {end}')
	plt.ylabel('Annualized Volatility')
	plt.title('Parkinson Volatility Cone')
	Min = plt.plot(x, y1, label = 'Min')
	quartile25 = plt.plot(x, y2, label = '25%')
	quartile50 = plt.plot(x, y3, label = '50%')
	quartile75 = plt.plot(x, y4, label = '75%')
	Max = plt.plot(x, y5, label = 'Max')
	plt.legend()
	return(parkinsondf)

def rodgersatchellcone(ticker, start, end):

    rs30 = od.rvoldata(ticker, start, end, 30).describe()
    rs60 = od.rvoldata(ticker, start, end, 60).describe()
    rs90 = od.rvoldata(ticker, start, end, 90).describe()
    rs120 = od.rvoldata(ticker, start, end, 120).describe()
    rs30 = pd.Series(rs30['Rodger Satchell']).reset_index()
    rs60 = pd.Series(rs60['Rodger Satchell']).reset_index()
    rs90 = pd.Series(rs90['Rodger Satchell']).reset_index()
    rs120 = pd.Series(rs120['Rodger Satchell']).reset_index()
    rsdf = pd.DataFrame({
        'rs30': rs30['Rodger Satchell'],
        'rs60': rs60['Rodger Satchell'],
        'rs90': rs90['Rodger Satchell'],
        'rs120': rs120['Rodger Satchell']
    })
    rsdf.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    rsdf = rsdf.iloc[3:]
    rsdf = rsdf.T
    x = [30, 60, 90, 120]
    y1 = rsdf['min']
    y2 = rsdf['25%']
    y3 = rsdf['50%']
    y4 = rsdf['75%']
    y5 = rsdf['max']
    plt.figure(figsize = (12, 8))
    plt.xlabel('Time(days) ' + f'{start} - {end}')
    plt.ylabel('Annualized Volatility')
    plt.title('Rodgers Satchell Volatility Cone')
    Min = plt.plot(x, y1, label = 'Min')
    quartile25 = plt.plot(x, y2, label='25%')
    quartile50 = plt.plot(x, y3, label='50%')
    quartile75 = plt.plot(x, y4, label='75%')
    Max = plt.plot(x, y5, label='Max')
    plt.legend()
    return(rsdf)

def garmanklasscone(ticker, start, end):

    gk30 = od.rvoldata(ticker, start, end, 30).describe()
    gk60 = od.rvoldata(ticker, start, end, 60).describe()
    gk90 = od.rvoldata(ticker, start, end, 90).describe()
    gk120 = od.rvoldata(ticker, start, end, 120).describe()
    gk30 = pd.Series(gk30['Garman Klass']).reset_index()
    gk60 = pd.Series(gk60['Garman Klass']).reset_index()
    gk90 = pd.Series(gk90['Garman Klass']).reset_index()
    gk120 = pd.Series(gk120['Garman Klass']).reset_index()
    gkdf = pd.DataFrame({
        'gk30': gk30['Garman Klass'],
        'gk60': gk60['Garman Klass'],
        'gk90': gk90['Garman Klass'],
        'gk120': gk120['Garman Klass']
    })
    gkdf.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    gkdf = gkdf.iloc[3:]
    gkdf = gkdf.T
    x = [30, 60, 90, 120]
    y1 = gkdf['min']
    y2 = gkdf['25%']
    y3 = gkdf['50%']
    y4 = gkdf['75%']
    y5 = gkdf['max']
    plt.figure(figsize = (12, 8))
    plt.xlabel('Time(days) ' + f'{start} - {end}')
    plt.ylabel('Annualized Volatility')
    plt.title('Garman Klass Volatility Cone')
    Min = plt.plot(x, y1, label='Min')
    quartile25 = plt.plot(x, y2, label='25%')
    quartile50 = plt.plot(x, y3, label='50%')
    quartile75 = plt.plot(x, y4, label='75%')
    Max = plt.plot(x, y5, label='Min')
    plt.legend()
    return(gkdf)

def yangzangcone(ticker, start, end):
	
    yz30 = od.rvoldata(ticker, start, end, 30).describe()
    yz60 = od.rvoldata(ticker, start, end, 60).describe()
    yz90 = od.rvoldata(ticker, start, end, 90).describe()
    yz120 = od.rvoldata(ticker, start, end, 120).describe()
    yz30 = pd.Series(yz30['Yang Zang']).reset_index()
    yz60 = pd.Series(yz60['Yang Zang']).reset_index()
    yz90 = pd.Series(yz90['Yang Zang']).reset_index()
    yz120 = pd.Series(yz120['Yang Zang']).reset_index()
    yzdf = pd.DataFrame({
        'yz30': yz30['Yang Zang'],
        'yz60': yz60['Yang Zang'],
        'yz90': yz90['Yang Zang'],
        'yz120': yz120['Yang Zang']
    })
    yzdf.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    yzdf = yzdf.iloc[3:]
    yzdf = yzdf.T
    x = [30, 60, 90, 120]
    y1 = yzdf['min']
    y2 = yzdf['25%']
    y3 = yzdf['50%']
    y4 = yzdf['75%']
    y5 = yzdf['max']
    plt.figure(figsize = (12, 8))
    plt.xlabel('Time(days) ' + f'{start} - {end}')
    plt.ylabel('Annualized Volatility')
    plt.title('Yang Zang Volatility Cone')
    Min = plt.plot(x, y1, label='Min')
    quartile25 = plt.plot(x, y2, label='25%')
    quartile50 = plt.plot(x, y3, label='50%')
    quartile75 = plt.plot(x, y4, label='75%')
    Max = plt.plot(x, y5, label='Max')
    plt.legend()
    return(yzdf)

def ewmacone(ticker, start, end):
	
    ewma30 = od.rvoldata(ticker, start, end, 30).describe()
    ewma60 = od.rvoldata(ticker, start, end, 60).describe()
    ewma90 = od.rvoldata(ticker, start, end, 90).describe()
    ewma120 = od.rvoldata(ticker, start, end, 120).describe()
    ewma30 = pd.Series(ewma30['ewma']).reset_index()
    ewma60 = pd.Series(ewma60['ewma']).reset_index()
    ewma90 = pd.Series(ewma90['ewma']).reset_index()
    ewma120 = pd.Series(ewma120['ewma']).reset_index()
    ewmadf = pd.DataFrame({
        'ewma30': ewma30['ewma'],
        'ewma60': ewma60['ewma'],
        'ewma90': ewma90['ewma'],
        'ewma120': ewma120['ewma']
    })
    ewmadf.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    ewmadf = ewmadf.iloc[3:]
    ewmadf = ewmadf.T
    x = [30, 60, 90, 120]
    y1 = ewmadf['min']
    y2 = ewmadf['25%']
    y3 = ewmadf['50%']
    y4 = ewmadf['75%']
    y5 = ewmadf['max']
    plt.figure(figsize = (12, 8))
    plt.xlabel('Time(days) ' + f'{start} - {end}')
    plt.ylabel('Annualized Volatility')
    plt.title('EWMA Volatility Cone')
    Min = plt.plot(x, y1, label='Min')
    quartile25 = plt.plot(x, y2, label='25%')
    quartile50 = plt.plot(x, y3, label='50%')
    quartile75 = plt.plot(x, y4, label='75%')
    Max = plt.plot(x, y5, label='Max')
    plt.legend()
    return(ewmadf)