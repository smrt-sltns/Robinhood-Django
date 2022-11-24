import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pandas_datareader as data
from datetime import date

stock =  'AAPL'
start_date = '2010-01-01' 
end_date = date.today() 

df = data.DataReader( stock, 'yahoo', start_date, end_date )
df = df.drop(['Adj Close'], axis = 1)

