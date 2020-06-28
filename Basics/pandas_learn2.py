import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
df = pd.read_csv('ZILLOW-M1300_MPPRSF.csv')

print(df.head())

df.set_index('Date', inplace=True)

df.to_csv('new_csv.csv')
'''

##df = pd.read_csv('new_csv.csv', index_col=0)
##
###print(df.head())
##
##df.columns=['Austin_HPI']
##
##df.to_csv('new_csv2.csv')
##df.to_csv('new_csv3.csv', header=False)
##
##df = pd.read_csv('new_csv3.csv', names=['Date', 'Austin_HPI'], index_col=0)
##print(df.head())
##
##df.to_json('new_json.json')
##df.to_html('new_html.html')

df = pd.read_csv('new_csv3.csv', names=['Date', 'Austin_HPI'])

print(df.head())

df.rename(columns={'Austin_HPI':'77006_HPI'}, inplace=True)

print(df.head())
