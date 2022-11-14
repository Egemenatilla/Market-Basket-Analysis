# Impoting related libraries
import numpy as np
import pandas as pd 
import time, warnings
import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import pickle
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Reading the data from the file and copying it so that the raw leather does not deteriorate
filename = pd.read_csv('data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})
df=filename.copy()
print(df)
print(df['Country'].unique())

#%%
print(df[df['Country'] == 'United Kingdom'].shape)
print(df[df['Country'] == 'France'].shape)
print(df[df['Country'] == 'Germany'].shape)

#%% Cleaning
df['Description'] = df['Description'].str.strip()
print(df.head())

df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]
print(df.head())

#%%
basket = (df[df['Country'] =="France"].groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
print(basket)

#%% There are too many 0's in the data. We wrote this function to make sure that positive values ​​are separated as 1 and negative values ​​as 0.

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)


# 0 not received, 1 received by customers
print(basket_sets)

#%%
# Frequent items
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

# Support , Lift, Confidence
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules.head())

#%% We see the matches here, the important thing here is the support, confidence and lift values.
# The SUPPORT dual comparison value represents the confidence value, and the fact that the Lift is greater than 1 serves to get healthy results.

print(rules[ (rules['lift'] >= 6) & (rules['confidence'] >= 0.8) ])

