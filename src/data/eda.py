import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read vw_train.csv
train_data_path = r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\raw\vw_train.csv'
df = pd.read_csv(train_data_path, index_col=False, dtype={'id': int})

## EXPLORATORY DATA ANALYSIS AND DATA CLEANING

# convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# remove unnamed column
df.drop('Unnamed: 0', axis=1, inplace=True)

# sort index by timestamp
df.sort_index(inplace=True)

# check shape of df, number of rows and columns, unique values, missing values, data types, etc.

df.head()
df.tail()
df.shape 
df.columns
df.info() #  all desriptors are numeric
df.describe()

# unique appliances
df['appliances'].unique() # nan 
len(df['appliances'].unique())

# check every column for missing values
df.isnull().sum() # appliances has 20 nan values, all other have no missing values

# drop nan values
df.dropna(inplace=True)

# check for duplicates
df.duplicated().sum()

# show duplicates
df[df.duplicated(keep=False)]

# remove duplicates
df.drop_duplicates(inplace=True)

# remove target variable and save it as a series to be added back later
appliances = df['appliances']
df.drop('appliances', axis=1, inplace=True)

# remove id column and store as series in case needed later
id_col = df['id']
df.drop('id', axis=1, inplace=True)

### check collinearity to understand relationship and potential multicollinearity between features
df_corr = df.corr()
df_corr

# plot correlation matrix
plt.figure(figsize=(20,20))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.show() 
    # many features have high correlation with each other
    # will have to do vif analysis to check for multicollinearity in build_features.py

# add target variable and id back to df
df['appliances'] = appliances
df['id'] = id_col

df.info()

# save df as csv
df.to_csv(r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\interim\vw_train_clean.csv', index=True)





