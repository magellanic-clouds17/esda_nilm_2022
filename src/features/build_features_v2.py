import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


# read vw_train_clean.csv from r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\interim\vw_train_clean.csv'
train_data_path = r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\interim\vw_train_clean.csv'
df = pd.read_csv(train_data_path, index_col=0, dtype={'id': int})

# convert timestamp index to datetime
df.index = pd.to_datetime(df.index)

## build features out of timestamp (year, month, day, day of week, hour, minute)
df.index
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['day_of_week'] = df.index.dayofweek
df['hour'] = df.index.hour
df['minute'] = df.index.minute

# remove target variable and id save it as a series to be added back later
appliances = df[['appliances']]
df.drop('appliances', axis=1, inplace=True)
id_col = df[['id']]
df.drop('id', axis=1, inplace=True)

# ## remove outliers using IQR
# calculate IQR for each feature
Q1 = df.quantile(0.05)
Q3 = df.quantile(0.95)
IQR = Q3 - Q1

# remove outliers from df using IQR
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)] # tested with boxplots, it works as intended

# left merge id and appliances columns back to df on index
id_col = id_col[~id_col.index.duplicated(keep='first')]
appliances = appliances[~appliances.index.duplicated(keep='first')]

df = df.merge(id_col, how='left', left_index=True, right_index=True)
df = df.merge(appliances, how='left', left_index=True, right_index=True)

# remove featrues with high VIF (determined in build_features.py)

df = df.drop(['hertz', 'transient6', 'apparentPower', 'transient5', 'transient4','current', 'transient9', 'transient7', 'transient3', 'transient8','transient2', 'transient1'], axis=1)

# save df as df_outlier.csv in data/processed
df.to_csv(r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\processed\df_multico_outlier.csv')



