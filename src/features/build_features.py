import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# read vw_train_clean.csv from r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\interim\vw_train_clean.csv'
train_data_path = r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\interim\vw_train_clean.csv'
df = pd.read_csv(train_data_path, index_col=0, dtype={'id': int})


# identify features with high correlation (absolute value > 0.8) and print them out
corr_matrix = df.corr().abs()
high_corr_var=np.where(corr_matrix>0.8)
high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
high_corr_var

# write function that takes in df and returns sorted vif
def get_vif(df):
    vif = pd.DataFrame()
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif['features'] = df.columns
    vif.sort_values(by='VIF', ascending=False, inplace=True)
    return vif

# determine VIF (variance inflation factor) to check for multicollinearity using get_vif function
vif = get_vif(df)
vif


## step by step remove features with high VIF (> 10) and check VIF again

# make new df_low_corr with all features with vif > 1000 removed('hertz', 'transient6', 'apparentPower', 'transient5', 'transient4','current', 'transient9', 'transient7', 'transient3', 'transient8')

df_low_corr = df.drop(['hertz', 'transient6', 'apparentPower', 'transient5', 'transient4','current', 'transient9', 'transient7', 'transient3', 'transient8'], axis=1)
df_low_corr.info()

# check VIF again
vif = get_vif(df_low_corr)

# remove 'transient2' and 'transient1' from df_low_corr
df_low_corr.drop(['transient2', 'transient1'], axis=1, inplace=True)

# check VIF again
vif = get_vif(df_low_corr)
vif