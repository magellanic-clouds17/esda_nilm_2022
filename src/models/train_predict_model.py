import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

''' In this version I only removed features with high VIF (multicolinarity)
    Submission_v1 and v1_1
    Kaggle result: 0.82933 and 0.93260 (best so far)
'''

# read in low_corr_train.csv from data/processed with index int and index_col=False
train_data_path = r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\processed\df_low_corr.csv'
df = pd.read_csv(train_data_path, index_col=0)

# convert timestamp index to datetime
df.index = pd.to_datetime(df.index)

# remove id and save it as a series to be added back later
id_col_train = df['id']
df.drop('id', axis=1, inplace=True)

# x, y split
X_train = df.drop('appliances', axis=1)
y_train = df[['appliances']]

X_train.info()
y_train.info()

# encode y_train['appliances'] and add it back to y_train
le = LabelEncoder()
y_train['appliances'] = le.fit_transform(y_train['appliances'])

## prepare test data for prediction
# # load in test data from data/raw/vw_test.csv
test_data_path = r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\raw\vw_test.csv'
df_test = pd.read_csv(test_data_path)
df_test

# drop unnamed column, drop id column (save as series), convert timestamp to datetime (sort descending) and set as index
df_test.drop('Unnamed: 0', axis=1, inplace=True)

df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
df_test.set_index('timestamp', inplace=True)
df_test.sort_index(ascending=True, inplace=True)

id_col_test = df_test['id']
df_test.drop('id', axis=1, inplace=True)

# from timestamp create year, month, day, day_of_week, hour, minute like in train data
df_test['year'] = df_test.index.year
df_test['month'] = df_test.index.month
df_test['day'] = df_test.index.day
df_test['day_of_week'] = df_test.index.dayofweek
df_test['hour'] = df_test.index.hour
df_test['minute'] = df_test.index.minute

# test which columns exist in df_test that dont exist in df, save the column names in a list than drop the list from test
df_test.columns.difference(df.columns)
l = ['apparentPower', 'current', 'hertz', 'transient1', 'transient2',
    'transient3', 'transient4', 'transient5', 'transient6', 'transient7',
    'transient8', 'transient9']
df_test.drop(l, axis=1, inplace=True)	

# name df_test as X_test
X_test = df_test

## set up xgb model for classification of appliances
# review X_train y_train and X_test
print(X_train.info())
print(y_train.info())
print(X_test.info())

# instantiate xgb model
num_classes = np.unique(y_train.values).shape[0]

'''
#model version 1
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class= num_classes, 
            max_depth=3, learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8)
'''
'''
# model version 1_1
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class= num_classes, 
            max_depth=3, learning_rate=0.1, n_estimators=500)
'''
'''
# model version 1_2
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class= num_classes, 
            max_depth=3, learning_rate=0.1, n_estimators=800)
'''
#model version 1_3
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class= num_classes, 
            max_depth=3, learning_rate=0.1, n_estimators=2500)

# fit model to training data
predictions = xgb_model.fit(X_train, y_train).predict(X_test)

# plot feature importance
xgb.plot_importance(xgb_model)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()

## PREPARE SUBMISSION FILE
# deencode predictions
predictions = le.inverse_transform(predictions)

# add deencoded predictions to X_test
X_test['appliances'] = predictions

# add id column back to X_test
X_test['id'] = id_col_test

# restructure X_test to match sample_submission.csv
X_test = X_test[['id', 'appliances']]
X_test.reset_index(inplace=True)
# drop timestamp column
X_test.drop('timestamp', axis=1, inplace=True)
X_test 


# read sample_submission.csv file and compare to X_test
raw_data_path_test = r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\raw\sample_submission.csv'
df_sample_submission = pd.read_csv(raw_data_path_test, index_col=False, dtype={'id': int})
df_sample_submission

# save X_test as submission_v1.csv
X_test.to_csv(r'C:\Users\Latitude\Desktop\Kaggle\esda_nilm_2022\data\processed\submission_v1_3.csv', index=False)



