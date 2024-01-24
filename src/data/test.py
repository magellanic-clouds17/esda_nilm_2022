import pandas as pd

# read sample_submission.csv file

raw_data_path_test = r'C:\Users\Latitude\Desktop\Kaggle\time_series_energy_portfolio_project\data\vw_test.csv'
raw_data_path_train = r'C:\Users\Latitude\Desktop\Kaggle\time_series_energy_portfolio_project\data\vw_train.csv'  

#read test data
df = pd.read_csv(raw_data_path_test, index_col=False, dtype={'id': int})
# add appliances column to test data with all values set to +fridge+tumble_dryer+washer_dryer+microwave as type object
df['appliances'] = '+fridge+tumble_dryer+washer_dryer+microwave'
df['appliances'].dtype 

df = df[['id', 'appliances']]
df.head()

# save df as vw_test_processed.csv
processed_data_path = r'C:\Users\Latitude\Desktop\Kaggle\time_series_energy_portfolio_project\data\vw_test_processed.csv'
df.to_csv(processed_data_path, index=False)