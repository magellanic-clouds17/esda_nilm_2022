# ESDA NILM 2022 Kaggle Competition Project

## Project Overview
This repository contains my work for the ESDA NILM (Non-Intrusive Load Monitoring) 2022 Kaggle competition, where the objective was to classify appliances based on energy consumption data. The project showcases a blend of data cleaning, exploratory analysis, feature engineering, and machine learning using XGBoost. The final submission achieved a Kaggle score of 0.96194, potentially placing first in the competition (second place score: 0.91).

## Files in the Repository
- `environment.yml`: Defines the project's Python environment and dependencies.
- `eda.py`: Script for exploratory data analysis and initial data cleaning.
- `build_features.py` & `build_features_v2.py`: Scripts for feature engineering, including handling multicollinearity,outlier removal, and temporal data utilization.
- `train_predict_model.py` & `train_predict_model_v2.py`: Model training and prediction scripts using XGBoost, with different iterations and parameter tuning.

## Key Insights
- Rigorous exploratory analysis and cleaning were pivotal in understanding the dataset.
- Feature engineering focused on reducing multicollinearity and leveraging timestamp data.
- Model tuning in XGBoost played a crucial role in improving the prediction accuracy from 0.82 to 0.96 Kaggle score.

## Usage
To replicate the analysis or use the scripts, clone the repository and set up the environment using the provided `environment.yml` file. Each Python script is documented for ease of understanding and modification.

For detailed analysis and results, please refer to individual scripts and comments within the repository.

## Results

### Current Score on Kaggle Competition 
![image](https://github.com/magellanic-clouds17/esda_nilm_2022/assets/72970703/c399200e-3f73-4934-a4a3-bc42136fbf92)

### Leaderboard (for comparison)
![image](https://github.com/magellanic-clouds17/esda_nilm_2022/assets/72970703/01fb6ae9-e948-49e1-bf89-5ae9f943979d)

### Final Feature Importance
![image](https://github.com/magellanic-clouds17/esda_nilm_2022/assets/72970703/71a222d5-903a-4593-b8a2-511d4302030d)

---
