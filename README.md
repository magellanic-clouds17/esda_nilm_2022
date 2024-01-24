# NILM ESDA 2022 Project (*ONGOING*)

## Project Overview
This project is an ongoing effort to use data frome the 2022 Kaggle ESDA NILM (Non-Intrusive Load Monitoring) competition to demonstrate my abilites in data science and machine learning. The challenge involves classifying appliances based on their energy consumption patterns using numerical time-series data. The goal is to develop a predictive model that can accurately determine the type of appliance from its unique energy usage profile.

## Data
The dataset is sourced from the "ESDA NILM 2022" Kaggle competition. It includes hourly measurements of electrical parameters such as active power, current, voltage, and various transient and harmonic features. The data spans from 2016 to 2018 and is used to predict appliance usage for the year 2019.

## Provisional Methodology
The approach for this project will follow these steps:
1. **Data Cleaning and Preprocessing**: Ensure data quality by handling missing values, removing duplicates, and addressing any anomalies.
2. **Exploratory Data Analysis (EDA)**: Conduct a thorough examination of the data to understand underlying patterns and relationships, using statistical summaries and visualizations.
3. **Feature Engineering**: Develop features that capture the temporal nature of the data, such as lag features, rolling window statistics, and extracted time-based attributes.
4. **Model Selection**: Evaluate different machine learning models, with an initial focus on XGBoost due to its robustness and effectiveness in handling numerical features for classification tasks.
5. **Model Training and Validation**: Implement a rigorous training and validation process to fine-tune the model and prevent overfitting.
6. **Performance Evaluation**: Use appropriate metrics to assess the model's predictive accuracy and its ability to generalize to unseen data.

---
