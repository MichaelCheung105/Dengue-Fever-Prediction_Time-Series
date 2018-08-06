# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:46:20 2018

@author: lupus
"""

# import necessary packages
import pandas as pd
import numpy as np
import os 
import re
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE

# reading data
data = {}
for csv in os.listdir("C:\\Users\\lupus\\OneDrive\\GIT\\OwnProjects\\DengAI_Competition\\Data"):
    data[re.sub(".csv", "", csv)] = pd.read_csv("C:\\Users\\lupus\\OneDrive\\GIT\\OwnProjects\\DengAI_Competition\\Data\\" + csv)

training_X = data.pop("DengAI_Predicting_Disease_Spread_-_Training_Data_Features")
training_Y = data.pop("DengAI_Predicting_Disease_Spread_-_Training_Data_Labels")
submit_X = data.pop("DengAI_Predicting_Disease_Spread_-_Test_Data_Features")
submit_Y = data.pop("DengAI_Predicting_Disease_Spread_-_Submission_Format")

# joining data for train_test_split in the later stage
training = training_X.merge(training_Y, on=['city', 'year', 'weekofyear'], how='left')
submit = submit_X.merge(submit_Y, on=['city', 'year', 'weekofyear'], how='left')

# set datetime as index
training.index = pd.DatetimeIndex(training.week_start_date)
submit.index = pd.DatetimeIndex(submit.week_start_date)

# generate datetime features
training['week_start_date'] = pd.to_datetime(training['week_start_date'], format='%Y-%m-%d')
training['quarter'] = training.week_start_date.dt.quarter
training['month'] = training.week_start_date.dt.month
training['day'] = training.week_start_date.dt.day
training['month-day'] = training['month'].astype(str) + "-" + training['day'].astype(str)

submit['week_start_date'] = pd.to_datetime(submit['week_start_date'], format='%Y-%m-%d')
submit['quarter'] = submit.week_start_date.dt.quarter
submit['month'] = submit.week_start_date.dt.month
submit['day'] = submit.week_start_date.dt.day
submit['month-day'] = submit['month'].astype(str) + "-" + submit['day'].astype(str)

# Converstion of categorical variables
cat_features = ['city', 'month-day']
for feature in cat_features:
    training[feature] = training[feature].astype('category')
    submit[feature] = submit[feature].astype('category')
    
# Check if there is any NA and then fill NA
training.isnull().sum()
training.groupby(['city'], as_index=False)['ndvi_ne'].fillna('bfill')

# Explore correlation between variables
sns.heatmap(training.corr(), xticklabels=training.corr().columns, yticklabels=training.corr().columns, center=0)

# drop week_start_date before doing train_test_split
training.drop('week_start_date', axis=1, inplace=True)
submit.drop('week_start_date', axis=1, inplace=True)

# train test split for train, test and cv
X_train, X_test, y_train, y_test = train_test_split(training.drop(['total_cases'], axis=1), training.total_cases, test_size=0.2, stratify=training.city, random_state=123)
X_train_cv, X_cv, y_train_cv, y_cv = train_test_split(X_train, y_train, test_size=0.25, stratify=X_train.city, random_state=123)

