
# Predicting Dengue Fever Cases using weather data via Catboost Algorithm

### Introduction

This project aims to tackle the Dengue Fever Challenge posted on Driven Data.
Five steps are included in this markdown:
1. importing packages and reading data from source
2. conducting basic EDA (Exploration Data Analysis)
3. feature engineering
4. train a model and test result simply using training data
5. train the model again using all training data and do prediction for the 'submit' set

### Step 1: importing packages and reading data from source 


```python
# import packages
import pandas as pd
import numpy as np
import os 
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from catboost import CatBoostRegressor
```


```python
# reading data
path = os.getcwd() + "/Data/"

data = {}
for file in os.listdir(path):
    data[re.sub(".csv", "", file)] = pd.read_csv(path + file)

training_X = data.pop("DengAI_Predicting_Disease_Spread_-_Training_Data_Features")
training_Y = data.pop("DengAI_Predicting_Disease_Spread_-_Training_Data_Labels")
submit_X = data.pop("DengAI_Predicting_Disease_Spread_-_Test_Data_Features")
submit_Y = data.pop("DengAI_Predicting_Disease_Spread_-_Submission_Format")

# joining data for train_test_split in the later stage
training = training_X.merge(training_Y, on=['city', 'year', 'weekofyear'], how='left')
submit = submit_X.merge(submit_Y, on=['city', 'year', 'weekofyear'], how='left')
```

### Step 2: conduct basic EDA (Exploratory Data Analysis)


```python
# check the shape of training data
training.shape
```




    (1456, 25)




```python
# check the shape of submission data
submit.shape
```




    (416, 25)




```python
# take a look at the training data
training.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>year</th>
      <th>weekofyear</th>
      <th>week_start_date</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>...</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
      <th>total_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sj</td>
      <td>1990</td>
      <td>18</td>
      <td>1990-04-30</td>
      <td>0.122600</td>
      <td>0.103725</td>
      <td>0.198483</td>
      <td>0.177617</td>
      <td>12.42</td>
      <td>297.572857</td>
      <td>...</td>
      <td>73.365714</td>
      <td>12.42</td>
      <td>14.012857</td>
      <td>2.628571</td>
      <td>25.442857</td>
      <td>6.900000</td>
      <td>29.4</td>
      <td>20.0</td>
      <td>16.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sj</td>
      <td>1990</td>
      <td>19</td>
      <td>1990-05-07</td>
      <td>0.169900</td>
      <td>0.142175</td>
      <td>0.162357</td>
      <td>0.155486</td>
      <td>22.82</td>
      <td>298.211429</td>
      <td>...</td>
      <td>77.368571</td>
      <td>22.82</td>
      <td>15.372857</td>
      <td>2.371429</td>
      <td>26.714286</td>
      <td>6.371429</td>
      <td>31.7</td>
      <td>22.2</td>
      <td>8.6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sj</td>
      <td>1990</td>
      <td>20</td>
      <td>1990-05-14</td>
      <td>0.032250</td>
      <td>0.172967</td>
      <td>0.157200</td>
      <td>0.170843</td>
      <td>34.54</td>
      <td>298.781429</td>
      <td>...</td>
      <td>82.052857</td>
      <td>34.54</td>
      <td>16.848571</td>
      <td>2.300000</td>
      <td>26.714286</td>
      <td>6.485714</td>
      <td>32.2</td>
      <td>22.8</td>
      <td>41.4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sj</td>
      <td>1990</td>
      <td>21</td>
      <td>1990-05-21</td>
      <td>0.128633</td>
      <td>0.245067</td>
      <td>0.227557</td>
      <td>0.235886</td>
      <td>15.36</td>
      <td>298.987143</td>
      <td>...</td>
      <td>80.337143</td>
      <td>15.36</td>
      <td>16.672857</td>
      <td>2.428571</td>
      <td>27.471429</td>
      <td>6.771429</td>
      <td>33.3</td>
      <td>23.3</td>
      <td>4.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sj</td>
      <td>1990</td>
      <td>22</td>
      <td>1990-05-28</td>
      <td>0.196200</td>
      <td>0.262200</td>
      <td>0.251200</td>
      <td>0.247340</td>
      <td>7.52</td>
      <td>299.518571</td>
      <td>...</td>
      <td>80.460000</td>
      <td>7.52</td>
      <td>17.210000</td>
      <td>3.014286</td>
      <td>28.942857</td>
      <td>9.371429</td>
      <td>35.0</td>
      <td>23.9</td>
      <td>5.8</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
# take a look at the submission data
submit.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>year</th>
      <th>weekofyear</th>
      <th>week_start_date</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>...</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
      <th>total_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sj</td>
      <td>2008</td>
      <td>18</td>
      <td>2008-04-29</td>
      <td>-0.0189</td>
      <td>-0.018900</td>
      <td>0.102729</td>
      <td>0.091200</td>
      <td>78.60</td>
      <td>298.492857</td>
      <td>...</td>
      <td>78.781429</td>
      <td>78.60</td>
      <td>15.918571</td>
      <td>3.128571</td>
      <td>26.528571</td>
      <td>7.057143</td>
      <td>33.3</td>
      <td>21.7</td>
      <td>75.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sj</td>
      <td>2008</td>
      <td>19</td>
      <td>2008-05-06</td>
      <td>-0.0180</td>
      <td>-0.012400</td>
      <td>0.082043</td>
      <td>0.072314</td>
      <td>12.56</td>
      <td>298.475714</td>
      <td>...</td>
      <td>78.230000</td>
      <td>12.56</td>
      <td>15.791429</td>
      <td>2.571429</td>
      <td>26.071429</td>
      <td>5.557143</td>
      <td>30.0</td>
      <td>22.2</td>
      <td>34.3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sj</td>
      <td>2008</td>
      <td>20</td>
      <td>2008-05-13</td>
      <td>-0.0015</td>
      <td>NaN</td>
      <td>0.151083</td>
      <td>0.091529</td>
      <td>3.66</td>
      <td>299.455714</td>
      <td>...</td>
      <td>78.270000</td>
      <td>3.66</td>
      <td>16.674286</td>
      <td>4.428571</td>
      <td>27.928571</td>
      <td>7.785714</td>
      <td>32.8</td>
      <td>22.8</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sj</td>
      <td>2008</td>
      <td>21</td>
      <td>2008-05-20</td>
      <td>NaN</td>
      <td>-0.019867</td>
      <td>0.124329</td>
      <td>0.125686</td>
      <td>0.00</td>
      <td>299.690000</td>
      <td>...</td>
      <td>73.015714</td>
      <td>0.00</td>
      <td>15.775714</td>
      <td>4.342857</td>
      <td>28.057143</td>
      <td>6.271429</td>
      <td>33.3</td>
      <td>24.4</td>
      <td>0.3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sj</td>
      <td>2008</td>
      <td>22</td>
      <td>2008-05-27</td>
      <td>0.0568</td>
      <td>0.039833</td>
      <td>0.062267</td>
      <td>0.075914</td>
      <td>0.76</td>
      <td>299.780000</td>
      <td>...</td>
      <td>74.084286</td>
      <td>0.76</td>
      <td>16.137143</td>
      <td>3.542857</td>
      <td>27.614286</td>
      <td>7.085714</td>
      <td>33.3</td>
      <td>23.3</td>
      <td>84.1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
# check datatypes of each column
training.dtypes
```




    city                                      object
    year                                       int64
    weekofyear                                 int64
    week_start_date                           object
    ndvi_ne                                  float64
    ndvi_nw                                  float64
    ndvi_se                                  float64
    ndvi_sw                                  float64
    precipitation_amt_mm                     float64
    reanalysis_air_temp_k                    float64
    reanalysis_avg_temp_k                    float64
    reanalysis_dew_point_temp_k              float64
    reanalysis_max_air_temp_k                float64
    reanalysis_min_air_temp_k                float64
    reanalysis_precip_amt_kg_per_m2          float64
    reanalysis_relative_humidity_percent     float64
    reanalysis_sat_precip_amt_mm             float64
    reanalysis_specific_humidity_g_per_kg    float64
    reanalysis_tdtr_k                        float64
    station_avg_temp_c                       float64
    station_diur_temp_rng_c                  float64
    station_max_temp_c                       float64
    station_min_temp_c                       float64
    station_precip_mm                        float64
    total_cases                                int64
    dtype: object




```python
# check summary of the training data
training.describe(include='all')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>year</th>
      <th>weekofyear</th>
      <th>week_start_date</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>...</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
      <th>total_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1456</td>
      <td>1456.000000</td>
      <td>1456.000000</td>
      <td>1456</td>
      <td>1262.000000</td>
      <td>1404.000000</td>
      <td>1434.000000</td>
      <td>1434.000000</td>
      <td>1443.000000</td>
      <td>1446.000000</td>
      <td>...</td>
      <td>1446.000000</td>
      <td>1443.000000</td>
      <td>1446.000000</td>
      <td>1446.000000</td>
      <td>1413.000000</td>
      <td>1413.000000</td>
      <td>1436.000000</td>
      <td>1442.000000</td>
      <td>1434.000000</td>
      <td>1456.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1049</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>sj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2003-07-23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>2001.031593</td>
      <td>26.503434</td>
      <td>NaN</td>
      <td>0.142294</td>
      <td>0.130553</td>
      <td>0.203783</td>
      <td>0.202305</td>
      <td>45.760388</td>
      <td>298.701852</td>
      <td>...</td>
      <td>82.161959</td>
      <td>45.760388</td>
      <td>16.746427</td>
      <td>4.903754</td>
      <td>27.185783</td>
      <td>8.059328</td>
      <td>32.452437</td>
      <td>22.102150</td>
      <td>39.326360</td>
      <td>24.675137</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>5.408314</td>
      <td>15.019437</td>
      <td>NaN</td>
      <td>0.140531</td>
      <td>0.119999</td>
      <td>0.073860</td>
      <td>0.083903</td>
      <td>43.715537</td>
      <td>1.362420</td>
      <td>...</td>
      <td>7.153897</td>
      <td>43.715537</td>
      <td>1.542494</td>
      <td>3.546445</td>
      <td>1.292347</td>
      <td>2.128568</td>
      <td>1.959318</td>
      <td>1.574066</td>
      <td>47.455314</td>
      <td>43.596000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>1990.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.406250</td>
      <td>-0.456100</td>
      <td>-0.015533</td>
      <td>-0.063457</td>
      <td>0.000000</td>
      <td>294.635714</td>
      <td>...</td>
      <td>57.787143</td>
      <td>0.000000</td>
      <td>11.715714</td>
      <td>1.357143</td>
      <td>21.400000</td>
      <td>4.528571</td>
      <td>26.700000</td>
      <td>14.700000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>1997.000000</td>
      <td>13.750000</td>
      <td>NaN</td>
      <td>0.044950</td>
      <td>0.049217</td>
      <td>0.155087</td>
      <td>0.144209</td>
      <td>9.800000</td>
      <td>297.658929</td>
      <td>...</td>
      <td>77.177143</td>
      <td>9.800000</td>
      <td>15.557143</td>
      <td>2.328571</td>
      <td>26.300000</td>
      <td>6.514286</td>
      <td>31.100000</td>
      <td>21.100000</td>
      <td>8.700000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>2002.000000</td>
      <td>26.500000</td>
      <td>NaN</td>
      <td>0.128817</td>
      <td>0.121429</td>
      <td>0.196050</td>
      <td>0.189450</td>
      <td>38.340000</td>
      <td>298.646429</td>
      <td>...</td>
      <td>80.301429</td>
      <td>38.340000</td>
      <td>17.087143</td>
      <td>2.857143</td>
      <td>27.414286</td>
      <td>7.300000</td>
      <td>32.800000</td>
      <td>22.200000</td>
      <td>23.850000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>2005.000000</td>
      <td>39.250000</td>
      <td>NaN</td>
      <td>0.248483</td>
      <td>0.216600</td>
      <td>0.248846</td>
      <td>0.246982</td>
      <td>70.235000</td>
      <td>299.833571</td>
      <td>...</td>
      <td>86.357857</td>
      <td>70.235000</td>
      <td>17.978214</td>
      <td>7.625000</td>
      <td>28.157143</td>
      <td>9.566667</td>
      <td>33.900000</td>
      <td>23.300000</td>
      <td>53.900000</td>
      <td>28.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>2010.000000</td>
      <td>53.000000</td>
      <td>NaN</td>
      <td>0.508357</td>
      <td>0.454429</td>
      <td>0.538314</td>
      <td>0.546017</td>
      <td>390.600000</td>
      <td>302.200000</td>
      <td>...</td>
      <td>98.610000</td>
      <td>390.600000</td>
      <td>20.461429</td>
      <td>16.028571</td>
      <td>30.800000</td>
      <td>15.800000</td>
      <td>42.200000</td>
      <td>25.600000</td>
      <td>543.300000</td>
      <td>461.000000</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 25 columns</p>
</div>




```python
# check summary of the submission data
submit.describe(include='all')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>year</th>
      <th>weekofyear</th>
      <th>week_start_date</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>...</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
      <th>total_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>416</td>
      <td>416.000000</td>
      <td>416.000000</td>
      <td>416</td>
      <td>373.000000</td>
      <td>405.000000</td>
      <td>415.000000</td>
      <td>415.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>...</td>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>413.000000</td>
      <td>407.000000</td>
      <td>411.000000</td>
      <td>416.0</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>269</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>sj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011-07-23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>260</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>2010.766827</td>
      <td>26.439904</td>
      <td>NaN</td>
      <td>0.126050</td>
      <td>0.126803</td>
      <td>0.207702</td>
      <td>0.201721</td>
      <td>38.354324</td>
      <td>298.818295</td>
      <td>...</td>
      <td>82.499810</td>
      <td>38.354324</td>
      <td>16.927088</td>
      <td>5.124569</td>
      <td>27.369587</td>
      <td>7.810991</td>
      <td>32.534625</td>
      <td>22.368550</td>
      <td>34.278589</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>1.434835</td>
      <td>14.978257</td>
      <td>NaN</td>
      <td>0.164353</td>
      <td>0.141420</td>
      <td>0.079102</td>
      <td>0.092028</td>
      <td>35.171126</td>
      <td>1.469501</td>
      <td>...</td>
      <td>7.378243</td>
      <td>35.171126</td>
      <td>1.557868</td>
      <td>3.542870</td>
      <td>1.232608</td>
      <td>2.449718</td>
      <td>1.920429</td>
      <td>1.731437</td>
      <td>34.655966</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>2008.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.463400</td>
      <td>-0.211800</td>
      <td>0.006200</td>
      <td>-0.014671</td>
      <td>0.000000</td>
      <td>294.554286</td>
      <td>...</td>
      <td>64.920000</td>
      <td>0.000000</td>
      <td>12.537143</td>
      <td>1.485714</td>
      <td>24.157143</td>
      <td>4.042857</td>
      <td>27.200000</td>
      <td>14.200000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>2010.000000</td>
      <td>13.750000</td>
      <td>NaN</td>
      <td>-0.001500</td>
      <td>0.015975</td>
      <td>0.148670</td>
      <td>0.134079</td>
      <td>8.175000</td>
      <td>297.751429</td>
      <td>...</td>
      <td>77.397143</td>
      <td>8.175000</td>
      <td>15.792857</td>
      <td>2.446429</td>
      <td>26.514286</td>
      <td>5.928571</td>
      <td>31.100000</td>
      <td>21.200000</td>
      <td>9.100000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>2011.000000</td>
      <td>26.000000</td>
      <td>NaN</td>
      <td>0.110100</td>
      <td>0.088700</td>
      <td>0.204171</td>
      <td>0.186471</td>
      <td>31.455000</td>
      <td>298.547143</td>
      <td>...</td>
      <td>80.330000</td>
      <td>31.455000</td>
      <td>17.337143</td>
      <td>2.914286</td>
      <td>27.483333</td>
      <td>6.642857</td>
      <td>32.800000</td>
      <td>22.200000</td>
      <td>23.600000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>2012.000000</td>
      <td>39.000000</td>
      <td>NaN</td>
      <td>0.263329</td>
      <td>0.242400</td>
      <td>0.254871</td>
      <td>0.253243</td>
      <td>57.772500</td>
      <td>300.240357</td>
      <td>...</td>
      <td>88.328929</td>
      <td>57.772500</td>
      <td>18.174643</td>
      <td>8.171429</td>
      <td>28.319048</td>
      <td>9.812500</td>
      <td>33.900000</td>
      <td>23.300000</td>
      <td>47.750000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>2013.000000</td>
      <td>53.000000</td>
      <td>NaN</td>
      <td>0.500400</td>
      <td>0.649000</td>
      <td>0.453043</td>
      <td>0.529043</td>
      <td>169.340000</td>
      <td>301.935714</td>
      <td>...</td>
      <td>97.982857</td>
      <td>169.340000</td>
      <td>19.598571</td>
      <td>14.485714</td>
      <td>30.271429</td>
      <td>14.725000</td>
      <td>38.400000</td>
      <td>26.700000</td>
      <td>212.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 25 columns</p>
</div>




```python
# Check if there is any NA in training data
training.isnull().sum()
```




    city                                       0
    year                                       0
    weekofyear                                 0
    week_start_date                            0
    ndvi_ne                                  194
    ndvi_nw                                   52
    ndvi_se                                   22
    ndvi_sw                                   22
    precipitation_amt_mm                      13
    reanalysis_air_temp_k                     10
    reanalysis_avg_temp_k                     10
    reanalysis_dew_point_temp_k               10
    reanalysis_max_air_temp_k                 10
    reanalysis_min_air_temp_k                 10
    reanalysis_precip_amt_kg_per_m2           10
    reanalysis_relative_humidity_percent      10
    reanalysis_sat_precip_amt_mm              13
    reanalysis_specific_humidity_g_per_kg     10
    reanalysis_tdtr_k                         10
    station_avg_temp_c                        43
    station_diur_temp_rng_c                   43
    station_max_temp_c                        20
    station_min_temp_c                        14
    station_precip_mm                         22
    total_cases                                0
    dtype: int64




```python
# Check if there is any NA in submission data
submit.isnull().sum()
```




    city                                      0
    year                                      0
    weekofyear                                0
    week_start_date                           0
    ndvi_ne                                  43
    ndvi_nw                                  11
    ndvi_se                                   1
    ndvi_sw                                   1
    precipitation_amt_mm                      2
    reanalysis_air_temp_k                     2
    reanalysis_avg_temp_k                     2
    reanalysis_dew_point_temp_k               2
    reanalysis_max_air_temp_k                 2
    reanalysis_min_air_temp_k                 2
    reanalysis_precip_amt_kg_per_m2           2
    reanalysis_relative_humidity_percent      2
    reanalysis_sat_precip_amt_mm              2
    reanalysis_specific_humidity_g_per_kg     2
    reanalysis_tdtr_k                         2
    station_avg_temp_c                       12
    station_diur_temp_rng_c                  12
    station_max_temp_c                        3
    station_min_temp_c                        9
    station_precip_mm                         5
    total_cases                               0
    dtype: int64




```python
# Observe histogram of each numeric feature
plt.figure()
training.hist(figsize=(20,20), layout=(8,3))
plt.tight_layout()
plt.show()
```


    <matplotlib.figure.Figure at 0x24bfd4ed198>



![png](output_15_1.png)



```python
# Explore correlation between variables
plt.figure(figsize=(20,20))
sns.heatmap(training.corr(), xticklabels=training.corr().columns, yticklabels=training.corr().columns, center=0, annot=True)
plt.show()
```


![png](output_16_0.png)


### Step 3: feature engineering

Based on the exploratory data analysis, it seems that the following steps could/should be done:
1. generating datetime features
2. filling NAs
3. removing highly correlated features
4. one-hot encoding for categorical features

For step 2, since the NA values for each column are weather related data, it makes sense to fillna in groups of city and month.
And as the NA values only accounts for less than 10% of the total data for each column, a naive method, median would be used.


```python
# generating date_time features
training['week_start_date'] = pd.to_datetime(training['week_start_date'], format='%Y-%m-%d')
training['quarter'] = training.week_start_date.dt.quarter
training['month'] = training.week_start_date.dt.month
training['day'] = training.week_start_date.dt.day

submit['week_start_date'] = pd.to_datetime(submit['week_start_date'], format='%Y-%m-%d')
submit['quarter'] = submit.week_start_date.dt.quarter
submit['month'] = submit.week_start_date.dt.month
submit['day'] = submit.week_start_date.dt.day
```


```python
# fillna via naive method, median, grouped by city and month
training = training.groupby(['city', 'month'], as_index=False).apply(lambda x: x.fillna(x.median())).reset_index(drop=True)
submit = submit.groupby(['city', 'month'], as_index=False).apply(lambda x: x.fillna(x.median())).reset_index(drop=True)
```


```python
# Remove some highly correlated non-datetime related features (absolute correlation > 0.9)
features_to_remove = ['reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_avg_temp_k', 'reanalysis_tdtr_k']
training.drop(features_to_remove, axis=1, inplace=True)
submit.drop(features_to_remove, axis=1, inplace=True)
```


```python
# one-hot encoding for categorical data: city
training = pd.concat([pd.get_dummies(training.city).astype(int), training], axis=1)
submit = pd.concat([pd.get_dummies(submit.city).astype(int), submit], axis=1)
```

### Step 4: train a model and test result simply using training data

This step helps us understand the accuracy of our model before using it to predict the 'submit' dataset


```python
# drop week_start_date before training
training.drop(['week_start_date'], axis=1, inplace=True)
submit.drop(['week_start_date'], axis=1, inplace=True)
```


```python
# train test split for train, test and cv and then drop categorical variable: city
X_train, X_test, y_train, y_test = train_test_split(training.drop(['total_cases'], axis=1), training.total_cases, test_size=0.2, stratify=training.city, random_state=123)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.25, stratify=X_train.city, random_state=123)

X_train.drop('city', axis=1, inplace=True)
X_cv.drop('city', axis=1, inplace=True)
X_test.drop('city', axis=1, inplace=True)
```


```python
# train the model
model = CatBoostRegressor(iterations = 4000, learning_rate = 0.3, loss_function='MAE', eval_metric='MAE', use_best_model=True, random_seed=123)
model.fit(X_train, y_train, eval_set=(X_cv, y_cv), verbose=1)
```

    0:	learn: 24.6619113	test: 21.8641054	best: 21.8641054 (0)	total: 79.4ms	remaining: 5m 17s
    1:	learn: 24.5425861	test: 21.7437191	best: 21.7437191 (1)	total: 108ms	remaining: 3m 36s
    2:	learn: 24.4293375	test: 21.6288173	best: 21.6288173 (2)	total: 136ms	remaining: 3m 1s
    3:	learn: 24.3207233	test: 21.5181544	best: 21.5181544 (3)	total: 165ms	remaining: 2m 44s
    4:	learn: 24.2077914	test: 21.4016279	best: 21.4016279 (4)	total: 184ms	remaining: 2m 27s
    5:	learn: 24.1002789	test: 21.2923822	best: 21.2923822 (5)	total: 216ms	remaining: 2m 23s
    6:	learn: 23.9919965	test: 21.1836038	best: 21.1836038 (6)	total: 244ms	remaining: 2m 19s
    7:	learn: 23.8914076	test: 21.0845934	best: 21.0845934 (7)	total: 274ms	remaining: 2m 16s
    8:	learn: 23.7742649	test: 20.9702220	best: 20.9702220 (8)	total: 309ms	remaining: 2m 16s
    9:	learn: 23.6773737	test: 20.8759363	best: 20.8759363 (9)	total: 350ms	remaining: 2m 19s
    10:	learn: 23.5643492	test: 20.7703917	best: 20.7703917 (10)	total: 393ms	remaining: 2m 22s
    11:	learn: 23.4653834	test: 20.6767265	best: 20.6767265 (11)	total: 422ms	remaining: 2m 20s
    12:	learn: 23.3557876	test: 20.5721624	best: 20.5721624 (12)	total: 450ms	remaining: 2m 18s
    13:	learn: 23.2543542	test: 20.4811777	best: 20.4811777 (13)	total: 478ms	remaining: 2m 16s
    14:	learn: 23.1508799	test: 20.3885535	best: 20.3885535 (14)	total: 509ms	remaining: 2m 15s
    15:	learn: 23.0548491	test: 20.3007305	best: 20.3007305 (15)	total: 550ms	remaining: 2m 16s
    16:	learn: 22.9648791	test: 20.2147672	best: 20.2147672 (16)	total: 590ms	remaining: 2m 18s
    17:	learn: 22.8689467	test: 20.1215499	best: 20.1215499 (17)	total: 601ms	remaining: 2m 12s
    18:	learn: 22.7601604	test: 20.0161199	best: 20.0161199 (18)	total: 624ms	remaining: 2m 10s
    19:	learn: 22.6593199	test: 19.9188855	best: 19.9188855 (19)	total: 655ms	remaining: 2m 10s
    20:	learn: 22.5616450	test: 19.8235960	best: 19.8235960 (20)	total: 695ms	remaining: 2m 11s
    21:	learn: 22.4706892	test: 19.7352392	best: 19.7352392 (21)	total: 744ms	remaining: 2m 14s
    22:	learn: 22.3844599	test: 19.6515137	best: 19.6515137 (22)	total: 797ms	remaining: 2m 17s
    23:	learn: 22.2987369	test: 19.5723911	best: 19.5723911 (23)	total: 821ms	remaining: 2m 16s
    24:	learn: 22.2158877	test: 19.4974134	best: 19.4974134 (24)	total: 856ms	remaining: 2m 16s
    25:	learn: 22.1282814	test: 19.4159362	best: 19.4159362 (25)	total: 891ms	remaining: 2m 16s
    26:	learn: 22.0400503	test: 19.3327140	best: 19.3327140 (26)	total: 929ms	remaining: 2m 16s
    27:	learn: 21.9526313	test: 19.2525461	best: 19.2525461 (27)	total: 971ms	remaining: 2m 17s
    28:	learn: 21.8667571	test: 19.1769863	best: 19.1769863 (28)	total: 1.03s	remaining: 2m 21s
    29:	learn: 21.7782680	test: 19.0983818	best: 19.0983818 (29)	total: 1.07s	remaining: 2m 22s
    30:	learn: 21.6953884	test: 19.0234650	best: 19.0234650 (30)	total: 1.11s	remaining: 2m 22s
    31:	learn: 21.6147324	test: 18.9555002	best: 18.9555002 (31)	total: 1.15s	remaining: 2m 22s
    32:	learn: 21.5386823	test: 18.8867150	best: 18.8867150 (32)	total: 1.2s	remaining: 2m 24s
    33:	learn: 21.4620697	test: 18.8200466	best: 18.8200466 (33)	total: 1.25s	remaining: 2m 25s
    34:	learn: 21.3894318	test: 18.7563647	best: 18.7563647 (34)	total: 1.28s	remaining: 2m 24s
    35:	learn: 21.3193370	test: 18.6943935	best: 18.6943935 (35)	total: 1.31s	remaining: 2m 24s
    36:	learn: 21.2506381	test: 18.6353593	best: 18.6353593 (36)	total: 1.35s	remaining: 2m 24s
    37:	learn: 21.1785053	test: 18.5756844	best: 18.5756844 (37)	total: 1.38s	remaining: 2m 24s
    38:	learn: 21.1068086	test: 18.5133385	best: 18.5133385 (38)	total: 1.43s	remaining: 2m 24s
    39:	learn: 21.0369946	test: 18.4562788	best: 18.4562788 (39)	total: 1.47s	remaining: 2m 25s
    40:	learn: 20.9729859	test: 18.3978325	best: 18.3978325 (40)	total: 1.5s	remaining: 2m 24s
    41:	learn: 20.9065007	test: 18.3397453	best: 18.3397453 (41)	total: 1.52s	remaining: 2m 23s
    42:	learn: 20.8427594	test: 18.2902885	best: 18.2902885 (42)	total: 1.56s	remaining: 2m 23s
    43:	learn: 20.7759947	test: 18.2338018	best: 18.2338018 (43)	total: 1.59s	remaining: 2m 23s
    44:	learn: 20.7156781	test: 18.1843711	best: 18.1843711 (44)	total: 1.65s	remaining: 2m 25s
    45:	learn: 20.6522410	test: 18.1338769	best: 18.1338769 (45)	total: 1.7s	remaining: 2m 25s
    46:	learn: 20.5909784	test: 18.0796089	best: 18.0796089 (46)	total: 1.73s	remaining: 2m 25s
    47:	learn: 20.5297460	test: 18.0238773	best: 18.0238773 (47)	total: 1.76s	remaining: 2m 25s
    48:	learn: 20.4685124	test: 17.9723331	best: 17.9723331 (48)	total: 1.8s	remaining: 2m 25s
    49:	learn: 20.4099962	test: 17.9239052	best: 17.9239052 (49)	total: 1.84s	remaining: 2m 25s
    50:	learn: 20.3551495	test: 17.8798033	best: 17.8798033 (50)	total: 1.87s	remaining: 2m 25s
    51:	learn: 20.2972616	test: 17.8296172	best: 17.8296172 (51)	total: 1.9s	remaining: 2m 24s
    52:	learn: 20.2415775	test: 17.7845182	best: 17.7845182 (52)	total: 1.93s	remaining: 2m 24s
    53:	learn: 20.1821384	test: 17.7372130	best: 17.7372130 (53)	total: 1.97s	remaining: 2m 23s
    54:	learn: 20.1291309	test: 17.6950498	best: 17.6950498 (54)	total: 2s	remaining: 2m 23s
    55:	learn: 20.0789829	test: 17.6554953	best: 17.6554953 (55)	total: 2.03s	remaining: 2m 23s
    56:	learn: 20.0185510	test: 17.6053520	best: 17.6053520 (56)	total: 2.08s	remaining: 2m 23s
    57:	learn: 19.9649447	test: 17.5624599	best: 17.5624599 (57)	total: 2.11s	remaining: 2m 23s
    58:	learn: 19.9129135	test: 17.5157462	best: 17.5157462 (58)	total: 2.17s	remaining: 2m 25s
    59:	learn: 19.8603310	test: 17.4709460	best: 17.4709460 (59)	total: 2.24s	remaining: 2m 27s
    60:	learn: 19.8151395	test: 17.4292454	best: 17.4292454 (60)	total: 2.31s	remaining: 2m 29s
    61:	learn: 19.7676865	test: 17.3898612	best: 17.3898612 (61)	total: 2.36s	remaining: 2m 30s
    62:	learn: 19.7195983	test: 17.3476552	best: 17.3476552 (62)	total: 2.4s	remaining: 2m 30s
    63:	learn: 19.6779071	test: 17.3170240	best: 17.3170240 (63)	total: 2.44s	remaining: 2m 30s
    64:	learn: 19.6320189	test: 17.2793886	best: 17.2793886 (64)	total: 2.5s	remaining: 2m 31s
    65:	learn: 19.5849774	test: 17.2416351	best: 17.2416351 (65)	total: 2.55s	remaining: 2m 32s
    66:	learn: 19.5381556	test: 17.2034596	best: 17.2034596 (66)	total: 2.58s	remaining: 2m 31s
    67:	learn: 19.4995244	test: 17.1764147	best: 17.1764147 (67)	total: 2.62s	remaining: 2m 31s
    68:	learn: 19.4512369	test: 17.1316819	best: 17.1316819 (68)	total: 2.65s	remaining: 2m 31s
    69:	learn: 19.4068152	test: 17.0965019	best: 17.0965019 (69)	total: 2.69s	remaining: 2m 30s
    70:	learn: 19.3552431	test: 17.0493129	best: 17.0493129 (70)	total: 2.72s	remaining: 2m 30s
    71:	learn: 19.3082867	test: 17.0125126	best: 17.0125126 (71)	total: 2.75s	remaining: 2m 30s
    72:	learn: 19.2568456	test: 16.9688548	best: 16.9688548 (72)	total: 2.81s	remaining: 2m 31s
    73:	learn: 19.2043367	test: 16.9362607	best: 16.9362607 (73)	total: 2.95s	remaining: 2m 36s
    74:	learn: 19.1632999	test: 16.9043335	best: 16.9043335 (74)	total: 3.07s	remaining: 2m 40s
    75:	learn: 19.1182482	test: 16.8598230	best: 16.8598230 (75)	total: 3.13s	remaining: 2m 41s
    76:	learn: 19.0767517	test: 16.8222787	best: 16.8222787 (76)	total: 3.18s	remaining: 2m 42s
    77:	learn: 19.0380344	test: 16.7884446	best: 16.7884446 (77)	total: 3.24s	remaining: 2m 43s
    78:	learn: 18.9933598	test: 16.7463641	best: 16.7463641 (78)	total: 3.3s	remaining: 2m 43s
    79:	learn: 18.9472378	test: 16.7123161	best: 16.7123161 (79)	total: 3.35s	remaining: 2m 44s
    80:	learn: 18.9052342	test: 16.6709750	best: 16.6709750 (80)	total: 3.38s	remaining: 2m 43s
    81:	learn: 18.8679289	test: 16.6376703	best: 16.6376703 (81)	total: 3.43s	remaining: 2m 43s
    82:	learn: 18.8273406	test: 16.6058177	best: 16.6058177 (82)	total: 3.48s	remaining: 2m 44s
    83:	learn: 18.7815289	test: 16.5728333	best: 16.5728333 (83)	total: 3.51s	remaining: 2m 43s
    84:	learn: 18.7403649	test: 16.5375549	best: 16.5375549 (84)	total: 3.55s	remaining: 2m 43s
    85:	learn: 18.6926404	test: 16.4978677	best: 16.4978677 (85)	total: 3.59s	remaining: 2m 43s
    86:	learn: 18.6483752	test: 16.4639718	best: 16.4639718 (86)	total: 3.62s	remaining: 2m 42s
    87:	learn: 18.6026893	test: 16.4218761	best: 16.4218761 (87)	total: 3.67s	remaining: 2m 43s
    88:	learn: 18.5607289	test: 16.3904364	best: 16.3904364 (88)	total: 3.71s	remaining: 2m 43s
    89:	learn: 18.5252829	test: 16.3567477	best: 16.3567477 (89)	total: 3.74s	remaining: 2m 42s
    90:	learn: 18.4903937	test: 16.3269341	best: 16.3269341 (90)	total: 3.77s	remaining: 2m 42s
    91:	learn: 18.4510576	test: 16.2911433	best: 16.2911433 (91)	total: 3.81s	remaining: 2m 42s
    92:	learn: 18.4097419	test: 16.2521330	best: 16.2521330 (92)	total: 3.86s	remaining: 2m 42s
    93:	learn: 18.3718731	test: 16.2203392	best: 16.2203392 (93)	total: 3.89s	remaining: 2m 41s
    94:	learn: 18.3244292	test: 16.1773656	best: 16.1773656 (94)	total: 3.92s	remaining: 2m 41s
    95:	learn: 18.2826362	test: 16.1386670	best: 16.1386670 (95)	total: 3.96s	remaining: 2m 41s
    96:	learn: 18.2323329	test: 16.0879348	best: 16.0879348 (96)	total: 4s	remaining: 2m 41s
    97:	learn: 18.1957038	test: 16.0600652	best: 16.0600652 (97)	total: 4.04s	remaining: 2m 40s
    98:	learn: 18.1501738	test: 16.0176395	best: 16.0176395 (98)	total: 4.07s	remaining: 2m 40s
    99:	learn: 18.1165727	test: 15.9941483	best: 15.9941483 (99)	total: 4.1s	remaining: 2m 39s
    100:	learn: 18.0740836	test: 15.9597122	best: 15.9597122 (100)	total: 4.13s	remaining: 2m 39s
    101:	learn: 18.0416858	test: 15.9336412	best: 15.9336412 (101)	total: 4.16s	remaining: 2m 39s
    102:	learn: 18.0093994	test: 15.9100160	best: 15.9100160 (102)	total: 4.2s	remaining: 2m 38s
    103:	learn: 17.9771320	test: 15.8847607	best: 15.8847607 (103)	total: 4.24s	remaining: 2m 38s
    104:	learn: 17.9405122	test: 15.8547039	best: 15.8547039 (104)	total: 4.27s	remaining: 2m 38s
    105:	learn: 17.9169913	test: 15.8387332	best: 15.8387332 (105)	total: 4.3s	remaining: 2m 38s
    106:	learn: 17.8846999	test: 15.8122762	best: 15.8122762 (106)	total: 4.34s	remaining: 2m 37s
    107:	learn: 17.8537030	test: 15.7841112	best: 15.7841112 (107)	total: 4.38s	remaining: 2m 38s
    108:	learn: 17.8163308	test: 15.7497246	best: 15.7497246 (108)	total: 4.43s	remaining: 2m 38s
    109:	learn: 17.7821260	test: 15.7199939	best: 15.7199939 (109)	total: 4.5s	remaining: 2m 39s
    110:	learn: 17.7424944	test: 15.6841521	best: 15.6841521 (110)	total: 4.56s	remaining: 2m 39s
    111:	learn: 17.7036692	test: 15.6504980	best: 15.6504980 (111)	total: 4.61s	remaining: 2m 39s
    112:	learn: 17.6702445	test: 15.6227210	best: 15.6227210 (112)	total: 4.64s	remaining: 2m 39s
    113:	learn: 17.6347931	test: 15.5913586	best: 15.5913586 (113)	total: 4.67s	remaining: 2m 39s
    114:	learn: 17.5981590	test: 15.5633360	best: 15.5633360 (114)	total: 4.7s	remaining: 2m 38s
    115:	learn: 17.5668832	test: 15.5394692	best: 15.5394692 (115)	total: 4.73s	remaining: 2m 38s
    116:	learn: 17.5393332	test: 15.5209620	best: 15.5209620 (116)	total: 4.77s	remaining: 2m 38s
    117:	learn: 17.5103572	test: 15.5030922	best: 15.5030922 (117)	total: 4.8s	remaining: 2m 37s
    118:	learn: 17.4738283	test: 15.4732644	best: 15.4732644 (118)	total: 4.83s	remaining: 2m 37s
    119:	learn: 17.4374397	test: 15.4436374	best: 15.4436374 (119)	total: 4.87s	remaining: 2m 37s
    120:	learn: 17.4046174	test: 15.4146903	best: 15.4146903 (120)	total: 4.9s	remaining: 2m 37s
    121:	learn: 17.3667515	test: 15.3857077	best: 15.3857077 (121)	total: 4.93s	remaining: 2m 36s
    122:	learn: 17.3281358	test: 15.3555968	best: 15.3555968 (122)	total: 4.96s	remaining: 2m 36s
    123:	learn: 17.2932568	test: 15.3250952	best: 15.3250952 (123)	total: 5s	remaining: 2m 36s
    124:	learn: 17.2557505	test: 15.2956288	best: 15.2956288 (124)	total: 5.03s	remaining: 2m 35s
    125:	learn: 17.2270210	test: 15.2693362	best: 15.2693362 (125)	total: 5.06s	remaining: 2m 35s
    126:	learn: 17.1907202	test: 15.2416636	best: 15.2416636 (126)	total: 5.09s	remaining: 2m 35s
    127:	learn: 17.1641071	test: 15.2211920	best: 15.2211920 (127)	total: 5.12s	remaining: 2m 35s
    128:	learn: 17.1445240	test: 15.2075957	best: 15.2075957 (128)	total: 5.15s	remaining: 2m 34s
    129:	learn: 17.1108491	test: 15.1819583	best: 15.1819583 (129)	total: 5.19s	remaining: 2m 34s
    130:	learn: 17.0777953	test: 15.1586283	best: 15.1586283 (130)	total: 5.22s	remaining: 2m 34s
    131:	learn: 17.0452519	test: 15.1353515	best: 15.1353515 (131)	total: 5.26s	remaining: 2m 34s
    132:	learn: 17.0176973	test: 15.1153024	best: 15.1153024 (132)	total: 5.29s	remaining: 2m 33s
    133:	learn: 16.9940315	test: 15.0961188	best: 15.0961188 (133)	total: 5.32s	remaining: 2m 33s
    134:	learn: 16.9655098	test: 15.0739752	best: 15.0739752 (134)	total: 5.36s	remaining: 2m 33s
    135:	learn: 16.9313344	test: 15.0489934	best: 15.0489934 (135)	total: 5.39s	remaining: 2m 33s
    136:	learn: 16.8972414	test: 15.0236842	best: 15.0236842 (136)	total: 5.44s	remaining: 2m 33s
    137:	learn: 16.8696126	test: 15.0043970	best: 15.0043970 (137)	total: 5.5s	remaining: 2m 33s
    138:	learn: 16.8376593	test: 14.9778526	best: 14.9778526 (138)	total: 5.56s	remaining: 2m 34s
    139:	learn: 16.8127519	test: 14.9619863	best: 14.9619863 (139)	total: 5.64s	remaining: 2m 35s
    140:	learn: 16.7809334	test: 14.9356616	best: 14.9356616 (140)	total: 5.71s	remaining: 2m 36s
    141:	learn: 16.7550904	test: 14.9160502	best: 14.9160502 (141)	total: 5.94s	remaining: 2m 41s
    142:	learn: 16.7272295	test: 14.8919825	best: 14.8919825 (142)	total: 5.99s	remaining: 2m 41s
    143:	learn: 16.6982038	test: 14.8692318	best: 14.8692318 (143)	total: 6.02s	remaining: 2m 41s
    144:	learn: 16.6740210	test: 14.8497119	best: 14.8497119 (144)	total: 6.05s	remaining: 2m 40s
    145:	learn: 16.6437073	test: 14.8259948	best: 14.8259948 (145)	total: 6.08s	remaining: 2m 40s
    146:	learn: 16.6158171	test: 14.8075861	best: 14.8075861 (146)	total: 6.1s	remaining: 2m 39s
    147:	learn: 16.5938748	test: 14.7888217	best: 14.7888217 (147)	total: 6.13s	remaining: 2m 39s
    148:	learn: 16.5711359	test: 14.7717877	best: 14.7717877 (148)	total: 6.16s	remaining: 2m 39s
    149:	learn: 16.5512949	test: 14.7557942	best: 14.7557942 (149)	total: 6.19s	remaining: 2m 38s
    150:	learn: 16.5250356	test: 14.7339991	best: 14.7339991 (150)	total: 6.22s	remaining: 2m 38s
    151:	learn: 16.5023532	test: 14.7172522	best: 14.7172522 (151)	total: 6.25s	remaining: 2m 38s
    152:	learn: 16.4810447	test: 14.7017761	best: 14.7017761 (152)	total: 6.28s	remaining: 2m 37s
    153:	learn: 16.4618425	test: 14.6857591	best: 14.6857591 (153)	total: 6.3s	remaining: 2m 37s
    154:	learn: 16.4412632	test: 14.6712044	best: 14.6712044 (154)	total: 6.33s	remaining: 2m 37s
    155:	learn: 16.4127791	test: 14.6498218	best: 14.6498218 (155)	total: 6.37s	remaining: 2m 36s
    156:	learn: 16.3907939	test: 14.6313739	best: 14.6313739 (156)	total: 6.4s	remaining: 2m 36s
    157:	learn: 16.3703526	test: 14.6145018	best: 14.6145018 (157)	total: 6.44s	remaining: 2m 36s
    158:	learn: 16.3439177	test: 14.5917273	best: 14.5917273 (158)	total: 6.47s	remaining: 2m 36s
    159:	learn: 16.3168019	test: 14.5687709	best: 14.5687709 (159)	total: 6.5s	remaining: 2m 36s
    160:	learn: 16.2965771	test: 14.5494209	best: 14.5494209 (160)	total: 6.54s	remaining: 2m 35s
    161:	learn: 16.2701417	test: 14.5283545	best: 14.5283545 (161)	total: 6.57s	remaining: 2m 35s
    162:	learn: 16.2438589	test: 14.5056496	best: 14.5056496 (162)	total: 6.6s	remaining: 2m 35s
    163:	learn: 16.2179707	test: 14.4851645	best: 14.4851645 (163)	total: 6.63s	remaining: 2m 35s
    164:	learn: 16.1973224	test: 14.4701209	best: 14.4701209 (164)	total: 6.65s	remaining: 2m 34s
    165:	learn: 16.1744298	test: 14.4504146	best: 14.4504146 (165)	total: 6.68s	remaining: 2m 34s
    166:	learn: 16.1524411	test: 14.4325977	best: 14.4325977 (166)	total: 6.71s	remaining: 2m 33s
    167:	learn: 16.1351038	test: 14.4191207	best: 14.4191207 (167)	total: 6.73s	remaining: 2m 33s
    168:	learn: 16.1151955	test: 14.4072763	best: 14.4072763 (168)	total: 6.76s	remaining: 2m 33s
    169:	learn: 16.0908209	test: 14.3867151	best: 14.3867151 (169)	total: 6.78s	remaining: 2m 32s
    170:	learn: 16.0668218	test: 14.3642148	best: 14.3642148 (170)	total: 6.81s	remaining: 2m 32s
    171:	learn: 16.0455943	test: 14.3446667	best: 14.3446667 (171)	total: 6.85s	remaining: 2m 32s
    172:	learn: 16.0211538	test: 14.3241397	best: 14.3241397 (172)	total: 6.88s	remaining: 2m 32s
    173:	learn: 15.9955318	test: 14.3075483	best: 14.3075483 (173)	total: 6.91s	remaining: 2m 31s
    174:	learn: 15.9696665	test: 14.2829351	best: 14.2829351 (174)	total: 6.94s	remaining: 2m 31s
    175:	learn: 15.9474765	test: 14.2672528	best: 14.2672528 (175)	total: 6.97s	remaining: 2m 31s
    176:	learn: 15.9221307	test: 14.2467923	best: 14.2467923 (176)	total: 7s	remaining: 2m 31s
    177:	learn: 15.8981402	test: 14.2261759	best: 14.2261759 (177)	total: 7.04s	remaining: 2m 31s
    178:	learn: 15.8781185	test: 14.2079305	best: 14.2079305 (178)	total: 7.08s	remaining: 2m 31s
    179:	learn: 15.8548048	test: 14.1907876	best: 14.1907876 (179)	total: 7.11s	remaining: 2m 30s
    180:	learn: 15.8345713	test: 14.1737219	best: 14.1737219 (180)	total: 7.14s	remaining: 2m 30s
    181:	learn: 15.8172256	test: 14.1603343	best: 14.1603343 (181)	total: 7.17s	remaining: 2m 30s
    182:	learn: 15.7954899	test: 14.1448746	best: 14.1448746 (182)	total: 7.21s	remaining: 2m 30s
    183:	learn: 15.7718340	test: 14.1259002	best: 14.1259002 (183)	total: 7.24s	remaining: 2m 30s
    184:	learn: 15.7561469	test: 14.1171351	best: 14.1171351 (184)	total: 7.28s	remaining: 2m 30s
    185:	learn: 15.7356355	test: 14.1016834	best: 14.1016834 (185)	total: 7.31s	remaining: 2m 29s
    186:	learn: 15.7087330	test: 14.0801837	best: 14.0801837 (186)	total: 7.35s	remaining: 2m 29s
    187:	learn: 15.6822317	test: 14.0623615	best: 14.0623615 (187)	total: 7.38s	remaining: 2m 29s
    188:	learn: 15.6625663	test: 14.0540055	best: 14.0540055 (188)	total: 7.42s	remaining: 2m 29s
    189:	learn: 15.6411345	test: 14.0351007	best: 14.0351007 (189)	total: 7.45s	remaining: 2m 29s
    190:	learn: 15.6255730	test: 14.0250683	best: 14.0250683 (190)	total: 7.49s	remaining: 2m 29s
    191:	learn: 15.6028705	test: 14.0058744	best: 14.0058744 (191)	total: 7.53s	remaining: 2m 29s
    192:	learn: 15.5803207	test: 13.9868708	best: 13.9868708 (192)	total: 7.56s	remaining: 2m 29s
    193:	learn: 15.5608578	test: 13.9716314	best: 13.9716314 (193)	total: 7.6s	remaining: 2m 29s
    194:	learn: 15.5397430	test: 13.9551761	best: 13.9551761 (194)	total: 7.63s	remaining: 2m 28s
    195:	learn: 15.5171157	test: 13.9372827	best: 13.9372827 (195)	total: 7.66s	remaining: 2m 28s
    196:	learn: 15.4986476	test: 13.9237154	best: 13.9237154 (196)	total: 7.69s	remaining: 2m 28s
    197:	learn: 15.4836413	test: 13.9129320	best: 13.9129320 (197)	total: 7.73s	remaining: 2m 28s
    198:	learn: 15.4562919	test: 13.8965819	best: 13.8965819 (198)	total: 7.76s	remaining: 2m 28s
    199:	learn: 15.4433056	test: 13.8879489	best: 13.8879489 (199)	total: 7.79s	remaining: 2m 28s
    200:	learn: 15.4287207	test: 13.8755853	best: 13.8755853 (200)	total: 7.82s	remaining: 2m 27s
    201:	learn: 15.4089147	test: 13.8612412	best: 13.8612412 (201)	total: 7.85s	remaining: 2m 27s
    202:	learn: 15.3939612	test: 13.8455085	best: 13.8455085 (202)	total: 7.88s	remaining: 2m 27s
    203:	learn: 15.3723294	test: 13.8283366	best: 13.8283366 (203)	total: 7.91s	remaining: 2m 27s
    204:	learn: 15.3524234	test: 13.8136916	best: 13.8136916 (204)	total: 7.94s	remaining: 2m 27s
    205:	learn: 15.3293906	test: 13.8002509	best: 13.8002509 (205)	total: 7.97s	remaining: 2m 26s
    206:	learn: 15.3143288	test: 13.7947989	best: 13.7947989 (206)	total: 8s	remaining: 2m 26s
    207:	learn: 15.2951249	test: 13.7819826	best: 13.7819826 (207)	total: 8.03s	remaining: 2m 26s
    208:	learn: 15.2725794	test: 13.7642748	best: 13.7642748 (208)	total: 8.06s	remaining: 2m 26s
    209:	learn: 15.2610478	test: 13.7562115	best: 13.7562115 (209)	total: 8.09s	remaining: 2m 26s
    210:	learn: 15.2477262	test: 13.7466344	best: 13.7466344 (210)	total: 8.12s	remaining: 2m 25s
    211:	learn: 15.2338663	test: 13.7395080	best: 13.7395080 (211)	total: 8.15s	remaining: 2m 25s
    212:	learn: 15.2247256	test: 13.7342874	best: 13.7342874 (212)	total: 8.18s	remaining: 2m 25s
    213:	learn: 15.2114837	test: 13.7263359	best: 13.7263359 (213)	total: 8.21s	remaining: 2m 25s
    214:	learn: 15.1901567	test: 13.7095166	best: 13.7095166 (214)	total: 8.23s	remaining: 2m 24s
    215:	learn: 15.1720456	test: 13.6958730	best: 13.6958730 (215)	total: 8.26s	remaining: 2m 24s
    216:	learn: 15.1559295	test: 13.6840452	best: 13.6840452 (216)	total: 8.28s	remaining: 2m 24s
    217:	learn: 15.1398770	test: 13.6738582	best: 13.6738582 (217)	total: 8.32s	remaining: 2m 24s
    218:	learn: 15.1238760	test: 13.6608943	best: 13.6608943 (218)	total: 8.35s	remaining: 2m 24s
    219:	learn: 15.1103973	test: 13.6546958	best: 13.6546958 (219)	total: 8.38s	remaining: 2m 23s
    220:	learn: 15.0994622	test: 13.6459137	best: 13.6459137 (220)	total: 8.41s	remaining: 2m 23s
    221:	learn: 15.0834105	test: 13.6309808	best: 13.6309808 (221)	total: 8.44s	remaining: 2m 23s
    222:	learn: 15.0689023	test: 13.6194732	best: 13.6194732 (222)	total: 8.47s	remaining: 2m 23s
    223:	learn: 15.0527438	test: 13.6055754	best: 13.6055754 (223)	total: 8.5s	remaining: 2m 23s
    224:	learn: 15.0382245	test: 13.5968813	best: 13.5968813 (224)	total: 8.53s	remaining: 2m 23s
    225:	learn: 15.0216757	test: 13.5903968	best: 13.5903968 (225)	total: 8.57s	remaining: 2m 23s
    226:	learn: 15.0009659	test: 13.5756866	best: 13.5756866 (226)	total: 8.6s	remaining: 2m 22s
    227:	learn: 14.9805972	test: 13.5619346	best: 13.5619346 (227)	total: 8.62s	remaining: 2m 22s
    228:	learn: 14.9729984	test: 13.5588990	best: 13.5588990 (228)	total: 8.65s	remaining: 2m 22s
    229:	learn: 14.9538905	test: 13.5422512	best: 13.5422512 (229)	total: 8.67s	remaining: 2m 22s
    230:	learn: 14.9446620	test: 13.5360838	best: 13.5360838 (230)	total: 8.7s	remaining: 2m 21s
    231:	learn: 14.9253568	test: 13.5214274	best: 13.5214274 (231)	total: 8.73s	remaining: 2m 21s
    232:	learn: 14.9080760	test: 13.5092707	best: 13.5092707 (232)	total: 8.76s	remaining: 2m 21s
    233:	learn: 14.8921305	test: 13.4984032	best: 13.4984032 (233)	total: 8.8s	remaining: 2m 21s
    234:	learn: 14.8824522	test: 13.4924169	best: 13.4924169 (234)	total: 8.82s	remaining: 2m 21s
    235:	learn: 14.8658679	test: 13.4816896	best: 13.4816896 (235)	total: 8.85s	remaining: 2m 21s
    236:	learn: 14.8547332	test: 13.4752167	best: 13.4752167 (236)	total: 8.88s	remaining: 2m 20s
    237:	learn: 14.8430957	test: 13.4667643	best: 13.4667643 (237)	total: 8.9s	remaining: 2m 20s
    238:	learn: 14.8324738	test: 13.4581119	best: 13.4581119 (238)	total: 8.93s	remaining: 2m 20s
    239:	learn: 14.8151261	test: 13.4435656	best: 13.4435656 (239)	total: 8.96s	remaining: 2m 20s
    240:	learn: 14.7968513	test: 13.4282488	best: 13.4282488 (240)	total: 8.99s	remaining: 2m 20s
    241:	learn: 14.7829795	test: 13.4150665	best: 13.4150665 (241)	total: 9.02s	remaining: 2m 20s
    242:	learn: 14.7679837	test: 13.4058801	best: 13.4058801 (242)	total: 9.05s	remaining: 2m 19s
    243:	learn: 14.7528612	test: 13.3947123	best: 13.3947123 (243)	total: 9.07s	remaining: 2m 19s
    244:	learn: 14.7453813	test: 13.3914964	best: 13.3914964 (244)	total: 9.1s	remaining: 2m 19s
    245:	learn: 14.7262546	test: 13.3764212	best: 13.3764212 (245)	total: 9.13s	remaining: 2m 19s
    246:	learn: 14.7182153	test: 13.3720266	best: 13.3720266 (246)	total: 9.15s	remaining: 2m 19s
    247:	learn: 14.7033975	test: 13.3611472	best: 13.3611472 (247)	total: 9.18s	remaining: 2m 18s
    248:	learn: 14.6864614	test: 13.3480223	best: 13.3480223 (248)	total: 9.21s	remaining: 2m 18s
    249:	learn: 14.6732922	test: 13.3382256	best: 13.3382256 (249)	total: 9.24s	remaining: 2m 18s
    250:	learn: 14.6588662	test: 13.3294935	best: 13.3294935 (250)	total: 9.27s	remaining: 2m 18s
    251:	learn: 14.6463918	test: 13.3176133	best: 13.3176133 (251)	total: 9.29s	remaining: 2m 18s
    252:	learn: 14.6300370	test: 13.3087720	best: 13.3087720 (252)	total: 9.32s	remaining: 2m 18s
    253:	learn: 14.6145001	test: 13.3024763	best: 13.3024763 (253)	total: 9.35s	remaining: 2m 17s
    254:	learn: 14.6068954	test: 13.2974066	best: 13.2974066 (254)	total: 9.38s	remaining: 2m 17s
    255:	learn: 14.6008888	test: 13.2944449	best: 13.2944449 (255)	total: 9.41s	remaining: 2m 17s
    256:	learn: 14.5911558	test: 13.2871228	best: 13.2871228 (256)	total: 9.44s	remaining: 2m 17s
    257:	learn: 14.5852809	test: 13.2834341	best: 13.2834341 (257)	total: 9.47s	remaining: 2m 17s
    258:	learn: 14.5756665	test: 13.2780494	best: 13.2780494 (258)	total: 9.5s	remaining: 2m 17s
    259:	learn: 14.5608996	test: 13.2676705	best: 13.2676705 (259)	total: 9.53s	remaining: 2m 17s
    260:	learn: 14.5508094	test: 13.2602934	best: 13.2602934 (260)	total: 9.56s	remaining: 2m 16s
    261:	learn: 14.5359368	test: 13.2471717	best: 13.2471717 (261)	total: 9.58s	remaining: 2m 16s
    262:	learn: 14.5245707	test: 13.2434106	best: 13.2434106 (262)	total: 9.61s	remaining: 2m 16s
    263:	learn: 14.5143849	test: 13.2380743	best: 13.2380743 (263)	total: 9.64s	remaining: 2m 16s
    264:	learn: 14.5084206	test: 13.2365132	best: 13.2365132 (264)	total: 9.67s	remaining: 2m 16s
    265:	learn: 14.4988481	test: 13.2296130	best: 13.2296130 (265)	total: 9.7s	remaining: 2m 16s
    266:	learn: 14.4815979	test: 13.2173176	best: 13.2173176 (266)	total: 9.73s	remaining: 2m 16s
    267:	learn: 14.4672241	test: 13.2083156	best: 13.2083156 (267)	total: 9.75s	remaining: 2m 15s
    268:	learn: 14.4509668	test: 13.1928497	best: 13.1928497 (268)	total: 9.78s	remaining: 2m 15s
    269:	learn: 14.4357403	test: 13.1826206	best: 13.1826206 (269)	total: 9.81s	remaining: 2m 15s
    270:	learn: 14.4268767	test: 13.1796546	best: 13.1796546 (270)	total: 9.83s	remaining: 2m 15s
    271:	learn: 14.4193941	test: 13.1754061	best: 13.1754061 (271)	total: 9.86s	remaining: 2m 15s
    272:	learn: 14.4081447	test: 13.1681395	best: 13.1681395 (272)	total: 9.89s	remaining: 2m 15s
    273:	learn: 14.3993350	test: 13.1665942	best: 13.1665942 (273)	total: 9.92s	remaining: 2m 14s
    274:	learn: 14.3925528	test: 13.1635906	best: 13.1635906 (274)	total: 9.95s	remaining: 2m 14s
    275:	learn: 14.3823693	test: 13.1585874	best: 13.1585874 (275)	total: 9.98s	remaining: 2m 14s
    276:	learn: 14.3736676	test: 13.1547053	best: 13.1547053 (276)	total: 10.1s	remaining: 2m 15s
    277:	learn: 14.3660674	test: 13.1495072	best: 13.1495072 (277)	total: 10.1s	remaining: 2m 15s
    278:	learn: 14.3611895	test: 13.1474971	best: 13.1474971 (278)	total: 10.2s	remaining: 2m 15s
    279:	learn: 14.3485758	test: 13.1356053	best: 13.1356053 (279)	total: 10.2s	remaining: 2m 15s
    280:	learn: 14.3366058	test: 13.1291471	best: 13.1291471 (280)	total: 10.2s	remaining: 2m 15s
    281:	learn: 14.3302272	test: 13.1260694	best: 13.1260694 (281)	total: 10.2s	remaining: 2m 14s
    282:	learn: 14.3238286	test: 13.1230128	best: 13.1230128 (282)	total: 10.3s	remaining: 2m 14s
    283:	learn: 14.3134223	test: 13.1170905	best: 13.1170905 (283)	total: 10.3s	remaining: 2m 14s
    284:	learn: 14.3011846	test: 13.1115496	best: 13.1115496 (284)	total: 10.3s	remaining: 2m 14s
    285:	learn: 14.2923339	test: 13.1092544	best: 13.1092544 (285)	total: 10.4s	remaining: 2m 14s
    286:	learn: 14.2841722	test: 13.1052445	best: 13.1052445 (286)	total: 10.4s	remaining: 2m 14s
    287:	learn: 14.2757965	test: 13.0984118	best: 13.0984118 (287)	total: 10.4s	remaining: 2m 14s
    288:	learn: 14.2681743	test: 13.0969996	best: 13.0969996 (288)	total: 10.4s	remaining: 2m 13s
    289:	learn: 14.2639050	test: 13.0949015	best: 13.0949015 (289)	total: 10.5s	remaining: 2m 13s
    290:	learn: 14.2560240	test: 13.0920261	best: 13.0920261 (290)	total: 10.5s	remaining: 2m 13s
    291:	learn: 14.2520105	test: 13.0908502	best: 13.0908502 (291)	total: 10.5s	remaining: 2m 13s
    292:	learn: 14.2485565	test: 13.0881662	best: 13.0881662 (292)	total: 10.6s	remaining: 2m 13s
    293:	learn: 14.2409581	test: 13.0828971	best: 13.0828971 (293)	total: 10.6s	remaining: 2m 13s
    294:	learn: 14.2301229	test: 13.0767641	best: 13.0767641 (294)	total: 10.6s	remaining: 2m 13s
    295:	learn: 14.2202721	test: 13.0675978	best: 13.0675978 (295)	total: 10.7s	remaining: 2m 13s
    296:	learn: 14.2148802	test: 13.0637127	best: 13.0637127 (296)	total: 10.7s	remaining: 2m 13s
    297:	learn: 14.2108173	test: 13.0623640	best: 13.0623640 (297)	total: 10.7s	remaining: 2m 13s
    298:	learn: 14.2053807	test: 13.0594945	best: 13.0594945 (298)	total: 10.7s	remaining: 2m 12s
    299:	learn: 14.1979141	test: 13.0550315	best: 13.0550315 (299)	total: 10.8s	remaining: 2m 12s
    300:	learn: 14.1851053	test: 13.0435830	best: 13.0435830 (300)	total: 10.8s	remaining: 2m 12s
    301:	learn: 14.1809823	test: 13.0420142	best: 13.0420142 (301)	total: 10.8s	remaining: 2m 12s
    302:	learn: 14.1745567	test: 13.0401478	best: 13.0401478 (302)	total: 10.8s	remaining: 2m 12s
    303:	learn: 14.1647857	test: 13.0323896	best: 13.0323896 (303)	total: 10.9s	remaining: 2m 12s
    304:	learn: 14.1538909	test: 13.0258183	best: 13.0258183 (304)	total: 10.9s	remaining: 2m 12s
    305:	learn: 14.1489291	test: 13.0210901	best: 13.0210901 (305)	total: 10.9s	remaining: 2m 11s
    306:	learn: 14.1405057	test: 13.0149192	best: 13.0149192 (306)	total: 11s	remaining: 2m 11s
    307:	learn: 14.1263580	test: 13.0080137	best: 13.0080137 (307)	total: 11s	remaining: 2m 11s
    308:	learn: 14.1193933	test: 13.0071084	best: 13.0071084 (308)	total: 11s	remaining: 2m 11s
    309:	learn: 14.1081173	test: 13.0013888	best: 13.0013888 (309)	total: 11s	remaining: 2m 11s
    310:	learn: 14.0984816	test: 12.9943679	best: 12.9943679 (310)	total: 11.1s	remaining: 2m 11s
    311:	learn: 14.0878532	test: 12.9865896	best: 12.9865896 (311)	total: 11.1s	remaining: 2m 11s
    312:	learn: 14.0795500	test: 12.9821135	best: 12.9821135 (312)	total: 11.1s	remaining: 2m 11s
    313:	learn: 14.0767849	test: 12.9839682	best: 12.9821135 (312)	total: 11.2s	remaining: 2m 10s
    314:	learn: 14.0645143	test: 12.9773513	best: 12.9773513 (314)	total: 11.2s	remaining: 2m 10s
    315:	learn: 14.0585349	test: 12.9727607	best: 12.9727607 (315)	total: 11.2s	remaining: 2m 10s
    316:	learn: 14.0476496	test: 12.9667907	best: 12.9667907 (316)	total: 11.3s	remaining: 2m 11s
    317:	learn: 14.0387989	test: 12.9612576	best: 12.9612576 (317)	total: 11.3s	remaining: 2m 10s
    318:	learn: 14.0336091	test: 12.9594752	best: 12.9594752 (318)	total: 11.3s	remaining: 2m 10s
    319:	learn: 14.0258932	test: 12.9516929	best: 12.9516929 (319)	total: 11.4s	remaining: 2m 11s
    320:	learn: 14.0235092	test: 12.9498727	best: 12.9498727 (320)	total: 11.4s	remaining: 2m 11s
    321:	learn: 14.0138371	test: 12.9427609	best: 12.9427609 (321)	total: 11.5s	remaining: 2m 11s
    322:	learn: 14.0076141	test: 12.9398590	best: 12.9398590 (322)	total: 11.5s	remaining: 2m 10s
    323:	learn: 13.9994636	test: 12.9318935	best: 12.9318935 (323)	total: 11.5s	remaining: 2m 10s
    324:	learn: 13.9957471	test: 12.9281467	best: 12.9281467 (324)	total: 11.6s	remaining: 2m 10s
    325:	learn: 13.9904723	test: 12.9260651	best: 12.9260651 (325)	total: 11.6s	remaining: 2m 10s
    326:	learn: 13.9873451	test: 12.9260926	best: 12.9260651 (325)	total: 11.6s	remaining: 2m 10s
    327:	learn: 13.9795201	test: 12.9217276	best: 12.9217276 (327)	total: 11.6s	remaining: 2m 10s
    328:	learn: 13.9730736	test: 12.9181967	best: 12.9181967 (328)	total: 11.7s	remaining: 2m 10s
    329:	learn: 13.9657652	test: 12.9125614	best: 12.9125614 (329)	total: 11.7s	remaining: 2m 10s
    330:	learn: 13.9625724	test: 12.9110577	best: 12.9110577 (330)	total: 11.7s	remaining: 2m 9s
    331:	learn: 13.9542377	test: 12.9059390	best: 12.9059390 (331)	total: 11.8s	remaining: 2m 9s
    332:	learn: 13.9465926	test: 12.9001740	best: 12.9001740 (332)	total: 11.8s	remaining: 2m 9s
    333:	learn: 13.9427649	test: 12.8980900	best: 12.8980900 (333)	total: 11.8s	remaining: 2m 9s
    334:	learn: 13.9344160	test: 12.8961638	best: 12.8961638 (334)	total: 11.9s	remaining: 2m 9s
    335:	learn: 13.9221078	test: 12.8864132	best: 12.8864132 (335)	total: 11.9s	remaining: 2m 9s
    336:	learn: 13.9159529	test: 12.8823284	best: 12.8823284 (336)	total: 11.9s	remaining: 2m 9s
    337:	learn: 13.9101379	test: 12.8784339	best: 12.8784339 (337)	total: 12s	remaining: 2m 9s
    338:	learn: 13.9044803	test: 12.8745501	best: 12.8745501 (338)	total: 12s	remaining: 2m 9s
    339:	learn: 13.8949280	test: 12.8696817	best: 12.8696817 (339)	total: 12.1s	remaining: 2m 9s
    340:	learn: 13.8833124	test: 12.8638672	best: 12.8638672 (340)	total: 12.1s	remaining: 2m 9s
    341:	learn: 13.8731300	test: 12.8534253	best: 12.8534253 (341)	total: 12.1s	remaining: 2m 9s
    342:	learn: 13.8697199	test: 12.8535373	best: 12.8534253 (341)	total: 12.2s	remaining: 2m 9s
    343:	learn: 13.8635187	test: 12.8489749	best: 12.8489749 (343)	total: 12.2s	remaining: 2m 9s
    344:	learn: 13.8613749	test: 12.8480543	best: 12.8480543 (344)	total: 12.3s	remaining: 2m 9s
    345:	learn: 13.8582296	test: 12.8446866	best: 12.8446866 (345)	total: 12.3s	remaining: 2m 9s
    346:	learn: 13.8539727	test: 12.8423773	best: 12.8423773 (346)	total: 12.3s	remaining: 2m 9s
    347:	learn: 13.8456431	test: 12.8403307	best: 12.8403307 (347)	total: 12.4s	remaining: 2m 9s
    348:	learn: 13.8396832	test: 12.8370940	best: 12.8370940 (348)	total: 12.4s	remaining: 2m 9s
    349:	learn: 13.8330425	test: 12.8342645	best: 12.8342645 (349)	total: 12.4s	remaining: 2m 9s
    350:	learn: 13.8238427	test: 12.8307696	best: 12.8307696 (350)	total: 12.5s	remaining: 2m 9s
    351:	learn: 13.8212288	test: 12.8303526	best: 12.8303526 (351)	total: 12.5s	remaining: 2m 9s
    352:	learn: 13.8086192	test: 12.8204841	best: 12.8204841 (352)	total: 12.5s	remaining: 2m 9s
    353:	learn: 13.7959382	test: 12.8076647	best: 12.8076647 (353)	total: 12.6s	remaining: 2m 9s
    354:	learn: 13.7872275	test: 12.8008891	best: 12.8008891 (354)	total: 12.6s	remaining: 2m 9s
    355:	learn: 13.7802653	test: 12.7929659	best: 12.7929659 (355)	total: 12.6s	remaining: 2m 9s
    356:	learn: 13.7730988	test: 12.7906917	best: 12.7906917 (356)	total: 12.7s	remaining: 2m 9s
    357:	learn: 13.7628034	test: 12.7826837	best: 12.7826837 (357)	total: 12.7s	remaining: 2m 8s
    358:	learn: 13.7575444	test: 12.7825017	best: 12.7825017 (358)	total: 12.7s	remaining: 2m 8s
    359:	learn: 13.7484601	test: 12.7765618	best: 12.7765618 (359)	total: 12.7s	remaining: 2m 8s
    360:	learn: 13.7457437	test: 12.7764166	best: 12.7764166 (360)	total: 12.8s	remaining: 2m 8s
    361:	learn: 13.7385012	test: 12.7738230	best: 12.7738230 (361)	total: 12.8s	remaining: 2m 8s
    362:	learn: 13.7288607	test: 12.7675087	best: 12.7675087 (362)	total: 12.8s	remaining: 2m 8s
    363:	learn: 13.7232243	test: 12.7675267	best: 12.7675087 (362)	total: 12.9s	remaining: 2m 8s
    364:	learn: 13.7144639	test: 12.7628342	best: 12.7628342 (364)	total: 12.9s	remaining: 2m 8s
    365:	learn: 13.7071941	test: 12.7568205	best: 12.7568205 (365)	total: 12.9s	remaining: 2m 8s
    366:	learn: 13.6968330	test: 12.7529087	best: 12.7529087 (366)	total: 13s	remaining: 2m 8s
    367:	learn: 13.6859879	test: 12.7451367	best: 12.7451367 (367)	total: 13s	remaining: 2m 8s
    368:	learn: 13.6788667	test: 12.7394276	best: 12.7394276 (368)	total: 13s	remaining: 2m 8s
    369:	learn: 13.6701993	test: 12.7339785	best: 12.7339785 (369)	total: 13.1s	remaining: 2m 8s
    370:	learn: 13.6630572	test: 12.7324113	best: 12.7324113 (370)	total: 13.1s	remaining: 2m 8s
    371:	learn: 13.6540289	test: 12.7273085	best: 12.7273085 (371)	total: 13.1s	remaining: 2m 7s
    372:	learn: 13.6441423	test: 12.7214037	best: 12.7214037 (372)	total: 13.1s	remaining: 2m 7s
    373:	learn: 13.6391203	test: 12.7185050	best: 12.7185050 (373)	total: 13.2s	remaining: 2m 7s
    374:	learn: 13.6332143	test: 12.7144370	best: 12.7144370 (374)	total: 13.2s	remaining: 2m 7s
    375:	learn: 13.6268426	test: 12.7118050	best: 12.7118050 (375)	total: 13.2s	remaining: 2m 7s
    376:	learn: 13.6188682	test: 12.7063855	best: 12.7063855 (376)	total: 13.3s	remaining: 2m 7s
    377:	learn: 13.6084727	test: 12.7023509	best: 12.7023509 (377)	total: 13.3s	remaining: 2m 7s
    378:	learn: 13.6028074	test: 12.6985934	best: 12.6985934 (378)	total: 13.3s	remaining: 2m 7s
    379:	learn: 13.5949293	test: 12.6953706	best: 12.6953706 (379)	total: 13.3s	remaining: 2m 7s
    380:	learn: 13.5888141	test: 12.6932267	best: 12.6932267 (380)	total: 13.4s	remaining: 2m 7s
    381:	learn: 13.5811162	test: 12.6898658	best: 12.6898658 (381)	total: 13.4s	remaining: 2m 7s
    382:	learn: 13.5766369	test: 12.6862097	best: 12.6862097 (382)	total: 13.4s	remaining: 2m 7s
    383:	learn: 13.5685054	test: 12.6811381	best: 12.6811381 (383)	total: 13.5s	remaining: 2m 6s
    384:	learn: 13.5637836	test: 12.6807435	best: 12.6807435 (384)	total: 13.5s	remaining: 2m 6s
    385:	learn: 13.5620267	test: 12.6822928	best: 12.6807435 (384)	total: 13.5s	remaining: 2m 6s
    386:	learn: 13.5596482	test: 12.6817959	best: 12.6807435 (384)	total: 13.6s	remaining: 2m 6s
    387:	learn: 13.5564261	test: 12.6816017	best: 12.6807435 (384)	total: 13.6s	remaining: 2m 6s
    388:	learn: 13.5505516	test: 12.6768258	best: 12.6768258 (388)	total: 13.6s	remaining: 2m 6s
    389:	learn: 13.5427762	test: 12.6736361	best: 12.6736361 (389)	total: 13.6s	remaining: 2m 6s
    390:	learn: 13.5358338	test: 12.6695872	best: 12.6695872 (390)	total: 13.7s	remaining: 2m 6s
    391:	learn: 13.5314984	test: 12.6682166	best: 12.6682166 (391)	total: 13.7s	remaining: 2m 6s
    392:	learn: 13.5242267	test: 12.6627797	best: 12.6627797 (392)	total: 13.7s	remaining: 2m 6s
    393:	learn: 13.5175526	test: 12.6576223	best: 12.6576223 (393)	total: 13.8s	remaining: 2m 5s
    394:	learn: 13.5091202	test: 12.6529673	best: 12.6529673 (394)	total: 13.8s	remaining: 2m 5s
    395:	learn: 13.5055541	test: 12.6500563	best: 12.6500563 (395)	total: 13.8s	remaining: 2m 5s
    396:	learn: 13.5023166	test: 12.6464213	best: 12.6464213 (396)	total: 13.8s	remaining: 2m 5s
    397:	learn: 13.4943950	test: 12.6440427	best: 12.6440427 (397)	total: 13.9s	remaining: 2m 5s
    398:	learn: 13.4917425	test: 12.6433423	best: 12.6433423 (398)	total: 13.9s	remaining: 2m 5s
    399:	learn: 13.4850112	test: 12.6399235	best: 12.6399235 (399)	total: 13.9s	remaining: 2m 5s
    400:	learn: 13.4780839	test: 12.6373302	best: 12.6373302 (400)	total: 13.9s	remaining: 2m 5s
    401:	learn: 13.4705397	test: 12.6361140	best: 12.6361140 (401)	total: 14s	remaining: 2m 5s
    402:	learn: 13.4655530	test: 12.6344056	best: 12.6344056 (402)	total: 14s	remaining: 2m 4s
    403:	learn: 13.4625601	test: 12.6336884	best: 12.6336884 (403)	total: 14s	remaining: 2m 4s
    404:	learn: 13.4536919	test: 12.6264171	best: 12.6264171 (404)	total: 14.1s	remaining: 2m 4s
    405:	learn: 13.4523358	test: 12.6251242	best: 12.6251242 (405)	total: 14.1s	remaining: 2m 4s
    406:	learn: 13.4500424	test: 12.6255909	best: 12.6251242 (405)	total: 14.1s	remaining: 2m 4s
    407:	learn: 13.4432919	test: 12.6229046	best: 12.6229046 (407)	total: 14.1s	remaining: 2m 4s
    408:	learn: 13.4405550	test: 12.6216907	best: 12.6216907 (408)	total: 14.2s	remaining: 2m 4s
    409:	learn: 13.4382541	test: 12.6208081	best: 12.6208081 (409)	total: 14.2s	remaining: 2m 4s
    410:	learn: 13.4321241	test: 12.6175719	best: 12.6175719 (410)	total: 14.2s	remaining: 2m 4s
    411:	learn: 13.4284500	test: 12.6145652	best: 12.6145652 (411)	total: 14.3s	remaining: 2m 4s
    412:	learn: 13.4253752	test: 12.6152758	best: 12.6145652 (411)	total: 14.3s	remaining: 2m 4s
    413:	learn: 13.4234595	test: 12.6149122	best: 12.6145652 (411)	total: 14.3s	remaining: 2m 3s
    414:	learn: 13.4161854	test: 12.6119474	best: 12.6119474 (414)	total: 14.3s	remaining: 2m 3s
    415:	learn: 13.4082794	test: 12.6041187	best: 12.6041187 (415)	total: 14.4s	remaining: 2m 3s
    416:	learn: 13.4046733	test: 12.6019624	best: 12.6019624 (416)	total: 14.4s	remaining: 2m 3s
    417:	learn: 13.3976277	test: 12.5976976	best: 12.5976976 (417)	total: 14.4s	remaining: 2m 3s
    418:	learn: 13.3947990	test: 12.5966883	best: 12.5966883 (418)	total: 14.5s	remaining: 2m 3s
    419:	learn: 13.3895040	test: 12.5935511	best: 12.5935511 (419)	total: 14.5s	remaining: 2m 3s
    420:	learn: 13.3876800	test: 12.5933686	best: 12.5933686 (420)	total: 14.5s	remaining: 2m 3s
    421:	learn: 13.3854648	test: 12.5925008	best: 12.5925008 (421)	total: 14.5s	remaining: 2m 3s
    422:	learn: 13.3741230	test: 12.5864141	best: 12.5864141 (422)	total: 14.6s	remaining: 2m 3s
    423:	learn: 13.3676361	test: 12.5826559	best: 12.5826559 (423)	total: 14.6s	remaining: 2m 3s
    424:	learn: 13.3657274	test: 12.5817677	best: 12.5817677 (424)	total: 14.6s	remaining: 2m 2s
    425:	learn: 13.3596821	test: 12.5791710	best: 12.5791710 (425)	total: 14.7s	remaining: 2m 3s
    426:	learn: 13.3565842	test: 12.5774162	best: 12.5774162 (426)	total: 14.7s	remaining: 2m 2s
    427:	learn: 13.3541010	test: 12.5762742	best: 12.5762742 (427)	total: 14.7s	remaining: 2m 2s
    428:	learn: 13.3520985	test: 12.5766189	best: 12.5762742 (427)	total: 14.8s	remaining: 2m 2s
    429:	learn: 13.3444453	test: 12.5734845	best: 12.5734845 (429)	total: 14.8s	remaining: 2m 2s
    430:	learn: 13.3376306	test: 12.5675913	best: 12.5675913 (430)	total: 14.8s	remaining: 2m 2s
    431:	learn: 13.3313722	test: 12.5652799	best: 12.5652799 (431)	total: 14.8s	remaining: 2m 2s
    432:	learn: 13.3270162	test: 12.5627631	best: 12.5627631 (432)	total: 14.9s	remaining: 2m 2s
    433:	learn: 13.3213691	test: 12.5581985	best: 12.5581985 (433)	total: 14.9s	remaining: 2m 2s
    434:	learn: 13.3176415	test: 12.5579434	best: 12.5579434 (434)	total: 14.9s	remaining: 2m 2s
    435:	learn: 13.3118988	test: 12.5536702	best: 12.5536702 (435)	total: 15s	remaining: 2m 2s
    436:	learn: 13.3090471	test: 12.5506933	best: 12.5506933 (436)	total: 15s	remaining: 2m 2s
    437:	learn: 13.3071657	test: 12.5501789	best: 12.5501789 (437)	total: 15s	remaining: 2m 2s
    438:	learn: 13.3053474	test: 12.5496033	best: 12.5496033 (438)	total: 15s	remaining: 2m 1s
    439:	learn: 13.3031324	test: 12.5492044	best: 12.5492044 (439)	total: 15.1s	remaining: 2m 1s
    440:	learn: 13.2941734	test: 12.5427150	best: 12.5427150 (440)	total: 15.1s	remaining: 2m 1s
    441:	learn: 13.2845936	test: 12.5401922	best: 12.5401922 (441)	total: 15.1s	remaining: 2m 1s
    442:	learn: 13.2762427	test: 12.5356285	best: 12.5356285 (442)	total: 15.1s	remaining: 2m 1s
    443:	learn: 13.2668458	test: 12.5324851	best: 12.5324851 (443)	total: 15.2s	remaining: 2m 1s
    444:	learn: 13.2652031	test: 12.5326045	best: 12.5324851 (443)	total: 15.2s	remaining: 2m 1s
    445:	learn: 13.2549361	test: 12.5264149	best: 12.5264149 (445)	total: 15.2s	remaining: 2m 1s
    446:	learn: 13.2528643	test: 12.5267626	best: 12.5264149 (445)	total: 15.3s	remaining: 2m 1s
    447:	learn: 13.2463408	test: 12.5240696	best: 12.5240696 (447)	total: 15.3s	remaining: 2m 1s
    448:	learn: 13.2423586	test: 12.5207655	best: 12.5207655 (448)	total: 15.3s	remaining: 2m 1s
    449:	learn: 13.2365757	test: 12.5155538	best: 12.5155538 (449)	total: 15.4s	remaining: 2m 1s
    450:	learn: 13.2357446	test: 12.5146894	best: 12.5146894 (450)	total: 15.4s	remaining: 2m 1s
    451:	learn: 13.2278049	test: 12.5087032	best: 12.5087032 (451)	total: 15.4s	remaining: 2m 1s
    452:	learn: 13.2207838	test: 12.5037881	best: 12.5037881 (452)	total: 15.5s	remaining: 2m 1s
    453:	learn: 13.2165670	test: 12.5029024	best: 12.5029024 (453)	total: 15.5s	remaining: 2m 1s
    454:	learn: 13.2097119	test: 12.4988500	best: 12.4988500 (454)	total: 15.5s	remaining: 2m 1s
    455:	learn: 13.2072064	test: 12.4975701	best: 12.4975701 (455)	total: 15.6s	remaining: 2m 1s
    456:	learn: 13.1981584	test: 12.4918016	best: 12.4918016 (456)	total: 15.6s	remaining: 2m
    457:	learn: 13.1912024	test: 12.4915843	best: 12.4915843 (457)	total: 15.6s	remaining: 2m
    458:	learn: 13.1873301	test: 12.4911052	best: 12.4911052 (458)	total: 15.7s	remaining: 2m
    459:	learn: 13.1833801	test: 12.4909802	best: 12.4909802 (459)	total: 15.7s	remaining: 2m
    460:	learn: 13.1815394	test: 12.4910985	best: 12.4909802 (459)	total: 15.7s	remaining: 2m
    461:	learn: 13.1755936	test: 12.4863975	best: 12.4863975 (461)	total: 15.7s	remaining: 2m
    462:	learn: 13.1684774	test: 12.4865311	best: 12.4863975 (461)	total: 15.8s	remaining: 2m
    463:	learn: 13.1609747	test: 12.4838717	best: 12.4838717 (463)	total: 15.8s	remaining: 2m
    464:	learn: 13.1532360	test: 12.4787177	best: 12.4787177 (464)	total: 15.8s	remaining: 2m
    465:	learn: 13.1449622	test: 12.4750101	best: 12.4750101 (465)	total: 15.9s	remaining: 2m
    466:	learn: 13.1420173	test: 12.4743209	best: 12.4743209 (466)	total: 15.9s	remaining: 2m
    467:	learn: 13.1365970	test: 12.4713882	best: 12.4713882 (467)	total: 15.9s	remaining: 2m
    468:	learn: 13.1295430	test: 12.4667003	best: 12.4667003 (468)	total: 15.9s	remaining: 2m
    469:	learn: 13.1279932	test: 12.4674979	best: 12.4667003 (468)	total: 16s	remaining: 1m 59s
    470:	learn: 13.1264234	test: 12.4668543	best: 12.4667003 (468)	total: 16s	remaining: 1m 59s
    471:	learn: 13.1210367	test: 12.4633383	best: 12.4633383 (471)	total: 16s	remaining: 1m 59s
    472:	learn: 13.1109969	test: 12.4580095	best: 12.4580095 (472)	total: 16.1s	remaining: 1m 59s
    473:	learn: 13.1086915	test: 12.4574551	best: 12.4574551 (473)	total: 16.1s	remaining: 1m 59s
    474:	learn: 13.1062988	test: 12.4560690	best: 12.4560690 (474)	total: 16.1s	remaining: 1m 59s
    475:	learn: 13.1024027	test: 12.4547859	best: 12.4547859 (475)	total: 16.1s	remaining: 1m 59s
    476:	learn: 13.0987322	test: 12.4546997	best: 12.4546997 (476)	total: 16.2s	remaining: 1m 59s
    477:	learn: 13.0935304	test: 12.4527589	best: 12.4527589 (477)	total: 16.2s	remaining: 1m 59s
    478:	learn: 13.0890330	test: 12.4507008	best: 12.4507008 (478)	total: 16.2s	remaining: 1m 59s
    479:	learn: 13.0847538	test: 12.4508773	best: 12.4507008 (478)	total: 16.3s	remaining: 1m 59s
    480:	learn: 13.0812873	test: 12.4501224	best: 12.4501224 (480)	total: 16.3s	remaining: 1m 59s
    481:	learn: 13.0771051	test: 12.4464441	best: 12.4464441 (481)	total: 16.3s	remaining: 1m 59s
    482:	learn: 13.0710109	test: 12.4450274	best: 12.4450274 (482)	total: 16.3s	remaining: 1m 59s
    483:	learn: 13.0619690	test: 12.4414554	best: 12.4414554 (483)	total: 16.4s	remaining: 1m 58s
    484:	learn: 13.0541779	test: 12.4339127	best: 12.4339127 (484)	total: 16.4s	remaining: 1m 58s
    485:	learn: 13.0493906	test: 12.4318515	best: 12.4318515 (485)	total: 16.4s	remaining: 1m 58s
    486:	learn: 13.0469425	test: 12.4297150	best: 12.4297150 (486)	total: 16.5s	remaining: 1m 58s
    487:	learn: 13.0413935	test: 12.4242180	best: 12.4242180 (487)	total: 16.5s	remaining: 1m 58s
    488:	learn: 13.0368085	test: 12.4192311	best: 12.4192311 (488)	total: 16.5s	remaining: 1m 58s
    489:	learn: 13.0294621	test: 12.4132969	best: 12.4132969 (489)	total: 16.5s	remaining: 1m 58s
    490:	learn: 13.0275277	test: 12.4115624	best: 12.4115624 (490)	total: 16.6s	remaining: 1m 58s
    491:	learn: 13.0239950	test: 12.4100168	best: 12.4100168 (491)	total: 16.6s	remaining: 1m 58s
    492:	learn: 13.0193425	test: 12.4092249	best: 12.4092249 (492)	total: 16.6s	remaining: 1m 58s
    493:	learn: 13.0151426	test: 12.4073059	best: 12.4073059 (493)	total: 16.7s	remaining: 1m 58s
    494:	learn: 13.0108815	test: 12.4043297	best: 12.4043297 (494)	total: 16.7s	remaining: 1m 58s
    495:	learn: 13.0056357	test: 12.4028861	best: 12.4028861 (495)	total: 16.7s	remaining: 1m 58s
    496:	learn: 13.0022449	test: 12.4011097	best: 12.4011097 (496)	total: 16.7s	remaining: 1m 57s
    497:	learn: 12.9992394	test: 12.3998670	best: 12.3998670 (497)	total: 16.8s	remaining: 1m 57s
    498:	learn: 12.9947022	test: 12.3976556	best: 12.3976556 (498)	total: 16.8s	remaining: 1m 57s
    499:	learn: 12.9901723	test: 12.3989114	best: 12.3976556 (498)	total: 16.8s	remaining: 1m 57s
    500:	learn: 12.9865650	test: 12.4006233	best: 12.3976556 (498)	total: 16.9s	remaining: 1m 57s
    501:	learn: 12.9819981	test: 12.3993832	best: 12.3976556 (498)	total: 16.9s	remaining: 1m 57s
    502:	learn: 12.9799877	test: 12.3991058	best: 12.3976556 (498)	total: 16.9s	remaining: 1m 57s
    503:	learn: 12.9765064	test: 12.4002353	best: 12.3976556 (498)	total: 16.9s	remaining: 1m 57s
    504:	learn: 12.9733019	test: 12.4003326	best: 12.3976556 (498)	total: 17s	remaining: 1m 57s
    505:	learn: 12.9690043	test: 12.3984817	best: 12.3976556 (498)	total: 17s	remaining: 1m 57s
    506:	learn: 12.9666441	test: 12.4001704	best: 12.3976556 (498)	total: 17s	remaining: 1m 57s
    507:	learn: 12.9594151	test: 12.3991422	best: 12.3976556 (498)	total: 17s	remaining: 1m 57s
    508:	learn: 12.9572673	test: 12.3982667	best: 12.3976556 (498)	total: 17.1s	remaining: 1m 57s
    509:	learn: 12.9533296	test: 12.3978093	best: 12.3976556 (498)	total: 17.1s	remaining: 1m 56s
    510:	learn: 12.9503735	test: 12.3948274	best: 12.3948274 (510)	total: 17.1s	remaining: 1m 56s
    511:	learn: 12.9484704	test: 12.3950815	best: 12.3948274 (510)	total: 17.2s	remaining: 1m 56s
    512:	learn: 12.9407506	test: 12.3932631	best: 12.3932631 (512)	total: 17.2s	remaining: 1m 56s
    513:	learn: 12.9330433	test: 12.3891703	best: 12.3891703 (513)	total: 17.2s	remaining: 1m 56s
    514:	learn: 12.9280552	test: 12.3886769	best: 12.3886769 (514)	total: 17.2s	remaining: 1m 56s
    515:	learn: 12.9262340	test: 12.3902806	best: 12.3886769 (514)	total: 17.3s	remaining: 1m 56s
    516:	learn: 12.9163099	test: 12.3859346	best: 12.3859346 (516)	total: 17.3s	remaining: 1m 56s
    517:	learn: 12.9080432	test: 12.3814041	best: 12.3814041 (517)	total: 17.3s	remaining: 1m 56s
    518:	learn: 12.9049763	test: 12.3792966	best: 12.3792966 (518)	total: 17.4s	remaining: 1m 56s
    519:	learn: 12.9002511	test: 12.3749824	best: 12.3749824 (519)	total: 17.4s	remaining: 1m 56s
    520:	learn: 12.8945313	test: 12.3742311	best: 12.3742311 (520)	total: 17.4s	remaining: 1m 56s
    521:	learn: 12.8917480	test: 12.3748198	best: 12.3742311 (520)	total: 17.4s	remaining: 1m 56s
    522:	learn: 12.8887286	test: 12.3759407	best: 12.3742311 (520)	total: 17.5s	remaining: 1m 56s
    523:	learn: 12.8841385	test: 12.3736512	best: 12.3736512 (523)	total: 17.5s	remaining: 1m 56s
    524:	learn: 12.8785337	test: 12.3728320	best: 12.3728320 (524)	total: 17.5s	remaining: 1m 56s
    525:	learn: 12.8687153	test: 12.3689842	best: 12.3689842 (525)	total: 17.6s	remaining: 1m 55s
    526:	learn: 12.8622665	test: 12.3671750	best: 12.3671750 (526)	total: 17.6s	remaining: 1m 55s
    527:	learn: 12.8612947	test: 12.3666110	best: 12.3666110 (527)	total: 17.6s	remaining: 1m 55s
    528:	learn: 12.8591774	test: 12.3659365	best: 12.3659365 (528)	total: 17.6s	remaining: 1m 55s
    529:	learn: 12.8576828	test: 12.3664633	best: 12.3659365 (528)	total: 17.7s	remaining: 1m 55s
    530:	learn: 12.8504076	test: 12.3583212	best: 12.3583212 (530)	total: 17.7s	remaining: 1m 55s
    531:	learn: 12.8485462	test: 12.3596190	best: 12.3583212 (530)	total: 17.7s	remaining: 1m 55s
    532:	learn: 12.8439104	test: 12.3578012	best: 12.3578012 (532)	total: 17.8s	remaining: 1m 55s
    533:	learn: 12.8420684	test: 12.3585583	best: 12.3578012 (532)	total: 17.8s	remaining: 1m 55s
    534:	learn: 12.8349291	test: 12.3536608	best: 12.3536608 (534)	total: 17.8s	remaining: 1m 55s
    535:	learn: 12.8296869	test: 12.3526600	best: 12.3526600 (535)	total: 17.8s	remaining: 1m 55s
    536:	learn: 12.8238701	test: 12.3480684	best: 12.3480684 (536)	total: 17.9s	remaining: 1m 55s
    537:	learn: 12.8176049	test: 12.3450084	best: 12.3450084 (537)	total: 17.9s	remaining: 1m 55s
    538:	learn: 12.8121061	test: 12.3410702	best: 12.3410702 (538)	total: 17.9s	remaining: 1m 55s
    539:	learn: 12.8079407	test: 12.3404114	best: 12.3404114 (539)	total: 18s	remaining: 1m 55s
    540:	learn: 12.8054174	test: 12.3398749	best: 12.3398749 (540)	total: 18s	remaining: 1m 55s
    541:	learn: 12.8010213	test: 12.3384781	best: 12.3384781 (541)	total: 18s	remaining: 1m 54s
    542:	learn: 12.7939691	test: 12.3349508	best: 12.3349508 (542)	total: 18s	remaining: 1m 54s
    543:	learn: 12.7905670	test: 12.3353586	best: 12.3349508 (542)	total: 18.1s	remaining: 1m 54s
    544:	learn: 12.7862807	test: 12.3333327	best: 12.3333327 (544)	total: 18.1s	remaining: 1m 54s
    545:	learn: 12.7793244	test: 12.3302728	best: 12.3302728 (545)	total: 18.2s	remaining: 1m 54s
    546:	learn: 12.7745013	test: 12.3305803	best: 12.3302728 (545)	total: 18.2s	remaining: 1m 54s
    547:	learn: 12.7716176	test: 12.3289100	best: 12.3289100 (547)	total: 18.2s	remaining: 1m 54s
    548:	learn: 12.7664579	test: 12.3244147	best: 12.3244147 (548)	total: 18.3s	remaining: 1m 54s
    549:	learn: 12.7612603	test: 12.3205561	best: 12.3205561 (549)	total: 18.3s	remaining: 1m 54s
    550:	learn: 12.7573343	test: 12.3198205	best: 12.3198205 (550)	total: 18.3s	remaining: 1m 54s
    551:	learn: 12.7543470	test: 12.3194763	best: 12.3194763 (551)	total: 18.4s	remaining: 1m 54s
    552:	learn: 12.7515295	test: 12.3186144	best: 12.3186144 (552)	total: 18.4s	remaining: 1m 54s
    553:	learn: 12.7450496	test: 12.3177769	best: 12.3177769 (553)	total: 18.4s	remaining: 1m 54s
    554:	learn: 12.7412845	test: 12.3151895	best: 12.3151895 (554)	total: 18.4s	remaining: 1m 54s
    555:	learn: 12.7344517	test: 12.3111843	best: 12.3111843 (555)	total: 18.5s	remaining: 1m 54s
    556:	learn: 12.7272322	test: 12.3103121	best: 12.3103121 (556)	total: 18.5s	remaining: 1m 54s
    557:	learn: 12.7176659	test: 12.3035681	best: 12.3035681 (557)	total: 18.5s	remaining: 1m 54s
    558:	learn: 12.7137679	test: 12.3019371	best: 12.3019371 (558)	total: 18.6s	remaining: 1m 54s
    559:	learn: 12.7130475	test: 12.3016828	best: 12.3016828 (559)	total: 18.6s	remaining: 1m 54s
    560:	learn: 12.7054914	test: 12.2954459	best: 12.2954459 (560)	total: 18.6s	remaining: 1m 54s
    561:	learn: 12.7033077	test: 12.2940805	best: 12.2940805 (561)	total: 18.6s	remaining: 1m 54s
    562:	learn: 12.6989745	test: 12.2941720	best: 12.2940805 (561)	total: 18.7s	remaining: 1m 53s
    563:	learn: 12.6961375	test: 12.2945717	best: 12.2940805 (561)	total: 18.7s	remaining: 1m 53s
    564:	learn: 12.6891956	test: 12.2911969	best: 12.2911969 (564)	total: 18.7s	remaining: 1m 53s
    565:	learn: 12.6857516	test: 12.2902071	best: 12.2902071 (565)	total: 18.8s	remaining: 1m 53s
    566:	learn: 12.6823544	test: 12.2913937	best: 12.2902071 (565)	total: 18.8s	remaining: 1m 53s
    567:	learn: 12.6767574	test: 12.2866764	best: 12.2866764 (567)	total: 18.9s	remaining: 1m 53s
    568:	learn: 12.6710651	test: 12.2861513	best: 12.2861513 (568)	total: 18.9s	remaining: 1m 54s
    569:	learn: 12.6690642	test: 12.2866789	best: 12.2861513 (568)	total: 19s	remaining: 1m 54s
    570:	learn: 12.6673913	test: 12.2864235	best: 12.2861513 (568)	total: 19s	remaining: 1m 54s
    571:	learn: 12.6647582	test: 12.2855983	best: 12.2855983 (571)	total: 19s	remaining: 1m 54s
    572:	learn: 12.6612406	test: 12.2859535	best: 12.2855983 (571)	total: 19.1s	remaining: 1m 53s
    573:	learn: 12.6563492	test: 12.2822367	best: 12.2822367 (573)	total: 19.1s	remaining: 1m 53s
    574:	learn: 12.6467102	test: 12.2760802	best: 12.2760802 (574)	total: 19.1s	remaining: 1m 53s
    575:	learn: 12.6415213	test: 12.2775045	best: 12.2760802 (574)	total: 19.1s	remaining: 1m 53s
    576:	learn: 12.6347747	test: 12.2747604	best: 12.2747604 (576)	total: 19.2s	remaining: 1m 53s
    577:	learn: 12.6244140	test: 12.2656051	best: 12.2656051 (577)	total: 19.2s	remaining: 1m 53s
    578:	learn: 12.6205986	test: 12.2642621	best: 12.2642621 (578)	total: 19.2s	remaining: 1m 53s
    579:	learn: 12.6156794	test: 12.2610560	best: 12.2610560 (579)	total: 19.2s	remaining: 1m 53s
    580:	learn: 12.6127745	test: 12.2592826	best: 12.2592826 (580)	total: 19.3s	remaining: 1m 53s
    581:	learn: 12.6063397	test: 12.2565826	best: 12.2565826 (581)	total: 19.3s	remaining: 1m 53s
    582:	learn: 12.6047885	test: 12.2587942	best: 12.2565826 (581)	total: 19.3s	remaining: 1m 53s
    583:	learn: 12.6007612	test: 12.2570500	best: 12.2565826 (581)	total: 19.3s	remaining: 1m 53s
    584:	learn: 12.5967891	test: 12.2555487	best: 12.2555487 (584)	total: 19.4s	remaining: 1m 53s
    585:	learn: 12.5918458	test: 12.2528601	best: 12.2528601 (585)	total: 19.4s	remaining: 1m 53s
    586:	learn: 12.5881342	test: 12.2496402	best: 12.2496402 (586)	total: 19.4s	remaining: 1m 53s
    587:	learn: 12.5870627	test: 12.2507063	best: 12.2496402 (586)	total: 19.5s	remaining: 1m 52s
    588:	learn: 12.5791967	test: 12.2467793	best: 12.2467793 (588)	total: 19.5s	remaining: 1m 52s
    589:	learn: 12.5714927	test: 12.2424506	best: 12.2424506 (589)	total: 19.5s	remaining: 1m 52s
    590:	learn: 12.5670044	test: 12.2425247	best: 12.2424506 (589)	total: 19.5s	remaining: 1m 52s
    591:	learn: 12.5642603	test: 12.2419054	best: 12.2419054 (591)	total: 19.6s	remaining: 1m 52s
    592:	learn: 12.5568929	test: 12.2361657	best: 12.2361657 (592)	total: 19.6s	remaining: 1m 52s
    593:	learn: 12.5552330	test: 12.2342162	best: 12.2342162 (593)	total: 19.6s	remaining: 1m 52s
    594:	learn: 12.5536416	test: 12.2333074	best: 12.2333074 (594)	total: 19.7s	remaining: 1m 52s
    595:	learn: 12.5516808	test: 12.2346173	best: 12.2333074 (594)	total: 19.7s	remaining: 1m 52s
    596:	learn: 12.5498254	test: 12.2322016	best: 12.2322016 (596)	total: 19.7s	remaining: 1m 52s
    597:	learn: 12.5456388	test: 12.2333271	best: 12.2322016 (596)	total: 19.7s	remaining: 1m 52s
    598:	learn: 12.5454366	test: 12.2340468	best: 12.2322016 (596)	total: 19.8s	remaining: 1m 52s
    599:	learn: 12.5443577	test: 12.2332783	best: 12.2322016 (596)	total: 19.8s	remaining: 1m 52s
    600:	learn: 12.5363058	test: 12.2302903	best: 12.2302903 (600)	total: 19.8s	remaining: 1m 52s
    601:	learn: 12.5285498	test: 12.2275760	best: 12.2275760 (601)	total: 19.8s	remaining: 1m 52s
    602:	learn: 12.5274519	test: 12.2272938	best: 12.2272938 (602)	total: 19.9s	remaining: 1m 51s
    603:	learn: 12.5228992	test: 12.2257970	best: 12.2257970 (603)	total: 19.9s	remaining: 1m 51s
    604:	learn: 12.5210324	test: 12.2257504	best: 12.2257504 (604)	total: 19.9s	remaining: 1m 51s
    605:	learn: 12.5178848	test: 12.2248467	best: 12.2248467 (605)	total: 20s	remaining: 1m 51s
    606:	learn: 12.5160613	test: 12.2255108	best: 12.2248467 (605)	total: 20s	remaining: 1m 51s
    607:	learn: 12.5132209	test: 12.2269578	best: 12.2248467 (605)	total: 20s	remaining: 1m 51s
    608:	learn: 12.5081807	test: 12.2270917	best: 12.2248467 (605)	total: 20s	remaining: 1m 51s
    609:	learn: 12.5063702	test: 12.2257186	best: 12.2248467 (605)	total: 20.1s	remaining: 1m 51s
    610:	learn: 12.5029218	test: 12.2236318	best: 12.2236318 (610)	total: 20.1s	remaining: 1m 51s
    611:	learn: 12.5011400	test: 12.2239764	best: 12.2236318 (610)	total: 20.1s	remaining: 1m 51s
    612:	learn: 12.4976656	test: 12.2234856	best: 12.2234856 (612)	total: 20.1s	remaining: 1m 51s
    613:	learn: 12.4913020	test: 12.2213499	best: 12.2213499 (613)	total: 20.2s	remaining: 1m 51s
    614:	learn: 12.4863232	test: 12.2194661	best: 12.2194661 (614)	total: 20.2s	remaining: 1m 51s
    615:	learn: 12.4810329	test: 12.2148268	best: 12.2148268 (615)	total: 20.2s	remaining: 1m 51s
    616:	learn: 12.4750262	test: 12.2129007	best: 12.2129007 (616)	total: 20.3s	remaining: 1m 51s
    617:	learn: 12.4676419	test: 12.2091652	best: 12.2091652 (617)	total: 20.3s	remaining: 1m 51s
    618:	learn: 12.4655757	test: 12.2081489	best: 12.2081489 (618)	total: 20.3s	remaining: 1m 51s
    619:	learn: 12.4637511	test: 12.2056477	best: 12.2056477 (619)	total: 20.4s	remaining: 1m 51s
    620:	learn: 12.4610533	test: 12.2044034	best: 12.2044034 (620)	total: 20.4s	remaining: 1m 51s
    621:	learn: 12.4582872	test: 12.2023795	best: 12.2023795 (621)	total: 20.5s	remaining: 1m 51s
    622:	learn: 12.4473083	test: 12.1956659	best: 12.1956659 (622)	total: 20.5s	remaining: 1m 51s
    623:	learn: 12.4423336	test: 12.1952386	best: 12.1952386 (623)	total: 20.6s	remaining: 1m 51s
    624:	learn: 12.4362901	test: 12.1919115	best: 12.1919115 (624)	total: 20.6s	remaining: 1m 51s
    625:	learn: 12.4336324	test: 12.1894526	best: 12.1894526 (625)	total: 20.7s	remaining: 1m 51s
    626:	learn: 12.4298886	test: 12.1876156	best: 12.1876156 (626)	total: 20.7s	remaining: 1m 51s
    627:	learn: 12.4247577	test: 12.1871494	best: 12.1871494 (627)	total: 20.7s	remaining: 1m 51s
    628:	learn: 12.4209493	test: 12.1851640	best: 12.1851640 (628)	total: 20.7s	remaining: 1m 51s
    629:	learn: 12.4164351	test: 12.1853278	best: 12.1851640 (628)	total: 20.8s	remaining: 1m 51s
    630:	learn: 12.4071671	test: 12.1795275	best: 12.1795275 (630)	total: 20.8s	remaining: 1m 51s
    631:	learn: 12.4030622	test: 12.1769804	best: 12.1769804 (631)	total: 20.8s	remaining: 1m 50s
    632:	learn: 12.3967852	test: 12.1757725	best: 12.1757725 (632)	total: 20.8s	remaining: 1m 50s
    633:	learn: 12.3928274	test: 12.1763934	best: 12.1757725 (632)	total: 20.9s	remaining: 1m 50s
    634:	learn: 12.3894748	test: 12.1752511	best: 12.1752511 (634)	total: 20.9s	remaining: 1m 50s
    635:	learn: 12.3855661	test: 12.1764478	best: 12.1752511 (634)	total: 20.9s	remaining: 1m 50s
    636:	learn: 12.3837747	test: 12.1761543	best: 12.1752511 (634)	total: 21s	remaining: 1m 50s
    637:	learn: 12.3808007	test: 12.1726274	best: 12.1726274 (637)	total: 21s	remaining: 1m 50s
    638:	learn: 12.3745229	test: 12.1696497	best: 12.1696497 (638)	total: 21s	remaining: 1m 50s
    639:	learn: 12.3732390	test: 12.1701462	best: 12.1696497 (638)	total: 21s	remaining: 1m 50s
    640:	learn: 12.3710571	test: 12.1683271	best: 12.1683271 (640)	total: 21.1s	remaining: 1m 50s
    641:	learn: 12.3688209	test: 12.1679173	best: 12.1679173 (641)	total: 21.1s	remaining: 1m 50s
    642:	learn: 12.3650114	test: 12.1683530	best: 12.1679173 (641)	total: 21.1s	remaining: 1m 50s
    643:	learn: 12.3621951	test: 12.1678284	best: 12.1678284 (643)	total: 21.2s	remaining: 1m 50s
    644:	learn: 12.3536712	test: 12.1654755	best: 12.1654755 (644)	total: 21.2s	remaining: 1m 50s
    645:	learn: 12.3522189	test: 12.1639529	best: 12.1639529 (645)	total: 21.2s	remaining: 1m 50s
    646:	learn: 12.3441858	test: 12.1623309	best: 12.1623309 (646)	total: 21.3s	remaining: 1m 50s
    647:	learn: 12.3422742	test: 12.1611014	best: 12.1611014 (647)	total: 21.3s	remaining: 1m 50s
    648:	learn: 12.3402776	test: 12.1603937	best: 12.1603937 (648)	total: 21.3s	remaining: 1m 50s
    649:	learn: 12.3328390	test: 12.1573098	best: 12.1573098 (649)	total: 21.3s	remaining: 1m 49s
    650:	learn: 12.3296682	test: 12.1575146	best: 12.1573098 (649)	total: 21.4s	remaining: 1m 49s
    651:	learn: 12.3276899	test: 12.1569855	best: 12.1569855 (651)	total: 21.4s	remaining: 1m 49s
    652:	learn: 12.3267200	test: 12.1565029	best: 12.1565029 (652)	total: 21.4s	remaining: 1m 49s
    653:	learn: 12.3223630	test: 12.1522818	best: 12.1522818 (653)	total: 21.5s	remaining: 1m 49s
    654:	learn: 12.3183777	test: 12.1498039	best: 12.1498039 (654)	total: 21.5s	remaining: 1m 49s
    655:	learn: 12.3151739	test: 12.1460364	best: 12.1460364 (655)	total: 21.5s	remaining: 1m 49s
    656:	learn: 12.3126575	test: 12.1428012	best: 12.1428012 (656)	total: 21.6s	remaining: 1m 49s
    657:	learn: 12.3108679	test: 12.1425891	best: 12.1425891 (657)	total: 21.6s	remaining: 1m 49s
    658:	learn: 12.3092064	test: 12.1415760	best: 12.1415760 (658)	total: 21.6s	remaining: 1m 49s
    659:	learn: 12.3021903	test: 12.1405446	best: 12.1405446 (659)	total: 21.7s	remaining: 1m 49s
    660:	learn: 12.2970450	test: 12.1391492	best: 12.1391492 (660)	total: 21.7s	remaining: 1m 49s
    661:	learn: 12.2933521	test: 12.1392495	best: 12.1391492 (660)	total: 21.7s	remaining: 1m 49s
    662:	learn: 12.2894974	test: 12.1398047	best: 12.1391492 (660)	total: 21.8s	remaining: 1m 49s
    663:	learn: 12.2872445	test: 12.1395003	best: 12.1391492 (660)	total: 21.8s	remaining: 1m 49s
    664:	learn: 12.2846428	test: 12.1369227	best: 12.1369227 (664)	total: 21.8s	remaining: 1m 49s
    665:	learn: 12.2823403	test: 12.1370640	best: 12.1369227 (664)	total: 21.8s	remaining: 1m 49s
    666:	learn: 12.2776722	test: 12.1362750	best: 12.1362750 (666)	total: 21.9s	remaining: 1m 49s
    667:	learn: 12.2701909	test: 12.1305351	best: 12.1305351 (667)	total: 21.9s	remaining: 1m 49s
    668:	learn: 12.2668750	test: 12.1314500	best: 12.1305351 (667)	total: 21.9s	remaining: 1m 49s
    669:	learn: 12.2639670	test: 12.1339140	best: 12.1305351 (667)	total: 21.9s	remaining: 1m 49s
    670:	learn: 12.2577641	test: 12.1267031	best: 12.1267031 (670)	total: 22s	remaining: 1m 48s
    671:	learn: 12.2555208	test: 12.1263227	best: 12.1263227 (671)	total: 22s	remaining: 1m 48s
    672:	learn: 12.2545999	test: 12.1259242	best: 12.1259242 (672)	total: 22s	remaining: 1m 48s
    673:	learn: 12.2531676	test: 12.1265744	best: 12.1259242 (672)	total: 22.1s	remaining: 1m 48s
    674:	learn: 12.2477537	test: 12.1214704	best: 12.1214704 (674)	total: 22.1s	remaining: 1m 48s
    675:	learn: 12.2429899	test: 12.1189983	best: 12.1189983 (675)	total: 22.1s	remaining: 1m 48s
    676:	learn: 12.2364624	test: 12.1158230	best: 12.1158230 (676)	total: 22.2s	remaining: 1m 48s
    677:	learn: 12.2318377	test: 12.1155064	best: 12.1155064 (677)	total: 22.2s	remaining: 1m 48s
    678:	learn: 12.2236862	test: 12.1124143	best: 12.1124143 (678)	total: 22.2s	remaining: 1m 48s
    679:	learn: 12.2218797	test: 12.1084721	best: 12.1084721 (679)	total: 22.3s	remaining: 1m 48s
    680:	learn: 12.2199710	test: 12.1076183	best: 12.1076183 (680)	total: 22.3s	remaining: 1m 48s
    681:	learn: 12.2137448	test: 12.1070950	best: 12.1070950 (681)	total: 22.4s	remaining: 1m 48s
    682:	learn: 12.2117646	test: 12.1065284	best: 12.1065284 (682)	total: 22.4s	remaining: 1m 48s
    683:	learn: 12.2099453	test: 12.1055681	best: 12.1055681 (683)	total: 22.5s	remaining: 1m 48s
    684:	learn: 12.2068420	test: 12.1052564	best: 12.1052564 (684)	total: 22.5s	remaining: 1m 48s
    685:	learn: 12.2034247	test: 12.1059229	best: 12.1052564 (684)	total: 22.5s	remaining: 1m 48s
    686:	learn: 12.2008391	test: 12.1058735	best: 12.1052564 (684)	total: 22.6s	remaining: 1m 48s
    687:	learn: 12.1992022	test: 12.1057557	best: 12.1052564 (684)	total: 22.6s	remaining: 1m 48s
    688:	learn: 12.1918927	test: 12.0995678	best: 12.0995678 (688)	total: 22.6s	remaining: 1m 48s
    689:	learn: 12.1849540	test: 12.0984825	best: 12.0984825 (689)	total: 22.6s	remaining: 1m 48s
    690:	learn: 12.1804360	test: 12.0949634	best: 12.0949634 (690)	total: 22.7s	remaining: 1m 48s
    691:	learn: 12.1790582	test: 12.0954822	best: 12.0949634 (690)	total: 22.7s	remaining: 1m 48s
    692:	learn: 12.1729694	test: 12.0867217	best: 12.0867217 (692)	total: 22.7s	remaining: 1m 48s
    693:	learn: 12.1692312	test: 12.0840534	best: 12.0840534 (693)	total: 22.8s	remaining: 1m 48s
    694:	learn: 12.1618657	test: 12.0875780	best: 12.0840534 (693)	total: 22.8s	remaining: 1m 48s
    695:	learn: 12.1563771	test: 12.0854628	best: 12.0840534 (693)	total: 22.8s	remaining: 1m 48s
    696:	learn: 12.1506534	test: 12.0850437	best: 12.0840534 (693)	total: 22.9s	remaining: 1m 48s
    697:	learn: 12.1488856	test: 12.0842193	best: 12.0840534 (693)	total: 22.9s	remaining: 1m 48s
    698:	learn: 12.1461732	test: 12.0854679	best: 12.0840534 (693)	total: 22.9s	remaining: 1m 48s
    699:	learn: 12.1392827	test: 12.0847560	best: 12.0840534 (693)	total: 23s	remaining: 1m 48s
    700:	learn: 12.1376997	test: 12.0850656	best: 12.0840534 (693)	total: 23s	remaining: 1m 48s
    701:	learn: 12.1345739	test: 12.0857200	best: 12.0840534 (693)	total: 23s	remaining: 1m 48s
    702:	learn: 12.1277591	test: 12.0825459	best: 12.0825459 (702)	total: 23s	remaining: 1m 48s
    703:	learn: 12.1244175	test: 12.0829628	best: 12.0825459 (702)	total: 23.1s	remaining: 1m 47s
    704:	learn: 12.1145900	test: 12.0767732	best: 12.0767732 (704)	total: 23.1s	remaining: 1m 47s
    705:	learn: 12.1130755	test: 12.0771192	best: 12.0767732 (704)	total: 23.1s	remaining: 1m 47s
    706:	learn: 12.1084242	test: 12.0748426	best: 12.0748426 (706)	total: 23.2s	remaining: 1m 47s
    707:	learn: 12.1044294	test: 12.0703920	best: 12.0703920 (707)	total: 23.2s	remaining: 1m 47s
    708:	learn: 12.1022709	test: 12.0701387	best: 12.0701387 (708)	total: 23.2s	remaining: 1m 47s
    709:	learn: 12.0986489	test: 12.0675945	best: 12.0675945 (709)	total: 23.2s	remaining: 1m 47s
    710:	learn: 12.0978351	test: 12.0693612	best: 12.0675945 (709)	total: 23.3s	remaining: 1m 47s
    711:	learn: 12.0945560	test: 12.0678786	best: 12.0675945 (709)	total: 23.3s	remaining: 1m 47s
    712:	learn: 12.0900622	test: 12.0660590	best: 12.0660590 (712)	total: 23.3s	remaining: 1m 47s
    713:	learn: 12.0883904	test: 12.0648118	best: 12.0648118 (713)	total: 23.4s	remaining: 1m 47s
    714:	learn: 12.0802256	test: 12.0585812	best: 12.0585812 (714)	total: 23.4s	remaining: 1m 47s
    715:	learn: 12.0782917	test: 12.0573811	best: 12.0573811 (715)	total: 23.4s	remaining: 1m 47s
    716:	learn: 12.0739948	test: 12.0578644	best: 12.0573811 (715)	total: 23.4s	remaining: 1m 47s
    717:	learn: 12.0708777	test: 12.0561117	best: 12.0561117 (717)	total: 23.5s	remaining: 1m 47s
    718:	learn: 12.0657222	test: 12.0562024	best: 12.0561117 (717)	total: 23.5s	remaining: 1m 47s
    719:	learn: 12.0612735	test: 12.0548240	best: 12.0548240 (719)	total: 23.5s	remaining: 1m 47s
    720:	learn: 12.0573227	test: 12.0558596	best: 12.0548240 (719)	total: 23.6s	remaining: 1m 47s
    721:	learn: 12.0537843	test: 12.0557577	best: 12.0548240 (719)	total: 23.6s	remaining: 1m 47s
    722:	learn: 12.0529077	test: 12.0561081	best: 12.0548240 (719)	total: 23.6s	remaining: 1m 47s
    723:	learn: 12.0518172	test: 12.0560543	best: 12.0548240 (719)	total: 23.6s	remaining: 1m 46s
    724:	learn: 12.0480654	test: 12.0539743	best: 12.0539743 (724)	total: 23.7s	remaining: 1m 46s
    725:	learn: 12.0457841	test: 12.0542678	best: 12.0539743 (724)	total: 23.7s	remaining: 1m 46s
    726:	learn: 12.0432248	test: 12.0537860	best: 12.0537860 (726)	total: 23.7s	remaining: 1m 46s
    727:	learn: 12.0407847	test: 12.0531514	best: 12.0531514 (727)	total: 23.8s	remaining: 1m 46s
    728:	learn: 12.0393540	test: 12.0532667	best: 12.0531514 (727)	total: 23.8s	remaining: 1m 46s
    729:	learn: 12.0380905	test: 12.0532653	best: 12.0531514 (727)	total: 23.8s	remaining: 1m 46s
    730:	learn: 12.0358575	test: 12.0554494	best: 12.0531514 (727)	total: 23.8s	remaining: 1m 46s
    731:	learn: 12.0349858	test: 12.0557065	best: 12.0531514 (727)	total: 23.9s	remaining: 1m 46s
    732:	learn: 12.0333436	test: 12.0557759	best: 12.0531514 (727)	total: 23.9s	remaining: 1m 46s
    733:	learn: 12.0281246	test: 12.0536788	best: 12.0531514 (727)	total: 23.9s	remaining: 1m 46s
    734:	learn: 12.0238478	test: 12.0547768	best: 12.0531514 (727)	total: 23.9s	remaining: 1m 46s
    735:	learn: 12.0215257	test: 12.0551314	best: 12.0531514 (727)	total: 24s	remaining: 1m 46s
    736:	learn: 12.0178055	test: 12.0562832	best: 12.0531514 (727)	total: 24s	remaining: 1m 46s
    737:	learn: 12.0143752	test: 12.0556904	best: 12.0531514 (727)	total: 24s	remaining: 1m 46s
    738:	learn: 12.0076426	test: 12.0534234	best: 12.0531514 (727)	total: 24.1s	remaining: 1m 46s
    739:	learn: 12.0058746	test: 12.0517762	best: 12.0517762 (739)	total: 24.1s	remaining: 1m 46s
    740:	learn: 12.0038393	test: 12.0513194	best: 12.0513194 (740)	total: 24.1s	remaining: 1m 46s
    741:	learn: 12.0011709	test: 12.0499876	best: 12.0499876 (741)	total: 24.1s	remaining: 1m 46s
    742:	learn: 11.9996252	test: 12.0495011	best: 12.0495011 (742)	total: 24.2s	remaining: 1m 45s
    743:	learn: 11.9976783	test: 12.0475226	best: 12.0475226 (743)	total: 24.2s	remaining: 1m 45s
    744:	learn: 11.9915856	test: 12.0456721	best: 12.0456721 (744)	total: 24.2s	remaining: 1m 45s
    745:	learn: 11.9898749	test: 12.0453565	best: 12.0453565 (745)	total: 24.3s	remaining: 1m 45s
    746:	learn: 11.9825248	test: 12.0432358	best: 12.0432358 (746)	total: 24.3s	remaining: 1m 45s
    747:	learn: 11.9806948	test: 12.0433871	best: 12.0432358 (746)	total: 24.3s	remaining: 1m 45s
    748:	learn: 11.9772941	test: 12.0438563	best: 12.0432358 (746)	total: 24.3s	remaining: 1m 45s
    749:	learn: 11.9700614	test: 12.0407651	best: 12.0407651 (749)	total: 24.4s	remaining: 1m 45s
    750:	learn: 11.9687369	test: 12.0394715	best: 12.0394715 (750)	total: 24.4s	remaining: 1m 45s
    751:	learn: 11.9649735	test: 12.0378388	best: 12.0378388 (751)	total: 24.4s	remaining: 1m 45s
    752:	learn: 11.9632280	test: 12.0383980	best: 12.0378388 (751)	total: 24.5s	remaining: 1m 45s
    753:	learn: 11.9619092	test: 12.0381339	best: 12.0378388 (751)	total: 24.5s	remaining: 1m 45s
    754:	learn: 11.9586863	test: 12.0389750	best: 12.0378388 (751)	total: 24.5s	remaining: 1m 45s
    755:	learn: 11.9548963	test: 12.0351611	best: 12.0351611 (755)	total: 24.5s	remaining: 1m 45s
    756:	learn: 11.9541781	test: 12.0346933	best: 12.0346933 (756)	total: 24.6s	remaining: 1m 45s
    757:	learn: 11.9475005	test: 12.0314615	best: 12.0314615 (757)	total: 24.6s	remaining: 1m 45s
    758:	learn: 11.9456917	test: 12.0309773	best: 12.0309773 (758)	total: 24.6s	remaining: 1m 45s
    759:	learn: 11.9429360	test: 12.0316593	best: 12.0309773 (758)	total: 24.7s	remaining: 1m 45s
    760:	learn: 11.9399651	test: 12.0307345	best: 12.0307345 (760)	total: 24.7s	remaining: 1m 45s
    761:	learn: 11.9367159	test: 12.0301476	best: 12.0301476 (761)	total: 24.7s	remaining: 1m 44s
    762:	learn: 11.9323232	test: 12.0258752	best: 12.0258752 (762)	total: 24.7s	remaining: 1m 44s
    763:	learn: 11.9312078	test: 12.0251437	best: 12.0251437 (763)	total: 24.8s	remaining: 1m 44s
    764:	learn: 11.9295306	test: 12.0246210	best: 12.0246210 (764)	total: 24.8s	remaining: 1m 44s
    765:	learn: 11.9257552	test: 12.0220104	best: 12.0220104 (765)	total: 24.8s	remaining: 1m 44s
    766:	learn: 11.9239055	test: 12.0235250	best: 12.0220104 (765)	total: 24.8s	remaining: 1m 44s
    767:	learn: 11.9204095	test: 12.0227987	best: 12.0220104 (765)	total: 24.9s	remaining: 1m 44s
    768:	learn: 11.9175683	test: 12.0225361	best: 12.0220104 (765)	total: 24.9s	remaining: 1m 44s
    769:	learn: 11.9144095	test: 12.0240839	best: 12.0220104 (765)	total: 24.9s	remaining: 1m 44s
    770:	learn: 11.9119894	test: 12.0234524	best: 12.0220104 (765)	total: 24.9s	remaining: 1m 44s
    771:	learn: 11.9109175	test: 12.0242541	best: 12.0220104 (765)	total: 25s	remaining: 1m 44s
    772:	learn: 11.9046893	test: 12.0226106	best: 12.0220104 (765)	total: 25s	remaining: 1m 44s
    773:	learn: 11.9019008	test: 12.0245066	best: 12.0220104 (765)	total: 25s	remaining: 1m 44s
    774:	learn: 11.8989865	test: 12.0236775	best: 12.0220104 (765)	total: 25.1s	remaining: 1m 44s
    775:	learn: 11.8930396	test: 12.0209103	best: 12.0209103 (775)	total: 25.1s	remaining: 1m 44s
    776:	learn: 11.8861984	test: 12.0231402	best: 12.0209103 (775)	total: 25.1s	remaining: 1m 44s
    777:	learn: 11.8835942	test: 12.0222123	best: 12.0209103 (775)	total: 25.1s	remaining: 1m 44s
    778:	learn: 11.8785839	test: 12.0198400	best: 12.0198400 (778)	total: 25.2s	remaining: 1m 44s
    779:	learn: 11.8756474	test: 12.0190161	best: 12.0190161 (779)	total: 25.2s	remaining: 1m 44s
    780:	learn: 11.8733068	test: 12.0189530	best: 12.0189530 (780)	total: 25.2s	remaining: 1m 43s
    781:	learn: 11.8657436	test: 12.0173272	best: 12.0173272 (781)	total: 25.3s	remaining: 1m 43s
    782:	learn: 11.8636398	test: 12.0178629	best: 12.0173272 (781)	total: 25.3s	remaining: 1m 43s
    783:	learn: 11.8614289	test: 12.0192674	best: 12.0173272 (781)	total: 25.3s	remaining: 1m 43s
    784:	learn: 11.8578598	test: 12.0154718	best: 12.0154718 (784)	total: 25.3s	remaining: 1m 43s
    785:	learn: 11.8540119	test: 12.0142521	best: 12.0142521 (785)	total: 25.4s	remaining: 1m 43s
    786:	learn: 11.8517158	test: 12.0146310	best: 12.0142521 (785)	total: 25.4s	remaining: 1m 43s
    787:	learn: 11.8513399	test: 12.0132362	best: 12.0132362 (787)	total: 25.4s	remaining: 1m 43s
    788:	learn: 11.8471149	test: 12.0127437	best: 12.0127437 (788)	total: 25.5s	remaining: 1m 43s
    789:	learn: 11.8450321	test: 12.0142447	best: 12.0127437 (788)	total: 25.5s	remaining: 1m 43s
    790:	learn: 11.8428972	test: 12.0133280	best: 12.0127437 (788)	total: 25.5s	remaining: 1m 43s
    791:	learn: 11.8347473	test: 12.0089295	best: 12.0089295 (791)	total: 25.5s	remaining: 1m 43s
    792:	learn: 11.8294262	test: 12.0070545	best: 12.0070545 (792)	total: 25.6s	remaining: 1m 43s
    793:	learn: 11.8279260	test: 12.0071173	best: 12.0070545 (792)	total: 25.6s	remaining: 1m 43s
    794:	learn: 11.8258801	test: 12.0072296	best: 12.0070545 (792)	total: 25.6s	remaining: 1m 43s
    795:	learn: 11.8212064	test: 12.0032936	best: 12.0032936 (795)	total: 25.7s	remaining: 1m 43s
    796:	learn: 11.8193135	test: 12.0023605	best: 12.0023605 (796)	total: 25.7s	remaining: 1m 43s
    797:	learn: 11.8139601	test: 11.9993213	best: 11.9993213 (797)	total: 25.7s	remaining: 1m 43s
    798:	learn: 11.8127556	test: 11.9982804	best: 11.9982804 (798)	total: 25.7s	remaining: 1m 43s
    799:	learn: 11.8117863	test: 11.9981707	best: 11.9981707 (799)	total: 25.8s	remaining: 1m 43s
    800:	learn: 11.8103324	test: 11.9966481	best: 11.9966481 (800)	total: 25.8s	remaining: 1m 43s
    801:	learn: 11.8078034	test: 11.9958255	best: 11.9958255 (801)	total: 25.8s	remaining: 1m 42s
    802:	learn: 11.8054133	test: 11.9947799	best: 11.9947799 (802)	total: 25.8s	remaining: 1m 42s
    803:	learn: 11.8019272	test: 11.9934328	best: 11.9934328 (803)	total: 25.9s	remaining: 1m 42s
    804:	learn: 11.7920163	test: 11.9879213	best: 11.9879213 (804)	total: 25.9s	remaining: 1m 42s
    805:	learn: 11.7899554	test: 11.9878520	best: 11.9878520 (805)	total: 25.9s	remaining: 1m 42s
    806:	learn: 11.7876396	test: 11.9863533	best: 11.9863533 (806)	total: 26s	remaining: 1m 42s
    807:	learn: 11.7849548	test: 11.9853609	best: 11.9853609 (807)	total: 26s	remaining: 1m 42s
    808:	learn: 11.7827162	test: 11.9840545	best: 11.9840545 (808)	total: 26s	remaining: 1m 42s
    809:	learn: 11.7803410	test: 11.9831917	best: 11.9831917 (809)	total: 26s	remaining: 1m 42s
    810:	learn: 11.7777874	test: 11.9826981	best: 11.9826981 (810)	total: 26.1s	remaining: 1m 42s
    811:	learn: 11.7769277	test: 11.9821043	best: 11.9821043 (811)	total: 26.1s	remaining: 1m 42s
    812:	learn: 11.7738650	test: 11.9836670	best: 11.9821043 (811)	total: 26.1s	remaining: 1m 42s
    813:	learn: 11.7672100	test: 11.9829756	best: 11.9821043 (811)	total: 26.2s	remaining: 1m 42s
    814:	learn: 11.7635824	test: 11.9798296	best: 11.9798296 (814)	total: 26.2s	remaining: 1m 42s
    815:	learn: 11.7621992	test: 11.9800696	best: 11.9798296 (814)	total: 26.2s	remaining: 1m 42s
    816:	learn: 11.7594151	test: 11.9786317	best: 11.9786317 (816)	total: 26.2s	remaining: 1m 42s
    817:	learn: 11.7581918	test: 11.9783027	best: 11.9783027 (817)	total: 26.3s	remaining: 1m 42s
    818:	learn: 11.7554172	test: 11.9744880	best: 11.9744880 (818)	total: 26.3s	remaining: 1m 42s
    819:	learn: 11.7540509	test: 11.9737483	best: 11.9737483 (819)	total: 26.3s	remaining: 1m 42s
    820:	learn: 11.7531969	test: 11.9730199	best: 11.9730199 (820)	total: 26.3s	remaining: 1m 41s
    821:	learn: 11.7517713	test: 11.9722945	best: 11.9722945 (821)	total: 26.4s	remaining: 1m 41s
    822:	learn: 11.7507341	test: 11.9731529	best: 11.9722945 (821)	total: 26.4s	remaining: 1m 41s
    823:	learn: 11.7493116	test: 11.9744634	best: 11.9722945 (821)	total: 26.4s	remaining: 1m 41s
    824:	learn: 11.7489708	test: 11.9751071	best: 11.9722945 (821)	total: 26.5s	remaining: 1m 41s
    825:	learn: 11.7476724	test: 11.9749417	best: 11.9722945 (821)	total: 26.5s	remaining: 1m 41s
    826:	learn: 11.7458662	test: 11.9757672	best: 11.9722945 (821)	total: 26.6s	remaining: 1m 41s
    827:	learn: 11.7435967	test: 11.9741081	best: 11.9722945 (821)	total: 26.6s	remaining: 1m 41s
    828:	learn: 11.7404805	test: 11.9741083	best: 11.9722945 (821)	total: 26.6s	remaining: 1m 41s
    829:	learn: 11.7367089	test: 11.9746269	best: 11.9722945 (821)	total: 26.7s	remaining: 1m 41s
    830:	learn: 11.7344507	test: 11.9737039	best: 11.9722945 (821)	total: 26.7s	remaining: 1m 41s
    831:	learn: 11.7336288	test: 11.9748778	best: 11.9722945 (821)	total: 26.7s	remaining: 1m 41s
    832:	learn: 11.7309778	test: 11.9775476	best: 11.9722945 (821)	total: 26.8s	remaining: 1m 41s
    833:	learn: 11.7279080	test: 11.9775416	best: 11.9722945 (821)	total: 26.8s	remaining: 1m 41s
    834:	learn: 11.7262271	test: 11.9770281	best: 11.9722945 (821)	total: 26.8s	remaining: 1m 41s
    835:	learn: 11.7240287	test: 11.9771694	best: 11.9722945 (821)	total: 26.8s	remaining: 1m 41s
    836:	learn: 11.7184837	test: 11.9771599	best: 11.9722945 (821)	total: 26.9s	remaining: 1m 41s
    837:	learn: 11.7165966	test: 11.9755266	best: 11.9722945 (821)	total: 26.9s	remaining: 1m 41s
    838:	learn: 11.7152486	test: 11.9746349	best: 11.9722945 (821)	total: 26.9s	remaining: 1m 41s
    839:	learn: 11.7116353	test: 11.9723847	best: 11.9722945 (821)	total: 26.9s	remaining: 1m 41s
    840:	learn: 11.7094744	test: 11.9723742	best: 11.9722945 (821)	total: 27s	remaining: 1m 41s
    841:	learn: 11.7080496	test: 11.9722371	best: 11.9722371 (841)	total: 27s	remaining: 1m 41s
    842:	learn: 11.7068699	test: 11.9719043	best: 11.9719043 (842)	total: 27s	remaining: 1m 41s
    843:	learn: 11.7061165	test: 11.9725587	best: 11.9719043 (842)	total: 27.1s	remaining: 1m 41s
    844:	learn: 11.6982091	test: 11.9696412	best: 11.9696412 (844)	total: 27.1s	remaining: 1m 41s
    845:	learn: 11.6927768	test: 11.9672942	best: 11.9672942 (845)	total: 27.1s	remaining: 1m 41s
    846:	learn: 11.6891668	test: 11.9668610	best: 11.9668610 (846)	total: 27.2s	remaining: 1m 41s
    847:	learn: 11.6863450	test: 11.9656282	best: 11.9656282 (847)	total: 27.2s	remaining: 1m 41s
    848:	learn: 11.6830989	test: 11.9672452	best: 11.9656282 (847)	total: 27.2s	remaining: 1m 40s
    849:	learn: 11.6814835	test: 11.9672480	best: 11.9656282 (847)	total: 27.2s	remaining: 1m 40s
    850:	learn: 11.6772384	test: 11.9665790	best: 11.9656282 (847)	total: 27.3s	remaining: 1m 40s
    851:	learn: 11.6758330	test: 11.9658564	best: 11.9656282 (847)	total: 27.3s	remaining: 1m 40s
    852:	learn: 11.6749451	test: 11.9676030	best: 11.9656282 (847)	total: 27.3s	remaining: 1m 40s
    853:	learn: 11.6721803	test: 11.9653617	best: 11.9653617 (853)	total: 27.3s	remaining: 1m 40s
    854:	learn: 11.6691974	test: 11.9658543	best: 11.9653617 (853)	total: 27.4s	remaining: 1m 40s
    855:	learn: 11.6652076	test: 11.9643493	best: 11.9643493 (855)	total: 27.4s	remaining: 1m 40s
    856:	learn: 11.6627446	test: 11.9630938	best: 11.9630938 (856)	total: 27.4s	remaining: 1m 40s
    857:	learn: 11.6611774	test: 11.9631502	best: 11.9630938 (856)	total: 27.4s	remaining: 1m 40s
    858:	learn: 11.6589058	test: 11.9641993	best: 11.9630938 (856)	total: 27.5s	remaining: 1m 40s
    859:	learn: 11.6572277	test: 11.9631087	best: 11.9630938 (856)	total: 27.5s	remaining: 1m 40s
    860:	learn: 11.6543387	test: 11.9632942	best: 11.9630938 (856)	total: 27.5s	remaining: 1m 40s
    861:	learn: 11.6518002	test: 11.9629329	best: 11.9629329 (861)	total: 27.6s	remaining: 1m 40s
    862:	learn: 11.6479771	test: 11.9612710	best: 11.9612710 (862)	total: 27.6s	remaining: 1m 40s
    863:	learn: 11.6455811	test: 11.9601622	best: 11.9601622 (863)	total: 27.6s	remaining: 1m 40s
    864:	learn: 11.6433780	test: 11.9583452	best: 11.9583452 (864)	total: 27.6s	remaining: 1m 40s
    865:	learn: 11.6410915	test: 11.9585499	best: 11.9583452 (864)	total: 27.7s	remaining: 1m 40s
    866:	learn: 11.6404309	test: 11.9592546	best: 11.9583452 (864)	total: 27.7s	remaining: 1m 40s
    867:	learn: 11.6355223	test: 11.9542878	best: 11.9542878 (867)	total: 27.7s	remaining: 1m 40s
    868:	learn: 11.6340730	test: 11.9529652	best: 11.9529652 (868)	total: 27.7s	remaining: 1m 39s
    869:	learn: 11.6318918	test: 11.9527011	best: 11.9527011 (869)	total: 27.8s	remaining: 1m 39s
    870:	learn: 11.6294832	test: 11.9537699	best: 11.9527011 (869)	total: 27.8s	remaining: 1m 39s
    871:	learn: 11.6227318	test: 11.9508130	best: 11.9508130 (871)	total: 27.8s	remaining: 1m 39s
    872:	learn: 11.6194493	test: 11.9500173	best: 11.9500173 (872)	total: 27.9s	remaining: 1m 39s
    873:	learn: 11.6168355	test: 11.9509328	best: 11.9500173 (872)	total: 27.9s	remaining: 1m 39s
    874:	learn: 11.6152861	test: 11.9507926	best: 11.9500173 (872)	total: 27.9s	remaining: 1m 39s
    875:	learn: 11.6113827	test: 11.9501471	best: 11.9500173 (872)	total: 27.9s	remaining: 1m 39s
    876:	learn: 11.6100473	test: 11.9509868	best: 11.9500173 (872)	total: 28s	remaining: 1m 39s
    877:	learn: 11.6087521	test: 11.9522780	best: 11.9500173 (872)	total: 28s	remaining: 1m 39s
    878:	learn: 11.6045065	test: 11.9517563	best: 11.9500173 (872)	total: 28s	remaining: 1m 39s
    879:	learn: 11.6003152	test: 11.9539545	best: 11.9500173 (872)	total: 28.1s	remaining: 1m 39s
    880:	learn: 11.5940272	test: 11.9535290	best: 11.9500173 (872)	total: 28.1s	remaining: 1m 39s
    881:	learn: 11.5922810	test: 11.9531921	best: 11.9500173 (872)	total: 28.1s	remaining: 1m 39s
    882:	learn: 11.5908232	test: 11.9525781	best: 11.9500173 (872)	total: 28.1s	remaining: 1m 39s
    883:	learn: 11.5887989	test: 11.9525655	best: 11.9500173 (872)	total: 28.2s	remaining: 1m 39s
    884:	learn: 11.5875664	test: 11.9527267	best: 11.9500173 (872)	total: 28.2s	remaining: 1m 39s
    885:	learn: 11.5862335	test: 11.9512097	best: 11.9500173 (872)	total: 28.2s	remaining: 1m 39s
    886:	learn: 11.5825842	test: 11.9520041	best: 11.9500173 (872)	total: 28.3s	remaining: 1m 39s
    887:	learn: 11.5785609	test: 11.9473119	best: 11.9473119 (887)	total: 28.3s	remaining: 1m 39s
    888:	learn: 11.5771454	test: 11.9462212	best: 11.9462212 (888)	total: 28.3s	remaining: 1m 39s
    889:	learn: 11.5746107	test: 11.9468904	best: 11.9462212 (888)	total: 28.3s	remaining: 1m 39s
    890:	learn: 11.5742328	test: 11.9468442	best: 11.9462212 (888)	total: 28.4s	remaining: 1m 38s
    891:	learn: 11.5740459	test: 11.9465233	best: 11.9462212 (888)	total: 28.4s	remaining: 1m 38s
    892:	learn: 11.5712127	test: 11.9483689	best: 11.9462212 (888)	total: 28.4s	remaining: 1m 38s
    893:	learn: 11.5687751	test: 11.9476937	best: 11.9462212 (888)	total: 28.4s	remaining: 1m 38s
    894:	learn: 11.5654275	test: 11.9482043	best: 11.9462212 (888)	total: 28.5s	remaining: 1m 38s
    895:	learn: 11.5638127	test: 11.9488562	best: 11.9462212 (888)	total: 28.5s	remaining: 1m 38s
    896:	learn: 11.5601078	test: 11.9485039	best: 11.9462212 (888)	total: 28.5s	remaining: 1m 38s
    897:	learn: 11.5591647	test: 11.9490602	best: 11.9462212 (888)	total: 28.6s	remaining: 1m 38s
    898:	learn: 11.5555507	test: 11.9506348	best: 11.9462212 (888)	total: 28.6s	remaining: 1m 38s
    899:	learn: 11.5535051	test: 11.9497349	best: 11.9462212 (888)	total: 28.6s	remaining: 1m 38s
    900:	learn: 11.5512898	test: 11.9514957	best: 11.9462212 (888)	total: 28.6s	remaining: 1m 38s
    901:	learn: 11.5502659	test: 11.9504197	best: 11.9462212 (888)	total: 28.7s	remaining: 1m 38s
    902:	learn: 11.5477298	test: 11.9485963	best: 11.9462212 (888)	total: 28.7s	remaining: 1m 38s
    903:	learn: 11.5435669	test: 11.9493972	best: 11.9462212 (888)	total: 28.7s	remaining: 1m 38s
    904:	learn: 11.5422754	test: 11.9489657	best: 11.9462212 (888)	total: 28.7s	remaining: 1m 38s
    905:	learn: 11.5410861	test: 11.9491055	best: 11.9462212 (888)	total: 28.8s	remaining: 1m 38s
    906:	learn: 11.5386564	test: 11.9484078	best: 11.9462212 (888)	total: 28.8s	remaining: 1m 38s
    907:	learn: 11.5380318	test: 11.9479222	best: 11.9462212 (888)	total: 28.8s	remaining: 1m 38s
    908:	learn: 11.5374911	test: 11.9487374	best: 11.9462212 (888)	total: 28.9s	remaining: 1m 38s
    909:	learn: 11.5336722	test: 11.9474588	best: 11.9462212 (888)	total: 28.9s	remaining: 1m 38s
    910:	learn: 11.5297958	test: 11.9461013	best: 11.9461013 (910)	total: 28.9s	remaining: 1m 38s
    911:	learn: 11.5281387	test: 11.9459266	best: 11.9459266 (911)	total: 28.9s	remaining: 1m 37s
    912:	learn: 11.5245645	test: 11.9474478	best: 11.9459266 (911)	total: 29s	remaining: 1m 37s
    913:	learn: 11.5200305	test: 11.9495524	best: 11.9459266 (911)	total: 29s	remaining: 1m 37s
    914:	learn: 11.5188900	test: 11.9503414	best: 11.9459266 (911)	total: 29s	remaining: 1m 37s
    915:	learn: 11.5182310	test: 11.9505894	best: 11.9459266 (911)	total: 29.1s	remaining: 1m 37s
    916:	learn: 11.5117948	test: 11.9495925	best: 11.9459266 (911)	total: 29.1s	remaining: 1m 37s
    917:	learn: 11.5096241	test: 11.9491456	best: 11.9459266 (911)	total: 29.1s	remaining: 1m 37s
    918:	learn: 11.5071445	test: 11.9468574	best: 11.9459266 (911)	total: 29.1s	remaining: 1m 37s
    919:	learn: 11.5055920	test: 11.9467575	best: 11.9459266 (911)	total: 29.2s	remaining: 1m 37s
    920:	learn: 11.5036534	test: 11.9463696	best: 11.9459266 (911)	total: 29.2s	remaining: 1m 37s
    921:	learn: 11.5015450	test: 11.9450780	best: 11.9450780 (921)	total: 29.2s	remaining: 1m 37s
    922:	learn: 11.4951044	test: 11.9433745	best: 11.9433745 (922)	total: 29.3s	remaining: 1m 37s
    923:	learn: 11.4941039	test: 11.9433037	best: 11.9433037 (923)	total: 29.3s	remaining: 1m 37s
    924:	learn: 11.4911993	test: 11.9429212	best: 11.9429212 (924)	total: 29.3s	remaining: 1m 37s
    925:	learn: 11.4901289	test: 11.9429993	best: 11.9429212 (924)	total: 29.3s	remaining: 1m 37s
    926:	learn: 11.4848364	test: 11.9409901	best: 11.9409901 (926)	total: 29.4s	remaining: 1m 37s
    927:	learn: 11.4820079	test: 11.9390677	best: 11.9390677 (927)	total: 29.4s	remaining: 1m 37s
    928:	learn: 11.4805838	test: 11.9387507	best: 11.9387507 (928)	total: 29.4s	remaining: 1m 37s
    929:	learn: 11.4782243	test: 11.9399198	best: 11.9387507 (928)	total: 29.4s	remaining: 1m 37s
    930:	learn: 11.4758173	test: 11.9402009	best: 11.9387507 (928)	total: 29.5s	remaining: 1m 37s
    931:	learn: 11.4720062	test: 11.9377558	best: 11.9377558 (931)	total: 29.5s	remaining: 1m 37s
    932:	learn: 11.4675672	test: 11.9354546	best: 11.9354546 (932)	total: 29.5s	remaining: 1m 37s
    933:	learn: 11.4664600	test: 11.9342456	best: 11.9342456 (933)	total: 29.6s	remaining: 1m 37s
    934:	learn: 11.4631147	test: 11.9342680	best: 11.9342456 (933)	total: 29.6s	remaining: 1m 36s
    935:	learn: 11.4608858	test: 11.9341039	best: 11.9341039 (935)	total: 29.6s	remaining: 1m 36s
    936:	learn: 11.4592526	test: 11.9338868	best: 11.9338868 (936)	total: 29.6s	remaining: 1m 36s
    937:	learn: 11.4581586	test: 11.9340869	best: 11.9338868 (936)	total: 29.7s	remaining: 1m 36s
    938:	learn: 11.4572293	test: 11.9346389	best: 11.9338868 (936)	total: 29.7s	remaining: 1m 36s
    939:	learn: 11.4516098	test: 11.9314725	best: 11.9314725 (939)	total: 29.7s	remaining: 1m 36s
    940:	learn: 11.4465880	test: 11.9287406	best: 11.9287406 (940)	total: 29.8s	remaining: 1m 36s
    941:	learn: 11.4431458	test: 11.9280829	best: 11.9280829 (941)	total: 29.8s	remaining: 1m 36s
    942:	learn: 11.4399038	test: 11.9274486	best: 11.9274486 (942)	total: 29.8s	remaining: 1m 36s
    943:	learn: 11.4386598	test: 11.9278350	best: 11.9274486 (942)	total: 29.9s	remaining: 1m 36s
    944:	learn: 11.4344860	test: 11.9257721	best: 11.9257721 (944)	total: 29.9s	remaining: 1m 36s
    945:	learn: 11.4316206	test: 11.9248408	best: 11.9248408 (945)	total: 29.9s	remaining: 1m 36s
    946:	learn: 11.4281165	test: 11.9257025	best: 11.9248408 (945)	total: 29.9s	remaining: 1m 36s
    947:	learn: 11.4207856	test: 11.9228844	best: 11.9228844 (947)	total: 30s	remaining: 1m 36s
    948:	learn: 11.4169146	test: 11.9204658	best: 11.9204658 (948)	total: 30s	remaining: 1m 36s
    949:	learn: 11.4126414	test: 11.9192335	best: 11.9192335 (949)	total: 30s	remaining: 1m 36s
    950:	learn: 11.4104158	test: 11.9230903	best: 11.9192335 (949)	total: 30.1s	remaining: 1m 36s
    951:	learn: 11.4080838	test: 11.9234712	best: 11.9192335 (949)	total: 30.1s	remaining: 1m 36s
    952:	learn: 11.4061593	test: 11.9241477	best: 11.9192335 (949)	total: 30.1s	remaining: 1m 36s
    953:	learn: 11.4030861	test: 11.9227896	best: 11.9192335 (949)	total: 30.1s	remaining: 1m 36s
    954:	learn: 11.4015497	test: 11.9229235	best: 11.9192335 (949)	total: 30.2s	remaining: 1m 36s
    955:	learn: 11.4004741	test: 11.9226548	best: 11.9192335 (949)	total: 30.2s	remaining: 1m 36s
    956:	learn: 11.3988484	test: 11.9222131	best: 11.9192335 (949)	total: 30.2s	remaining: 1m 36s
    957:	learn: 11.3961840	test: 11.9234217	best: 11.9192335 (949)	total: 30.2s	remaining: 1m 36s
    958:	learn: 11.3937082	test: 11.9243186	best: 11.9192335 (949)	total: 30.3s	remaining: 1m 36s
    959:	learn: 11.3916228	test: 11.9221164	best: 11.9192335 (949)	total: 30.3s	remaining: 1m 35s
    960:	learn: 11.3881474	test: 11.9219536	best: 11.9192335 (949)	total: 30.3s	remaining: 1m 35s
    961:	learn: 11.3821102	test: 11.9183137	best: 11.9183137 (961)	total: 30.4s	remaining: 1m 35s
    962:	learn: 11.3803711	test: 11.9172150	best: 11.9172150 (962)	total: 30.4s	remaining: 1m 35s
    963:	learn: 11.3775620	test: 11.9138020	best: 11.9138020 (963)	total: 30.4s	remaining: 1m 35s
    964:	learn: 11.3749823	test: 11.9123717	best: 11.9123717 (964)	total: 30.4s	remaining: 1m 35s
    965:	learn: 11.3745574	test: 11.9127817	best: 11.9123717 (964)	total: 30.5s	remaining: 1m 35s
    966:	learn: 11.3734554	test: 11.9131515	best: 11.9123717 (964)	total: 30.5s	remaining: 1m 35s
    967:	learn: 11.3719850	test: 11.9151486	best: 11.9123717 (964)	total: 30.5s	remaining: 1m 35s
    968:	learn: 11.3676104	test: 11.9106001	best: 11.9106001 (968)	total: 30.6s	remaining: 1m 35s
    969:	learn: 11.3669341	test: 11.9101970	best: 11.9101970 (969)	total: 30.6s	remaining: 1m 35s
    970:	learn: 11.3656431	test: 11.9095613	best: 11.9095613 (970)	total: 30.6s	remaining: 1m 35s
    971:	learn: 11.3642128	test: 11.9096245	best: 11.9095613 (970)	total: 30.6s	remaining: 1m 35s
    972:	learn: 11.3629121	test: 11.9100687	best: 11.9095613 (970)	total: 30.7s	remaining: 1m 35s
    973:	learn: 11.3624645	test: 11.9107747	best: 11.9095613 (970)	total: 30.7s	remaining: 1m 35s
    974:	learn: 11.3605832	test: 11.9101861	best: 11.9095613 (970)	total: 30.7s	remaining: 1m 35s
    975:	learn: 11.3594562	test: 11.9088444	best: 11.9088444 (975)	total: 30.7s	remaining: 1m 35s
    976:	learn: 11.3565486	test: 11.9091934	best: 11.9088444 (975)	total: 30.8s	remaining: 1m 35s
    977:	learn: 11.3556034	test: 11.9103125	best: 11.9088444 (975)	total: 30.8s	remaining: 1m 35s
    978:	learn: 11.3522796	test: 11.9098654	best: 11.9088444 (975)	total: 30.8s	remaining: 1m 35s
    979:	learn: 11.3501417	test: 11.9115470	best: 11.9088444 (975)	total: 30.9s	remaining: 1m 35s
    980:	learn: 11.3471485	test: 11.9111910	best: 11.9088444 (975)	total: 30.9s	remaining: 1m 35s
    981:	learn: 11.3407478	test: 11.9095231	best: 11.9088444 (975)	total: 30.9s	remaining: 1m 35s
    982:	learn: 11.3397753	test: 11.9113339	best: 11.9088444 (975)	total: 31s	remaining: 1m 35s
    983:	learn: 11.3384554	test: 11.9099786	best: 11.9088444 (975)	total: 31s	remaining: 1m 35s
    984:	learn: 11.3327161	test: 11.9064987	best: 11.9064987 (984)	total: 31s	remaining: 1m 34s
    985:	learn: 11.3317887	test: 11.9056317	best: 11.9056317 (985)	total: 31.1s	remaining: 1m 34s
    986:	learn: 11.3312979	test: 11.9049351	best: 11.9049351 (986)	total: 31.1s	remaining: 1m 34s
    987:	learn: 11.3295126	test: 11.9034647	best: 11.9034647 (987)	total: 31.1s	remaining: 1m 34s
    988:	learn: 11.3273763	test: 11.9028085	best: 11.9028085 (988)	total: 31.1s	remaining: 1m 34s
    989:	learn: 11.3238160	test: 11.9001568	best: 11.9001568 (989)	total: 31.2s	remaining: 1m 34s
    990:	learn: 11.3222857	test: 11.9000653	best: 11.9000653 (990)	total: 31.2s	remaining: 1m 34s
    991:	learn: 11.3208073	test: 11.9004001	best: 11.9000653 (990)	total: 31.2s	remaining: 1m 34s
    992:	learn: 11.3191542	test: 11.8979015	best: 11.8979015 (992)	total: 31.2s	remaining: 1m 34s
    993:	learn: 11.3151572	test: 11.8973141	best: 11.8973141 (993)	total: 31.3s	remaining: 1m 34s
    994:	learn: 11.3126275	test: 11.8942492	best: 11.8942492 (994)	total: 31.3s	remaining: 1m 34s
    995:	learn: 11.3083483	test: 11.8908704	best: 11.8908704 (995)	total: 31.3s	remaining: 1m 34s
    996:	learn: 11.3044760	test: 11.8910807	best: 11.8908704 (995)	total: 31.4s	remaining: 1m 34s
    997:	learn: 11.3023672	test: 11.8904096	best: 11.8904096 (997)	total: 31.4s	remaining: 1m 34s
    998:	learn: 11.2990046	test: 11.8897383	best: 11.8897383 (998)	total: 31.4s	remaining: 1m 34s
    999:	learn: 11.2940243	test: 11.8864198	best: 11.8864198 (999)	total: 31.4s	remaining: 1m 34s
    1000:	learn: 11.2898955	test: 11.8888931	best: 11.8864198 (999)	total: 31.5s	remaining: 1m 34s
    1001:	learn: 11.2881188	test: 11.8903582	best: 11.8864198 (999)	total: 31.5s	remaining: 1m 34s
    1002:	learn: 11.2874484	test: 11.8899049	best: 11.8864198 (999)	total: 31.5s	remaining: 1m 34s
    1003:	learn: 11.2799327	test: 11.8861695	best: 11.8861695 (1003)	total: 31.6s	remaining: 1m 34s
    1004:	learn: 11.2789195	test: 11.8857363	best: 11.8857363 (1004)	total: 31.6s	remaining: 1m 34s
    1005:	learn: 11.2760493	test: 11.8861063	best: 11.8857363 (1004)	total: 31.6s	remaining: 1m 34s
    1006:	learn: 11.2707974	test: 11.8835648	best: 11.8835648 (1006)	total: 31.6s	remaining: 1m 34s
    1007:	learn: 11.2654839	test: 11.8817515	best: 11.8817515 (1007)	total: 31.7s	remaining: 1m 33s
    1008:	learn: 11.2619826	test: 11.8800278	best: 11.8800278 (1008)	total: 31.7s	remaining: 1m 33s
    1009:	learn: 11.2580009	test: 11.8791045	best: 11.8791045 (1009)	total: 31.7s	remaining: 1m 33s
    1010:	learn: 11.2547101	test: 11.8744628	best: 11.8744628 (1010)	total: 31.7s	remaining: 1m 33s
    1011:	learn: 11.2529577	test: 11.8757222	best: 11.8744628 (1010)	total: 31.8s	remaining: 1m 33s
    1012:	learn: 11.2493717	test: 11.8770563	best: 11.8744628 (1010)	total: 31.8s	remaining: 1m 33s
    1013:	learn: 11.2455104	test: 11.8761592	best: 11.8744628 (1010)	total: 31.8s	remaining: 1m 33s
    1014:	learn: 11.2423148	test: 11.8761510	best: 11.8744628 (1010)	total: 31.9s	remaining: 1m 33s
    1015:	learn: 11.2401975	test: 11.8755096	best: 11.8744628 (1010)	total: 31.9s	remaining: 1m 33s
    1016:	learn: 11.2382259	test: 11.8761737	best: 11.8744628 (1010)	total: 31.9s	remaining: 1m 33s
    1017:	learn: 11.2336220	test: 11.8731350	best: 11.8731350 (1017)	total: 31.9s	remaining: 1m 33s
    1018:	learn: 11.2314135	test: 11.8725847	best: 11.8725847 (1018)	total: 32s	remaining: 1m 33s
    1019:	learn: 11.2255234	test: 11.8711355	best: 11.8711355 (1019)	total: 32s	remaining: 1m 33s
    1020:	learn: 11.2225796	test: 11.8716537	best: 11.8711355 (1019)	total: 32s	remaining: 1m 33s
    1021:	learn: 11.2204974	test: 11.8714803	best: 11.8711355 (1019)	total: 32.1s	remaining: 1m 33s
    1022:	learn: 11.2183957	test: 11.8701540	best: 11.8701540 (1022)	total: 32.1s	remaining: 1m 33s
    1023:	learn: 11.2124521	test: 11.8665792	best: 11.8665792 (1023)	total: 32.1s	remaining: 1m 33s
    1024:	learn: 11.2089709	test: 11.8682597	best: 11.8665792 (1023)	total: 32.2s	remaining: 1m 33s
    1025:	learn: 11.2074813	test: 11.8676568	best: 11.8665792 (1023)	total: 32.2s	remaining: 1m 33s
    1026:	learn: 11.2068152	test: 11.8678415	best: 11.8665792 (1023)	total: 32.3s	remaining: 1m 33s
    1027:	learn: 11.2047714	test: 11.8688348	best: 11.8665792 (1023)	total: 32.3s	remaining: 1m 33s
    1028:	learn: 11.2017218	test: 11.8653849	best: 11.8653849 (1028)	total: 32.4s	remaining: 1m 33s
    1029:	learn: 11.1971177	test: 11.8618410	best: 11.8618410 (1029)	total: 32.4s	remaining: 1m 33s
    1030:	learn: 11.1959562	test: 11.8627240	best: 11.8618410 (1029)	total: 32.5s	remaining: 1m 33s
    1031:	learn: 11.1939682	test: 11.8629304	best: 11.8618410 (1029)	total: 32.5s	remaining: 1m 33s
    1032:	learn: 11.1882471	test: 11.8616017	best: 11.8616017 (1032)	total: 32.5s	remaining: 1m 33s
    1033:	learn: 11.1871764	test: 11.8612013	best: 11.8612013 (1033)	total: 32.6s	remaining: 1m 33s
    1034:	learn: 11.1861896	test: 11.8613688	best: 11.8612013 (1033)	total: 32.6s	remaining: 1m 33s
    1035:	learn: 11.1826704	test: 11.8595634	best: 11.8595634 (1035)	total: 32.6s	remaining: 1m 33s
    1036:	learn: 11.1800482	test: 11.8579595	best: 11.8579595 (1036)	total: 32.6s	remaining: 1m 33s
    1037:	learn: 11.1783079	test: 11.8579533	best: 11.8579533 (1037)	total: 32.6s	remaining: 1m 33s
    1038:	learn: 11.1775000	test: 11.8561288	best: 11.8561288 (1038)	total: 32.7s	remaining: 1m 33s
    1039:	learn: 11.1759635	test: 11.8550930	best: 11.8550930 (1039)	total: 32.7s	remaining: 1m 33s
    1040:	learn: 11.1742075	test: 11.8545780	best: 11.8545780 (1040)	total: 32.7s	remaining: 1m 33s
    1041:	learn: 11.1723560	test: 11.8550395	best: 11.8545780 (1040)	total: 32.8s	remaining: 1m 32s
    1042:	learn: 11.1719996	test: 11.8553554	best: 11.8545780 (1040)	total: 32.8s	remaining: 1m 32s
    1043:	learn: 11.1711288	test: 11.8551767	best: 11.8545780 (1040)	total: 32.8s	remaining: 1m 32s
    1044:	learn: 11.1679393	test: 11.8533238	best: 11.8533238 (1044)	total: 32.8s	remaining: 1m 32s
    1045:	learn: 11.1667817	test: 11.8526662	best: 11.8526662 (1045)	total: 32.9s	remaining: 1m 32s
    1046:	learn: 11.1642853	test: 11.8513365	best: 11.8513365 (1046)	total: 32.9s	remaining: 1m 32s
    1047:	learn: 11.1624545	test: 11.8524552	best: 11.8513365 (1046)	total: 32.9s	remaining: 1m 32s
    1048:	learn: 11.1606224	test: 11.8529500	best: 11.8513365 (1046)	total: 33s	remaining: 1m 32s
    1049:	learn: 11.1590331	test: 11.8534191	best: 11.8513365 (1046)	total: 33s	remaining: 1m 32s
    1050:	learn: 11.1571963	test: 11.8524224	best: 11.8513365 (1046)	total: 33s	remaining: 1m 32s
    1051:	learn: 11.1550399	test: 11.8513898	best: 11.8513365 (1046)	total: 33s	remaining: 1m 32s
    1052:	learn: 11.1537520	test: 11.8514688	best: 11.8513365 (1046)	total: 33.1s	remaining: 1m 32s
    1053:	learn: 11.1527136	test: 11.8510884	best: 11.8510884 (1053)	total: 33.1s	remaining: 1m 32s
    1054:	learn: 11.1502645	test: 11.8504287	best: 11.8504287 (1054)	total: 33.1s	remaining: 1m 32s
    1055:	learn: 11.1490907	test: 11.8508160	best: 11.8504287 (1054)	total: 33.2s	remaining: 1m 32s
    1056:	learn: 11.1476837	test: 11.8500868	best: 11.8500868 (1056)	total: 33.2s	remaining: 1m 32s
    1057:	learn: 11.1449470	test: 11.8497536	best: 11.8497536 (1057)	total: 33.2s	remaining: 1m 32s
    1058:	learn: 11.1418022	test: 11.8488252	best: 11.8488252 (1058)	total: 33.3s	remaining: 1m 32s
    1059:	learn: 11.1412556	test: 11.8490664	best: 11.8488252 (1058)	total: 33.3s	remaining: 1m 32s
    1060:	learn: 11.1401696	test: 11.8483004	best: 11.8483004 (1060)	total: 33.3s	remaining: 1m 32s
    1061:	learn: 11.1381159	test: 11.8488055	best: 11.8483004 (1060)	total: 33.3s	remaining: 1m 32s
    1062:	learn: 11.1368517	test: 11.8484261	best: 11.8483004 (1060)	total: 33.4s	remaining: 1m 32s
    1063:	learn: 11.1319048	test: 11.8479587	best: 11.8479587 (1063)	total: 33.4s	remaining: 1m 32s
    1064:	learn: 11.1293939	test: 11.8472341	best: 11.8472341 (1064)	total: 33.4s	remaining: 1m 32s
    1065:	learn: 11.1249496	test: 11.8468547	best: 11.8468547 (1065)	total: 33.5s	remaining: 1m 32s
    1066:	learn: 11.1221864	test: 11.8466826	best: 11.8466826 (1066)	total: 33.5s	remaining: 1m 32s
    1067:	learn: 11.1204323	test: 11.8461060	best: 11.8461060 (1067)	total: 33.5s	remaining: 1m 32s
    1068:	learn: 11.1194097	test: 11.8466813	best: 11.8461060 (1067)	total: 33.6s	remaining: 1m 32s
    1069:	learn: 11.1166015	test: 11.8439183	best: 11.8439183 (1069)	total: 33.6s	remaining: 1m 32s
    1070:	learn: 11.1153357	test: 11.8435630	best: 11.8435630 (1070)	total: 33.6s	remaining: 1m 31s
    1071:	learn: 11.1143716	test: 11.8435192	best: 11.8435192 (1071)	total: 33.7s	remaining: 1m 31s
    1072:	learn: 11.1120673	test: 11.8425999	best: 11.8425999 (1072)	total: 33.7s	remaining: 1m 31s
    1073:	learn: 11.1104550	test: 11.8408559	best: 11.8408559 (1073)	total: 33.7s	remaining: 1m 31s
    1074:	learn: 11.1069752	test: 11.8396091	best: 11.8396091 (1074)	total: 33.8s	remaining: 1m 31s
    1075:	learn: 11.1050136	test: 11.8389784	best: 11.8389784 (1075)	total: 33.8s	remaining: 1m 31s
    1076:	learn: 11.1024510	test: 11.8398152	best: 11.8389784 (1075)	total: 33.8s	remaining: 1m 31s
    1077:	learn: 11.0980931	test: 11.8382451	best: 11.8382451 (1077)	total: 33.9s	remaining: 1m 31s
    1078:	learn: 11.0956920	test: 11.8382499	best: 11.8382451 (1077)	total: 33.9s	remaining: 1m 31s
    1079:	learn: 11.0953276	test: 11.8379035	best: 11.8379035 (1079)	total: 33.9s	remaining: 1m 31s
    1080:	learn: 11.0922381	test: 11.8385421	best: 11.8379035 (1079)	total: 34s	remaining: 1m 31s
    1081:	learn: 11.0888204	test: 11.8372088	best: 11.8372088 (1081)	total: 34s	remaining: 1m 31s
    1082:	learn: 11.0859876	test: 11.8379718	best: 11.8372088 (1081)	total: 34s	remaining: 1m 31s
    1083:	learn: 11.0820126	test: 11.8355033	best: 11.8355033 (1083)	total: 34.1s	remaining: 1m 31s
    1084:	learn: 11.0792505	test: 11.8368467	best: 11.8355033 (1083)	total: 34.1s	remaining: 1m 31s
    1085:	learn: 11.0757108	test: 11.8372175	best: 11.8355033 (1083)	total: 34.1s	remaining: 1m 31s
    1086:	learn: 11.0746696	test: 11.8364708	best: 11.8355033 (1083)	total: 34.1s	remaining: 1m 31s
    1087:	learn: 11.0731141	test: 11.8352805	best: 11.8352805 (1087)	total: 34.2s	remaining: 1m 31s
    1088:	learn: 11.0719969	test: 11.8339606	best: 11.8339606 (1088)	total: 34.2s	remaining: 1m 31s
    1089:	learn: 11.0689928	test: 11.8352149	best: 11.8339606 (1088)	total: 34.2s	remaining: 1m 31s
    1090:	learn: 11.0680240	test: 11.8329953	best: 11.8329953 (1090)	total: 34.3s	remaining: 1m 31s
    1091:	learn: 11.0672695	test: 11.8323743	best: 11.8323743 (1091)	total: 34.3s	remaining: 1m 31s
    1092:	learn: 11.0655298	test: 11.8319580	best: 11.8319580 (1092)	total: 34.3s	remaining: 1m 31s
    1093:	learn: 11.0636268	test: 11.8329048	best: 11.8319580 (1092)	total: 34.4s	remaining: 1m 31s
    1094:	learn: 11.0605243	test: 11.8343784	best: 11.8319580 (1092)	total: 34.4s	remaining: 1m 31s
    1095:	learn: 11.0562043	test: 11.8347735	best: 11.8319580 (1092)	total: 34.4s	remaining: 1m 31s
    1096:	learn: 11.0540119	test: 11.8345965	best: 11.8319580 (1092)	total: 34.5s	remaining: 1m 31s
    1097:	learn: 11.0535866	test: 11.8353456	best: 11.8319580 (1092)	total: 34.5s	remaining: 1m 31s
    1098:	learn: 11.0531336	test: 11.8344007	best: 11.8319580 (1092)	total: 34.5s	remaining: 1m 31s
    1099:	learn: 11.0505509	test: 11.8329378	best: 11.8319580 (1092)	total: 34.6s	remaining: 1m 31s
    1100:	learn: 11.0456827	test: 11.8309393	best: 11.8309393 (1100)	total: 34.6s	remaining: 1m 31s
    1101:	learn: 11.0433066	test: 11.8316524	best: 11.8309393 (1100)	total: 34.6s	remaining: 1m 31s
    1102:	learn: 11.0419953	test: 11.8305701	best: 11.8305701 (1102)	total: 34.7s	remaining: 1m 31s
    1103:	learn: 11.0401085	test: 11.8296462	best: 11.8296462 (1103)	total: 34.7s	remaining: 1m 31s
    1104:	learn: 11.0350795	test: 11.8283212	best: 11.8283212 (1104)	total: 34.7s	remaining: 1m 30s
    1105:	learn: 11.0349226	test: 11.8285085	best: 11.8283212 (1104)	total: 34.8s	remaining: 1m 30s
    1106:	learn: 11.0341440	test: 11.8288081	best: 11.8283212 (1104)	total: 34.8s	remaining: 1m 30s
    1107:	learn: 11.0320819	test: 11.8294114	best: 11.8283212 (1104)	total: 34.8s	remaining: 1m 30s
    1108:	learn: 11.0303550	test: 11.8309956	best: 11.8283212 (1104)	total: 34.8s	remaining: 1m 30s
    1109:	learn: 11.0269448	test: 11.8320733	best: 11.8283212 (1104)	total: 34.9s	remaining: 1m 30s
    1110:	learn: 11.0261063	test: 11.8326097	best: 11.8283212 (1104)	total: 34.9s	remaining: 1m 30s
    1111:	learn: 11.0247862	test: 11.8330378	best: 11.8283212 (1104)	total: 35s	remaining: 1m 30s
    1112:	learn: 11.0222740	test: 11.8331677	best: 11.8283212 (1104)	total: 35s	remaining: 1m 30s
    1113:	learn: 11.0192406	test: 11.8319196	best: 11.8283212 (1104)	total: 35s	remaining: 1m 30s
    1114:	learn: 11.0182968	test: 11.8303221	best: 11.8283212 (1104)	total: 35.1s	remaining: 1m 30s
    1115:	learn: 11.0152262	test: 11.8278334	best: 11.8278334 (1115)	total: 35.1s	remaining: 1m 30s
    1116:	learn: 11.0143932	test: 11.8271390	best: 11.8271390 (1116)	total: 35.2s	remaining: 1m 30s
    1117:	learn: 11.0108348	test: 11.8285351	best: 11.8271390 (1116)	total: 35.2s	remaining: 1m 30s
    1118:	learn: 11.0091254	test: 11.8270768	best: 11.8270768 (1118)	total: 35.2s	remaining: 1m 30s
    1119:	learn: 11.0066520	test: 11.8274353	best: 11.8270768 (1118)	total: 35.3s	remaining: 1m 30s
    1120:	learn: 11.0058675	test: 11.8284256	best: 11.8270768 (1118)	total: 35.3s	remaining: 1m 30s
    1121:	learn: 11.0045493	test: 11.8286309	best: 11.8270768 (1118)	total: 35.3s	remaining: 1m 30s
    1122:	learn: 11.0033467	test: 11.8286753	best: 11.8270768 (1118)	total: 35.4s	remaining: 1m 30s
    1123:	learn: 11.0026306	test: 11.8283240	best: 11.8270768 (1118)	total: 35.4s	remaining: 1m 30s
    1124:	learn: 10.9997178	test: 11.8314272	best: 11.8270768 (1118)	total: 35.4s	remaining: 1m 30s
    1125:	learn: 10.9990218	test: 11.8319589	best: 11.8270768 (1118)	total: 35.4s	remaining: 1m 30s
    1126:	learn: 10.9984966	test: 11.8312502	best: 11.8270768 (1118)	total: 35.5s	remaining: 1m 30s
    1127:	learn: 10.9969558	test: 11.8306860	best: 11.8270768 (1118)	total: 35.5s	remaining: 1m 30s
    1128:	learn: 10.9959386	test: 11.8304773	best: 11.8270768 (1118)	total: 35.5s	remaining: 1m 30s
    1129:	learn: 10.9950821	test: 11.8302194	best: 11.8270768 (1118)	total: 35.6s	remaining: 1m 30s
    1130:	learn: 10.9938370	test: 11.8298678	best: 11.8270768 (1118)	total: 35.6s	remaining: 1m 30s
    1131:	learn: 10.9935148	test: 11.8303005	best: 11.8270768 (1118)	total: 35.7s	remaining: 1m 30s
    1132:	learn: 10.9920444	test: 11.8300520	best: 11.8270768 (1118)	total: 35.7s	remaining: 1m 30s
    1133:	learn: 10.9896926	test: 11.8268188	best: 11.8268188 (1133)	total: 35.7s	remaining: 1m 30s
    1134:	learn: 10.9883794	test: 11.8257465	best: 11.8257465 (1134)	total: 35.8s	remaining: 1m 30s
    1135:	learn: 10.9875774	test: 11.8260567	best: 11.8257465 (1134)	total: 35.8s	remaining: 1m 30s
    1136:	learn: 10.9859226	test: 11.8245468	best: 11.8245468 (1136)	total: 35.8s	remaining: 1m 30s
    1137:	learn: 10.9834906	test: 11.8235861	best: 11.8235861 (1137)	total: 35.9s	remaining: 1m 30s
    1138:	learn: 10.9825026	test: 11.8227321	best: 11.8227321 (1138)	total: 35.9s	remaining: 1m 30s
    1139:	learn: 10.9761769	test: 11.8216356	best: 11.8216356 (1139)	total: 35.9s	remaining: 1m 30s
    1140:	learn: 10.9746373	test: 11.8204464	best: 11.8204464 (1140)	total: 36s	remaining: 1m 30s
    1141:	learn: 10.9726529	test: 11.8191260	best: 11.8191260 (1141)	total: 36s	remaining: 1m 30s
    1142:	learn: 10.9700610	test: 11.8198079	best: 11.8191260 (1141)	total: 36s	remaining: 1m 30s
    1143:	learn: 10.9679979	test: 11.8180487	best: 11.8180487 (1143)	total: 36.1s	remaining: 1m 30s
    1144:	learn: 10.9670044	test: 11.8168285	best: 11.8168285 (1144)	total: 36.1s	remaining: 1m 30s
    1145:	learn: 10.9639931	test: 11.8126346	best: 11.8126346 (1145)	total: 36.1s	remaining: 1m 29s
    1146:	learn: 10.9626067	test: 11.8109602	best: 11.8109602 (1146)	total: 36.2s	remaining: 1m 29s
    1147:	learn: 10.9613308	test: 11.8095967	best: 11.8095967 (1147)	total: 36.2s	remaining: 1m 29s
    1148:	learn: 10.9597120	test: 11.8074154	best: 11.8074154 (1148)	total: 36.2s	remaining: 1m 29s
    1149:	learn: 10.9597592	test: 11.8077164	best: 11.8074154 (1148)	total: 36.3s	remaining: 1m 29s
    1150:	learn: 10.9560879	test: 11.8118044	best: 11.8074154 (1148)	total: 36.3s	remaining: 1m 29s
    1151:	learn: 10.9536636	test: 11.8131819	best: 11.8074154 (1148)	total: 36.3s	remaining: 1m 29s
    1152:	learn: 10.9486081	test: 11.8102497	best: 11.8074154 (1148)	total: 36.4s	remaining: 1m 29s
    1153:	learn: 10.9467106	test: 11.8107822	best: 11.8074154 (1148)	total: 36.4s	remaining: 1m 29s
    1154:	learn: 10.9453947	test: 11.8104014	best: 11.8074154 (1148)	total: 36.4s	remaining: 1m 29s
    1155:	learn: 10.9437730	test: 11.8115045	best: 11.8074154 (1148)	total: 36.5s	remaining: 1m 29s
    1156:	learn: 10.9410041	test: 11.8116632	best: 11.8074154 (1148)	total: 36.5s	remaining: 1m 29s
    1157:	learn: 10.9380398	test: 11.8102864	best: 11.8074154 (1148)	total: 36.6s	remaining: 1m 29s
    1158:	learn: 10.9344849	test: 11.8081787	best: 11.8074154 (1148)	total: 36.6s	remaining: 1m 29s
    1159:	learn: 10.9311195	test: 11.8081690	best: 11.8074154 (1148)	total: 36.7s	remaining: 1m 29s
    1160:	learn: 10.9299368	test: 11.8077213	best: 11.8074154 (1148)	total: 36.7s	remaining: 1m 29s
    1161:	learn: 10.9262756	test: 11.8045535	best: 11.8045535 (1161)	total: 36.8s	remaining: 1m 29s
    1162:	learn: 10.9244803	test: 11.8034617	best: 11.8034617 (1162)	total: 36.8s	remaining: 1m 29s
    1163:	learn: 10.9232493	test: 11.8029381	best: 11.8029381 (1163)	total: 36.8s	remaining: 1m 29s
    1164:	learn: 10.9208170	test: 11.8043954	best: 11.8029381 (1163)	total: 36.9s	remaining: 1m 29s
    1165:	learn: 10.9201387	test: 11.8042028	best: 11.8029381 (1163)	total: 36.9s	remaining: 1m 29s
    1166:	learn: 10.9189895	test: 11.8046531	best: 11.8029381 (1163)	total: 36.9s	remaining: 1m 29s
    1167:	learn: 10.9174798	test: 11.8053466	best: 11.8029381 (1163)	total: 37s	remaining: 1m 29s
    1168:	learn: 10.9157921	test: 11.8032434	best: 11.8029381 (1163)	total: 37s	remaining: 1m 29s
    1169:	learn: 10.9141077	test: 11.8013253	best: 11.8013253 (1169)	total: 37s	remaining: 1m 29s
    1170:	learn: 10.9125812	test: 11.8013752	best: 11.8013253 (1169)	total: 37.1s	remaining: 1m 29s
    1171:	learn: 10.9118464	test: 11.8010704	best: 11.8010704 (1171)	total: 37.1s	remaining: 1m 29s
    1172:	learn: 10.9099886	test: 11.7997171	best: 11.7997171 (1172)	total: 37.2s	remaining: 1m 29s
    1173:	learn: 10.9082145	test: 11.7990101	best: 11.7990101 (1173)	total: 37.2s	remaining: 1m 29s
    1174:	learn: 10.9053736	test: 11.7998134	best: 11.7990101 (1173)	total: 37.3s	remaining: 1m 29s
    1175:	learn: 10.9041096	test: 11.8004425	best: 11.7990101 (1173)	total: 37.3s	remaining: 1m 29s
    1176:	learn: 10.9022777	test: 11.8003686	best: 11.7990101 (1173)	total: 37.3s	remaining: 1m 29s
    1177:	learn: 10.9020019	test: 11.8007518	best: 11.7990101 (1173)	total: 37.4s	remaining: 1m 29s
    1178:	learn: 10.8993431	test: 11.8008003	best: 11.7990101 (1173)	total: 37.4s	remaining: 1m 29s
    1179:	learn: 10.8973900	test: 11.7994248	best: 11.7990101 (1173)	total: 37.4s	remaining: 1m 29s
    1180:	learn: 10.8923569	test: 11.7971148	best: 11.7971148 (1180)	total: 37.4s	remaining: 1m 29s
    1181:	learn: 10.8907817	test: 11.7980975	best: 11.7971148 (1180)	total: 37.5s	remaining: 1m 29s
    1182:	learn: 10.8879631	test: 11.7958561	best: 11.7958561 (1182)	total: 37.5s	remaining: 1m 29s
    1183:	learn: 10.8874290	test: 11.7964154	best: 11.7958561 (1182)	total: 37.5s	remaining: 1m 29s
    1184:	learn: 10.8852751	test: 11.7961731	best: 11.7958561 (1182)	total: 37.6s	remaining: 1m 29s
    1185:	learn: 10.8831939	test: 11.7963078	best: 11.7958561 (1182)	total: 37.6s	remaining: 1m 29s
    1186:	learn: 10.8817228	test: 11.7950524	best: 11.7950524 (1186)	total: 37.6s	remaining: 1m 29s
    1187:	learn: 10.8801620	test: 11.7952891	best: 11.7950524 (1186)	total: 37.7s	remaining: 1m 29s
    1188:	learn: 10.8795434	test: 11.7962295	best: 11.7950524 (1186)	total: 37.7s	remaining: 1m 29s
    1189:	learn: 10.8781946	test: 11.7966832	best: 11.7950524 (1186)	total: 37.7s	remaining: 1m 29s
    1190:	learn: 10.8727692	test: 11.7961512	best: 11.7950524 (1186)	total: 37.7s	remaining: 1m 29s
    1191:	learn: 10.8718867	test: 11.7960700	best: 11.7950524 (1186)	total: 37.8s	remaining: 1m 28s
    1192:	learn: 10.8689146	test: 11.7951044	best: 11.7950524 (1186)	total: 37.8s	remaining: 1m 28s
    1193:	learn: 10.8642890	test: 11.7955623	best: 11.7950524 (1186)	total: 37.9s	remaining: 1m 28s
    1194:	learn: 10.8639622	test: 11.7957521	best: 11.7950524 (1186)	total: 37.9s	remaining: 1m 28s
    1195:	learn: 10.8622790	test: 11.7947617	best: 11.7947617 (1195)	total: 37.9s	remaining: 1m 28s
    1196:	learn: 10.8606109	test: 11.7945942	best: 11.7945942 (1196)	total: 38s	remaining: 1m 28s
    1197:	learn: 10.8599095	test: 11.7946566	best: 11.7945942 (1196)	total: 38s	remaining: 1m 28s
    1198:	learn: 10.8561666	test: 11.7943403	best: 11.7943403 (1198)	total: 38s	remaining: 1m 28s
    1199:	learn: 10.8547890	test: 11.7945086	best: 11.7943403 (1198)	total: 38.1s	remaining: 1m 28s
    1200:	learn: 10.8525867	test: 11.7936983	best: 11.7936983 (1200)	total: 38.1s	remaining: 1m 28s
    1201:	learn: 10.8515205	test: 11.7945550	best: 11.7936983 (1200)	total: 38.1s	remaining: 1m 28s
    1202:	learn: 10.8502806	test: 11.7937793	best: 11.7936983 (1200)	total: 38.2s	remaining: 1m 28s
    1203:	learn: 10.8467037	test: 11.7906714	best: 11.7906714 (1203)	total: 38.2s	remaining: 1m 28s
    1204:	learn: 10.8448240	test: 11.7887596	best: 11.7887596 (1204)	total: 38.2s	remaining: 1m 28s
    1205:	learn: 10.8437627	test: 11.7886594	best: 11.7886594 (1205)	total: 38.3s	remaining: 1m 28s
    1206:	learn: 10.8426915	test: 11.7881244	best: 11.7881244 (1206)	total: 38.3s	remaining: 1m 28s
    1207:	learn: 10.8411648	test: 11.7879287	best: 11.7879287 (1207)	total: 38.3s	remaining: 1m 28s
    1208:	learn: 10.8375636	test: 11.7885205	best: 11.7879287 (1207)	total: 38.4s	remaining: 1m 28s
    1209:	learn: 10.8358459	test: 11.7895969	best: 11.7879287 (1207)	total: 38.4s	remaining: 1m 28s
    1210:	learn: 10.8339110	test: 11.7896269	best: 11.7879287 (1207)	total: 38.4s	remaining: 1m 28s
    1211:	learn: 10.8319380	test: 11.7895383	best: 11.7879287 (1207)	total: 38.5s	remaining: 1m 28s
    1212:	learn: 10.8282309	test: 11.7854945	best: 11.7854945 (1212)	total: 38.5s	remaining: 1m 28s
    1213:	learn: 10.8273653	test: 11.7852182	best: 11.7852182 (1213)	total: 38.5s	remaining: 1m 28s
    1214:	learn: 10.8219058	test: 11.7831678	best: 11.7831678 (1214)	total: 38.6s	remaining: 1m 28s
    1215:	learn: 10.8195048	test: 11.7834092	best: 11.7831678 (1214)	total: 38.6s	remaining: 1m 28s
    1216:	learn: 10.8184929	test: 11.7829825	best: 11.7829825 (1216)	total: 38.7s	remaining: 1m 28s
    1217:	learn: 10.8165531	test: 11.7815341	best: 11.7815341 (1217)	total: 38.7s	remaining: 1m 28s
    1218:	learn: 10.8125314	test: 11.7772190	best: 11.7772190 (1218)	total: 38.8s	remaining: 1m 28s
    1219:	learn: 10.8105382	test: 11.7764405	best: 11.7764405 (1219)	total: 38.8s	remaining: 1m 28s
    1220:	learn: 10.8064888	test: 11.7757028	best: 11.7757028 (1220)	total: 38.9s	remaining: 1m 28s
    1221:	learn: 10.8055407	test: 11.7746496	best: 11.7746496 (1221)	total: 38.9s	remaining: 1m 28s
    1222:	learn: 10.8041141	test: 11.7741506	best: 11.7741506 (1222)	total: 39s	remaining: 1m 28s
    1223:	learn: 10.8014618	test: 11.7745608	best: 11.7741506 (1222)	total: 39s	remaining: 1m 28s
    1224:	learn: 10.8009733	test: 11.7749186	best: 11.7741506 (1222)	total: 39s	remaining: 1m 28s
    1225:	learn: 10.7985937	test: 11.7742037	best: 11.7741506 (1222)	total: 39.1s	remaining: 1m 28s
    1226:	learn: 10.7975386	test: 11.7758297	best: 11.7741506 (1222)	total: 39.1s	remaining: 1m 28s
    1227:	learn: 10.7967828	test: 11.7762149	best: 11.7741506 (1222)	total: 39.1s	remaining: 1m 28s
    1228:	learn: 10.7949808	test: 11.7763673	best: 11.7741506 (1222)	total: 39.2s	remaining: 1m 28s
    1229:	learn: 10.7939151	test: 11.7759894	best: 11.7741506 (1222)	total: 39.2s	remaining: 1m 28s
    1230:	learn: 10.7931608	test: 11.7748742	best: 11.7741506 (1222)	total: 39.2s	remaining: 1m 28s
    1231:	learn: 10.7884048	test: 11.7723642	best: 11.7723642 (1231)	total: 39.2s	remaining: 1m 28s
    1232:	learn: 10.7873884	test: 11.7734880	best: 11.7723642 (1231)	total: 39.3s	remaining: 1m 28s
    1233:	learn: 10.7866075	test: 11.7732322	best: 11.7723642 (1231)	total: 39.3s	remaining: 1m 28s
    1234:	learn: 10.7840500	test: 11.7726353	best: 11.7723642 (1231)	total: 39.3s	remaining: 1m 28s
    1235:	learn: 10.7836912	test: 11.7727859	best: 11.7723642 (1231)	total: 39.4s	remaining: 1m 28s
    1236:	learn: 10.7816837	test: 11.7733491	best: 11.7723642 (1231)	total: 39.4s	remaining: 1m 28s
    1237:	learn: 10.7775488	test: 11.7697406	best: 11.7697406 (1237)	total: 39.4s	remaining: 1m 27s
    1238:	learn: 10.7771440	test: 11.7704866	best: 11.7697406 (1237)	total: 39.5s	remaining: 1m 27s
    1239:	learn: 10.7763219	test: 11.7690721	best: 11.7690721 (1239)	total: 39.5s	remaining: 1m 27s
    1240:	learn: 10.7743638	test: 11.7692381	best: 11.7690721 (1239)	total: 39.5s	remaining: 1m 27s
    1241:	learn: 10.7734628	test: 11.7689976	best: 11.7689976 (1241)	total: 39.6s	remaining: 1m 27s
    1242:	learn: 10.7696784	test: 11.7664121	best: 11.7664121 (1242)	total: 39.6s	remaining: 1m 27s
    1243:	learn: 10.7676231	test: 11.7660173	best: 11.7660173 (1243)	total: 39.6s	remaining: 1m 27s
    1244:	learn: 10.7652458	test: 11.7656314	best: 11.7656314 (1244)	total: 39.6s	remaining: 1m 27s
    1245:	learn: 10.7636209	test: 11.7663738	best: 11.7656314 (1244)	total: 39.7s	remaining: 1m 27s
    1246:	learn: 10.7620522	test: 11.7655070	best: 11.7655070 (1246)	total: 39.7s	remaining: 1m 27s
    1247:	learn: 10.7609178	test: 11.7646309	best: 11.7646309 (1247)	total: 39.7s	remaining: 1m 27s
    1248:	learn: 10.7604517	test: 11.7648238	best: 11.7646309 (1247)	total: 39.8s	remaining: 1m 27s
    1249:	learn: 10.7585516	test: 11.7656596	best: 11.7646309 (1247)	total: 39.8s	remaining: 1m 27s
    1250:	learn: 10.7561801	test: 11.7623017	best: 11.7623017 (1250)	total: 39.8s	remaining: 1m 27s
    1251:	learn: 10.7535901	test: 11.7626919	best: 11.7623017 (1250)	total: 39.8s	remaining: 1m 27s
    1252:	learn: 10.7529879	test: 11.7619728	best: 11.7619728 (1252)	total: 39.9s	remaining: 1m 27s
    1253:	learn: 10.7518611	test: 11.7617792	best: 11.7617792 (1253)	total: 39.9s	remaining: 1m 27s
    1254:	learn: 10.7516249	test: 11.7621935	best: 11.7617792 (1253)	total: 39.9s	remaining: 1m 27s
    1255:	learn: 10.7508858	test: 11.7620309	best: 11.7617792 (1253)	total: 40s	remaining: 1m 27s
    1256:	learn: 10.7482418	test: 11.7624281	best: 11.7617792 (1253)	total: 40s	remaining: 1m 27s
    1257:	learn: 10.7462765	test: 11.7608694	best: 11.7608694 (1257)	total: 40s	remaining: 1m 27s
    1258:	learn: 10.7411314	test: 11.7587853	best: 11.7587853 (1258)	total: 40.1s	remaining: 1m 27s
    1259:	learn: 10.7400124	test: 11.7590221	best: 11.7587853 (1258)	total: 40.1s	remaining: 1m 27s
    1260:	learn: 10.7391046	test: 11.7604324	best: 11.7587853 (1258)	total: 40.1s	remaining: 1m 27s
    1261:	learn: 10.7384485	test: 11.7601997	best: 11.7587853 (1258)	total: 40.1s	remaining: 1m 27s
    1262:	learn: 10.7381921	test: 11.7603746	best: 11.7587853 (1258)	total: 40.2s	remaining: 1m 27s
    1263:	learn: 10.7367333	test: 11.7585553	best: 11.7585553 (1263)	total: 40.2s	remaining: 1m 27s
    1264:	learn: 10.7350067	test: 11.7591371	best: 11.7585553 (1263)	total: 40.2s	remaining: 1m 26s
    1265:	learn: 10.7330633	test: 11.7575644	best: 11.7575644 (1265)	total: 40.3s	remaining: 1m 26s
    1266:	learn: 10.7310842	test: 11.7565639	best: 11.7565639 (1266)	total: 40.3s	remaining: 1m 26s
    1267:	learn: 10.7307809	test: 11.7567760	best: 11.7565639 (1266)	total: 40.3s	remaining: 1m 26s
    1268:	learn: 10.7289922	test: 11.7585963	best: 11.7565639 (1266)	total: 40.4s	remaining: 1m 26s
    1269:	learn: 10.7277595	test: 11.7582604	best: 11.7565639 (1266)	total: 40.4s	remaining: 1m 26s
    1270:	learn: 10.7272364	test: 11.7577521	best: 11.7565639 (1266)	total: 40.4s	remaining: 1m 26s
    1271:	learn: 10.7253529	test: 11.7583831	best: 11.7565639 (1266)	total: 40.5s	remaining: 1m 26s
    1272:	learn: 10.7244714	test: 11.7607688	best: 11.7565639 (1266)	total: 40.5s	remaining: 1m 26s
    1273:	learn: 10.7234463	test: 11.7586744	best: 11.7565639 (1266)	total: 40.6s	remaining: 1m 26s
    1274:	learn: 10.7221032	test: 11.7587276	best: 11.7565639 (1266)	total: 40.6s	remaining: 1m 26s
    1275:	learn: 10.7202594	test: 11.7595598	best: 11.7565639 (1266)	total: 40.6s	remaining: 1m 26s
    1276:	learn: 10.7178903	test: 11.7570489	best: 11.7565639 (1266)	total: 40.6s	remaining: 1m 26s
    1277:	learn: 10.7170413	test: 11.7582130	best: 11.7565639 (1266)	total: 40.7s	remaining: 1m 26s
    1278:	learn: 10.7168053	test: 11.7583330	best: 11.7565639 (1266)	total: 40.7s	remaining: 1m 26s
    1279:	learn: 10.7147624	test: 11.7587936	best: 11.7565639 (1266)	total: 40.7s	remaining: 1m 26s
    1280:	learn: 10.7136384	test: 11.7573971	best: 11.7565639 (1266)	total: 40.7s	remaining: 1m 26s
    1281:	learn: 10.7127463	test: 11.7569707	best: 11.7565639 (1266)	total: 40.8s	remaining: 1m 26s
    1282:	learn: 10.7115914	test: 11.7574091	best: 11.7565639 (1266)	total: 40.8s	remaining: 1m 26s
    1283:	learn: 10.7105926	test: 11.7588596	best: 11.7565639 (1266)	total: 40.8s	remaining: 1m 26s
    1284:	learn: 10.7095156	test: 11.7589356	best: 11.7565639 (1266)	total: 40.9s	remaining: 1m 26s
    1285:	learn: 10.7077634	test: 11.7584938	best: 11.7565639 (1266)	total: 40.9s	remaining: 1m 26s
    1286:	learn: 10.7058726	test: 11.7578897	best: 11.7565639 (1266)	total: 40.9s	remaining: 1m 26s
    1287:	learn: 10.7052274	test: 11.7589505	best: 11.7565639 (1266)	total: 40.9s	remaining: 1m 26s
    1288:	learn: 10.7030117	test: 11.7590758	best: 11.7565639 (1266)	total: 41s	remaining: 1m 26s
    1289:	learn: 10.7000277	test: 11.7581727	best: 11.7565639 (1266)	total: 41s	remaining: 1m 26s
    1290:	learn: 10.6963851	test: 11.7568971	best: 11.7565639 (1266)	total: 41s	remaining: 1m 26s
    1291:	learn: 10.6953600	test: 11.7561310	best: 11.7561310 (1291)	total: 41s	remaining: 1m 26s
    1292:	learn: 10.6938954	test: 11.7558716	best: 11.7558716 (1292)	total: 41.1s	remaining: 1m 25s
    1293:	learn: 10.6934521	test: 11.7541350	best: 11.7541350 (1293)	total: 41.1s	remaining: 1m 25s
    1294:	learn: 10.6927377	test: 11.7534442	best: 11.7534442 (1294)	total: 41.1s	remaining: 1m 25s
    1295:	learn: 10.6873176	test: 11.7506062	best: 11.7506062 (1295)	total: 41.1s	remaining: 1m 25s
    1296:	learn: 10.6835895	test: 11.7481679	best: 11.7481679 (1296)	total: 41.2s	remaining: 1m 25s
    1297:	learn: 10.6795915	test: 11.7469450	best: 11.7469450 (1297)	total: 41.2s	remaining: 1m 25s
    1298:	learn: 10.6750305	test: 11.7437731	best: 11.7437731 (1298)	total: 41.3s	remaining: 1m 25s
    1299:	learn: 10.6730742	test: 11.7452330	best: 11.7437731 (1298)	total: 41.3s	remaining: 1m 25s
    1300:	learn: 10.6719385	test: 11.7456004	best: 11.7437731 (1298)	total: 41.3s	remaining: 1m 25s
    1301:	learn: 10.6702558	test: 11.7457158	best: 11.7437731 (1298)	total: 41.4s	remaining: 1m 25s
    1302:	learn: 10.6695468	test: 11.7461757	best: 11.7437731 (1298)	total: 41.4s	remaining: 1m 25s
    1303:	learn: 10.6684539	test: 11.7470682	best: 11.7437731 (1298)	total: 41.4s	remaining: 1m 25s
    1304:	learn: 10.6655515	test: 11.7482419	best: 11.7437731 (1298)	total: 41.5s	remaining: 1m 25s
    1305:	learn: 10.6645780	test: 11.7476458	best: 11.7437731 (1298)	total: 41.5s	remaining: 1m 25s
    1306:	learn: 10.6633057	test: 11.7473877	best: 11.7437731 (1298)	total: 41.6s	remaining: 1m 25s
    1307:	learn: 10.6620987	test: 11.7476006	best: 11.7437731 (1298)	total: 41.6s	remaining: 1m 25s
    1308:	learn: 10.6614829	test: 11.7462044	best: 11.7437731 (1298)	total: 41.6s	remaining: 1m 25s
    1309:	learn: 10.6604756	test: 11.7475130	best: 11.7437731 (1298)	total: 41.7s	remaining: 1m 25s
    1310:	learn: 10.6588131	test: 11.7468364	best: 11.7437731 (1298)	total: 41.7s	remaining: 1m 25s
    1311:	learn: 10.6577714	test: 11.7468839	best: 11.7437731 (1298)	total: 41.8s	remaining: 1m 25s
    1312:	learn: 10.6569031	test: 11.7469410	best: 11.7437731 (1298)	total: 41.8s	remaining: 1m 25s
    1313:	learn: 10.6566450	test: 11.7472568	best: 11.7437731 (1298)	total: 41.9s	remaining: 1m 25s
    1314:	learn: 10.6544413	test: 11.7448993	best: 11.7437731 (1298)	total: 41.9s	remaining: 1m 25s
    1315:	learn: 10.6536650	test: 11.7440425	best: 11.7437731 (1298)	total: 41.9s	remaining: 1m 25s
    1316:	learn: 10.6508649	test: 11.7433986	best: 11.7433986 (1316)	total: 42s	remaining: 1m 25s
    1317:	learn: 10.6500464	test: 11.7437977	best: 11.7433986 (1316)	total: 42s	remaining: 1m 25s
    1318:	learn: 10.6496862	test: 11.7448236	best: 11.7433986 (1316)	total: 42s	remaining: 1m 25s
    1319:	learn: 10.6483529	test: 11.7448995	best: 11.7433986 (1316)	total: 42s	remaining: 1m 25s
    1320:	learn: 10.6478396	test: 11.7456764	best: 11.7433986 (1316)	total: 42.1s	remaining: 1m 25s
    1321:	learn: 10.6470048	test: 11.7470319	best: 11.7433986 (1316)	total: 42.1s	remaining: 1m 25s
    1322:	learn: 10.6461805	test: 11.7464995	best: 11.7433986 (1316)	total: 42.1s	remaining: 1m 25s
    1323:	learn: 10.6446803	test: 11.7481615	best: 11.7433986 (1316)	total: 42.2s	remaining: 1m 25s
    1324:	learn: 10.6407288	test: 11.7501903	best: 11.7433986 (1316)	total: 42.2s	remaining: 1m 25s
    1325:	learn: 10.6394961	test: 11.7514437	best: 11.7433986 (1316)	total: 42.2s	remaining: 1m 25s
    1326:	learn: 10.6386320	test: 11.7514158	best: 11.7433986 (1316)	total: 42.3s	remaining: 1m 25s
    1327:	learn: 10.6373169	test: 11.7510439	best: 11.7433986 (1316)	total: 42.3s	remaining: 1m 25s
    1328:	learn: 10.6341329	test: 11.7500125	best: 11.7433986 (1316)	total: 42.3s	remaining: 1m 25s
    1329:	learn: 10.6335504	test: 11.7501691	best: 11.7433986 (1316)	total: 42.3s	remaining: 1m 25s
    1330:	learn: 10.6311695	test: 11.7495062	best: 11.7433986 (1316)	total: 42.4s	remaining: 1m 24s
    1331:	learn: 10.6293384	test: 11.7503171	best: 11.7433986 (1316)	total: 42.4s	remaining: 1m 24s
    1332:	learn: 10.6265952	test: 11.7520364	best: 11.7433986 (1316)	total: 42.4s	remaining: 1m 24s
    1333:	learn: 10.6221402	test: 11.7512687	best: 11.7433986 (1316)	total: 42.5s	remaining: 1m 24s
    1334:	learn: 10.6209015	test: 11.7515869	best: 11.7433986 (1316)	total: 42.5s	remaining: 1m 24s
    1335:	learn: 10.6197178	test: 11.7514164	best: 11.7433986 (1316)	total: 42.5s	remaining: 1m 24s
    1336:	learn: 10.6176612	test: 11.7524940	best: 11.7433986 (1316)	total: 42.5s	remaining: 1m 24s
    1337:	learn: 10.6152132	test: 11.7517774	best: 11.7433986 (1316)	total: 42.6s	remaining: 1m 24s
    1338:	learn: 10.6144794	test: 11.7508748	best: 11.7433986 (1316)	total: 42.6s	remaining: 1m 24s
    1339:	learn: 10.6123289	test: 11.7480934	best: 11.7433986 (1316)	total: 42.6s	remaining: 1m 24s
    1340:	learn: 10.6095815	test: 11.7458630	best: 11.7433986 (1316)	total: 42.7s	remaining: 1m 24s
    1341:	learn: 10.6082686	test: 11.7468977	best: 11.7433986 (1316)	total: 42.7s	remaining: 1m 24s
    1342:	learn: 10.6071553	test: 11.7456580	best: 11.7433986 (1316)	total: 42.7s	remaining: 1m 24s
    1343:	learn: 10.6040825	test: 11.7462128	best: 11.7433986 (1316)	total: 42.7s	remaining: 1m 24s
    1344:	learn: 10.6017489	test: 11.7447667	best: 11.7433986 (1316)	total: 42.8s	remaining: 1m 24s
    1345:	learn: 10.5992039	test: 11.7467310	best: 11.7433986 (1316)	total: 42.8s	remaining: 1m 24s
    1346:	learn: 10.5989808	test: 11.7476537	best: 11.7433986 (1316)	total: 42.8s	remaining: 1m 24s
    1347:	learn: 10.5975481	test: 11.7467617	best: 11.7433986 (1316)	total: 42.9s	remaining: 1m 24s
    1348:	learn: 10.5957508	test: 11.7465170	best: 11.7433986 (1316)	total: 42.9s	remaining: 1m 24s
    1349:	learn: 10.5932305	test: 11.7467075	best: 11.7433986 (1316)	total: 42.9s	remaining: 1m 24s
    1350:	learn: 10.5919609	test: 11.7464376	best: 11.7433986 (1316)	total: 43s	remaining: 1m 24s
    1351:	learn: 10.5914191	test: 11.7470018	best: 11.7433986 (1316)	total: 43s	remaining: 1m 24s
    1352:	learn: 10.5902386	test: 11.7445775	best: 11.7433986 (1316)	total: 43s	remaining: 1m 24s
    1353:	learn: 10.5900267	test: 11.7440877	best: 11.7433986 (1316)	total: 43.1s	remaining: 1m 24s
    1354:	learn: 10.5882057	test: 11.7447189	best: 11.7433986 (1316)	total: 43.1s	remaining: 1m 24s
    1355:	learn: 10.5875258	test: 11.7439803	best: 11.7433986 (1316)	total: 43.1s	remaining: 1m 24s
    1356:	learn: 10.5858584	test: 11.7442505	best: 11.7433986 (1316)	total: 43.1s	remaining: 1m 24s
    1357:	learn: 10.5839567	test: 11.7431466	best: 11.7431466 (1357)	total: 43.2s	remaining: 1m 23s
    1358:	learn: 10.5825205	test: 11.7435867	best: 11.7431466 (1357)	total: 43.2s	remaining: 1m 23s
    1359:	learn: 10.5813747	test: 11.7440920	best: 11.7431466 (1357)	total: 43.2s	remaining: 1m 23s
    1360:	learn: 10.5809391	test: 11.7435712	best: 11.7431466 (1357)	total: 43.2s	remaining: 1m 23s
    1361:	learn: 10.5796942	test: 11.7425805	best: 11.7425805 (1361)	total: 43.3s	remaining: 1m 23s
    1362:	learn: 10.5789694	test: 11.7421286	best: 11.7421286 (1362)	total: 43.3s	remaining: 1m 23s
    1363:	learn: 10.5747958	test: 11.7387172	best: 11.7387172 (1363)	total: 43.3s	remaining: 1m 23s
    1364:	learn: 10.5727462	test: 11.7369347	best: 11.7369347 (1364)	total: 43.4s	remaining: 1m 23s
    1365:	learn: 10.5710057	test: 11.7362873	best: 11.7362873 (1365)	total: 43.4s	remaining: 1m 23s
    1366:	learn: 10.5695389	test: 11.7376812	best: 11.7362873 (1365)	total: 43.4s	remaining: 1m 23s
    1367:	learn: 10.5649414	test: 11.7357383	best: 11.7357383 (1367)	total: 43.5s	remaining: 1m 23s
    1368:	learn: 10.5641240	test: 11.7354091	best: 11.7354091 (1368)	total: 43.5s	remaining: 1m 23s
    1369:	learn: 10.5632057	test: 11.7352651	best: 11.7352651 (1369)	total: 43.5s	remaining: 1m 23s
    1370:	learn: 10.5621899	test: 11.7352524	best: 11.7352524 (1370)	total: 43.5s	remaining: 1m 23s
    1371:	learn: 10.5603080	test: 11.7345085	best: 11.7345085 (1371)	total: 43.6s	remaining: 1m 23s
    1372:	learn: 10.5581678	test: 11.7349887	best: 11.7345085 (1371)	total: 43.6s	remaining: 1m 23s
    1373:	learn: 10.5574221	test: 11.7347462	best: 11.7345085 (1371)	total: 43.6s	remaining: 1m 23s
    1374:	learn: 10.5541414	test: 11.7319675	best: 11.7319675 (1374)	total: 43.7s	remaining: 1m 23s
    1375:	learn: 10.5513577	test: 11.7323238	best: 11.7319675 (1374)	total: 43.7s	remaining: 1m 23s
    1376:	learn: 10.5490540	test: 11.7317180	best: 11.7317180 (1376)	total: 43.7s	remaining: 1m 23s
    1377:	learn: 10.5482849	test: 11.7309559	best: 11.7309559 (1377)	total: 43.7s	remaining: 1m 23s
    1378:	learn: 10.5455737	test: 11.7301447	best: 11.7301447 (1378)	total: 43.8s	remaining: 1m 23s
    1379:	learn: 10.5445923	test: 11.7292640	best: 11.7292640 (1379)	total: 43.8s	remaining: 1m 23s
    1380:	learn: 10.5431627	test: 11.7305716	best: 11.7292640 (1379)	total: 43.8s	remaining: 1m 23s
    1381:	learn: 10.5420357	test: 11.7295767	best: 11.7292640 (1379)	total: 43.8s	remaining: 1m 23s
    1382:	learn: 10.5397183	test: 11.7289207	best: 11.7289207 (1382)	total: 43.9s	remaining: 1m 23s
    1383:	learn: 10.5376720	test: 11.7303988	best: 11.7289207 (1382)	total: 43.9s	remaining: 1m 22s
    1384:	learn: 10.5364365	test: 11.7302450	best: 11.7289207 (1382)	total: 43.9s	remaining: 1m 22s
    1385:	learn: 10.5343990	test: 11.7273106	best: 11.7273106 (1385)	total: 44s	remaining: 1m 22s
    1386:	learn: 10.5309239	test: 11.7263673	best: 11.7263673 (1386)	total: 44s	remaining: 1m 22s
    1387:	learn: 10.5297214	test: 11.7261301	best: 11.7261301 (1387)	total: 44s	remaining: 1m 22s
    1388:	learn: 10.5289471	test: 11.7264708	best: 11.7261301 (1387)	total: 44s	remaining: 1m 22s
    1389:	learn: 10.5266702	test: 11.7247172	best: 11.7247172 (1389)	total: 44.1s	remaining: 1m 22s
    1390:	learn: 10.5248979	test: 11.7241776	best: 11.7241776 (1390)	total: 44.1s	remaining: 1m 22s
    1391:	learn: 10.5232397	test: 11.7220636	best: 11.7220636 (1391)	total: 44.1s	remaining: 1m 22s
    1392:	learn: 10.5225767	test: 11.7227655	best: 11.7220636 (1391)	total: 44.1s	remaining: 1m 22s
    1393:	learn: 10.5209543	test: 11.7226293	best: 11.7220636 (1391)	total: 44.2s	remaining: 1m 22s
    1394:	learn: 10.5197805	test: 11.7224331	best: 11.7220636 (1391)	total: 44.2s	remaining: 1m 22s
    1395:	learn: 10.5192343	test: 11.7232238	best: 11.7220636 (1391)	total: 44.2s	remaining: 1m 22s
    1396:	learn: 10.5168672	test: 11.7200860	best: 11.7200860 (1396)	total: 44.3s	remaining: 1m 22s
    1397:	learn: 10.5161716	test: 11.7204120	best: 11.7200860 (1396)	total: 44.3s	remaining: 1m 22s
    1398:	learn: 10.5138042	test: 11.7187587	best: 11.7187587 (1398)	total: 44.3s	remaining: 1m 22s
    1399:	learn: 10.5127509	test: 11.7170889	best: 11.7170889 (1399)	total: 44.3s	remaining: 1m 22s
    1400:	learn: 10.5110883	test: 11.7186986	best: 11.7170889 (1399)	total: 44.4s	remaining: 1m 22s
    1401:	learn: 10.5087770	test: 11.7197582	best: 11.7170889 (1399)	total: 44.4s	remaining: 1m 22s
    1402:	learn: 10.5085625	test: 11.7198364	best: 11.7170889 (1399)	total: 44.4s	remaining: 1m 22s
    1403:	learn: 10.5055560	test: 11.7193739	best: 11.7170889 (1399)	total: 44.4s	remaining: 1m 22s
    1404:	learn: 10.5044836	test: 11.7184839	best: 11.7170889 (1399)	total: 44.5s	remaining: 1m 22s
    1405:	learn: 10.5024971	test: 11.7185653	best: 11.7170889 (1399)	total: 44.5s	remaining: 1m 22s
    1406:	learn: 10.5016871	test: 11.7188238	best: 11.7170889 (1399)	total: 44.5s	remaining: 1m 22s
    1407:	learn: 10.5012984	test: 11.7197124	best: 11.7170889 (1399)	total: 44.6s	remaining: 1m 22s
    1408:	learn: 10.5010328	test: 11.7207246	best: 11.7170889 (1399)	total: 44.6s	remaining: 1m 21s
    1409:	learn: 10.4981852	test: 11.7214152	best: 11.7170889 (1399)	total: 44.6s	remaining: 1m 21s
    1410:	learn: 10.4962158	test: 11.7231112	best: 11.7170889 (1399)	total: 44.7s	remaining: 1m 21s
    1411:	learn: 10.4956158	test: 11.7231258	best: 11.7170889 (1399)	total: 44.7s	remaining: 1m 21s
    1412:	learn: 10.4940941	test: 11.7231526	best: 11.7170889 (1399)	total: 44.8s	remaining: 1m 21s
    1413:	learn: 10.4914711	test: 11.7241368	best: 11.7170889 (1399)	total: 44.8s	remaining: 1m 21s
    1414:	learn: 10.4906257	test: 11.7245608	best: 11.7170889 (1399)	total: 44.8s	remaining: 1m 21s
    1415:	learn: 10.4881133	test: 11.7223150	best: 11.7170889 (1399)	total: 44.9s	remaining: 1m 21s
    1416:	learn: 10.4861798	test: 11.7223624	best: 11.7170889 (1399)	total: 44.9s	remaining: 1m 21s
    1417:	learn: 10.4833078	test: 11.7235769	best: 11.7170889 (1399)	total: 44.9s	remaining: 1m 21s
    1418:	learn: 10.4797232	test: 11.7216652	best: 11.7170889 (1399)	total: 45s	remaining: 1m 21s
    1419:	learn: 10.4769746	test: 11.7236387	best: 11.7170889 (1399)	total: 45s	remaining: 1m 21s
    1420:	learn: 10.4753252	test: 11.7234957	best: 11.7170889 (1399)	total: 45s	remaining: 1m 21s
    1421:	learn: 10.4747448	test: 11.7232060	best: 11.7170889 (1399)	total: 45s	remaining: 1m 21s
    1422:	learn: 10.4726080	test: 11.7228129	best: 11.7170889 (1399)	total: 45.1s	remaining: 1m 21s
    1423:	learn: 10.4721681	test: 11.7230321	best: 11.7170889 (1399)	total: 45.1s	remaining: 1m 21s
    1424:	learn: 10.4710359	test: 11.7235037	best: 11.7170889 (1399)	total: 45.1s	remaining: 1m 21s
    1425:	learn: 10.4693790	test: 11.7224931	best: 11.7170889 (1399)	total: 45.1s	remaining: 1m 21s
    1426:	learn: 10.4662886	test: 11.7216976	best: 11.7170889 (1399)	total: 45.2s	remaining: 1m 21s
    1427:	learn: 10.4651934	test: 11.7207427	best: 11.7170889 (1399)	total: 45.2s	remaining: 1m 21s
    1428:	learn: 10.4645199	test: 11.7213914	best: 11.7170889 (1399)	total: 45.2s	remaining: 1m 21s
    1429:	learn: 10.4639714	test: 11.7225620	best: 11.7170889 (1399)	total: 45.3s	remaining: 1m 21s
    1430:	learn: 10.4615928	test: 11.7242236	best: 11.7170889 (1399)	total: 45.3s	remaining: 1m 21s
    1431:	learn: 10.4604473	test: 11.7244697	best: 11.7170889 (1399)	total: 45.3s	remaining: 1m 21s
    1432:	learn: 10.4598128	test: 11.7232553	best: 11.7170889 (1399)	total: 45.3s	remaining: 1m 21s
    1433:	learn: 10.4587668	test: 11.7225225	best: 11.7170889 (1399)	total: 45.4s	remaining: 1m 21s
    1434:	learn: 10.4579656	test: 11.7226859	best: 11.7170889 (1399)	total: 45.5s	remaining: 1m 21s
    1435:	learn: 10.4570118	test: 11.7223612	best: 11.7170889 (1399)	total: 45.5s	remaining: 1m 21s
    1436:	learn: 10.4548174	test: 11.7212793	best: 11.7170889 (1399)	total: 45.6s	remaining: 1m 21s
    1437:	learn: 10.4531051	test: 11.7195323	best: 11.7170889 (1399)	total: 45.6s	remaining: 1m 21s
    1438:	learn: 10.4516030	test: 11.7180659	best: 11.7170889 (1399)	total: 45.6s	remaining: 1m 21s
    1439:	learn: 10.4489341	test: 11.7183274	best: 11.7170889 (1399)	total: 45.6s	remaining: 1m 21s
    1440:	learn: 10.4482034	test: 11.7179552	best: 11.7170889 (1399)	total: 45.7s	remaining: 1m 21s
    1441:	learn: 10.4477780	test: 11.7179558	best: 11.7170889 (1399)	total: 45.7s	remaining: 1m 21s
    1442:	learn: 10.4465684	test: 11.7178556	best: 11.7170889 (1399)	total: 45.7s	remaining: 1m 21s
    1443:	learn: 10.4454736	test: 11.7175234	best: 11.7170889 (1399)	total: 45.7s	remaining: 1m 20s
    1444:	learn: 10.4436493	test: 11.7160290	best: 11.7160290 (1444)	total: 45.8s	remaining: 1m 20s
    1445:	learn: 10.4408297	test: 11.7148420	best: 11.7148420 (1445)	total: 45.8s	remaining: 1m 20s
    1446:	learn: 10.4385513	test: 11.7156482	best: 11.7148420 (1445)	total: 45.8s	remaining: 1m 20s
    1447:	learn: 10.4371881	test: 11.7161724	best: 11.7148420 (1445)	total: 45.9s	remaining: 1m 20s
    1448:	learn: 10.4352302	test: 11.7171308	best: 11.7148420 (1445)	total: 45.9s	remaining: 1m 20s
    1449:	learn: 10.4335072	test: 11.7165229	best: 11.7148420 (1445)	total: 45.9s	remaining: 1m 20s
    1450:	learn: 10.4329251	test: 11.7163262	best: 11.7148420 (1445)	total: 45.9s	remaining: 1m 20s
    1451:	learn: 10.4315697	test: 11.7174412	best: 11.7148420 (1445)	total: 46s	remaining: 1m 20s
    1452:	learn: 10.4309027	test: 11.7178645	best: 11.7148420 (1445)	total: 46s	remaining: 1m 20s
    1453:	learn: 10.4300743	test: 11.7187609	best: 11.7148420 (1445)	total: 46s	remaining: 1m 20s
    1454:	learn: 10.4294071	test: 11.7175758	best: 11.7148420 (1445)	total: 46.1s	remaining: 1m 20s
    1455:	learn: 10.4291657	test: 11.7160316	best: 11.7148420 (1445)	total: 46.1s	remaining: 1m 20s
    1456:	learn: 10.4274620	test: 11.7162527	best: 11.7148420 (1445)	total: 46.1s	remaining: 1m 20s
    1457:	learn: 10.4266762	test: 11.7165545	best: 11.7148420 (1445)	total: 46.1s	remaining: 1m 20s
    1458:	learn: 10.4255277	test: 11.7170607	best: 11.7148420 (1445)	total: 46.2s	remaining: 1m 20s
    1459:	learn: 10.4243118	test: 11.7178630	best: 11.7148420 (1445)	total: 46.2s	remaining: 1m 20s
    1460:	learn: 10.4239911	test: 11.7175230	best: 11.7148420 (1445)	total: 46.2s	remaining: 1m 20s
    1461:	learn: 10.4227729	test: 11.7176757	best: 11.7148420 (1445)	total: 46.3s	remaining: 1m 20s
    1462:	learn: 10.4218434	test: 11.7183968	best: 11.7148420 (1445)	total: 46.3s	remaining: 1m 20s
    1463:	learn: 10.4211279	test: 11.7175887	best: 11.7148420 (1445)	total: 46.3s	remaining: 1m 20s
    1464:	learn: 10.4176867	test: 11.7138303	best: 11.7138303 (1464)	total: 46.3s	remaining: 1m 20s
    1465:	learn: 10.4166584	test: 11.7149803	best: 11.7138303 (1464)	total: 46.4s	remaining: 1m 20s
    1466:	learn: 10.4162733	test: 11.7148552	best: 11.7138303 (1464)	total: 46.4s	remaining: 1m 20s
    1467:	learn: 10.4152348	test: 11.7147702	best: 11.7138303 (1464)	total: 46.4s	remaining: 1m 20s
    1468:	learn: 10.4126962	test: 11.7158103	best: 11.7138303 (1464)	total: 46.4s	remaining: 1m 20s
    1469:	learn: 10.4114559	test: 11.7137886	best: 11.7137886 (1469)	total: 46.5s	remaining: 1m 19s
    1470:	learn: 10.4109247	test: 11.7133085	best: 11.7133085 (1470)	total: 46.5s	remaining: 1m 19s
    1471:	learn: 10.4091204	test: 11.7138818	best: 11.7133085 (1470)	total: 46.5s	remaining: 1m 19s
    1472:	learn: 10.4060517	test: 11.7142078	best: 11.7133085 (1470)	total: 46.6s	remaining: 1m 19s
    1473:	learn: 10.4053639	test: 11.7148962	best: 11.7133085 (1470)	total: 46.6s	remaining: 1m 19s
    1474:	learn: 10.4046594	test: 11.7145750	best: 11.7133085 (1470)	total: 46.6s	remaining: 1m 19s
    1475:	learn: 10.4021936	test: 11.7130764	best: 11.7130764 (1475)	total: 46.6s	remaining: 1m 19s
    1476:	learn: 10.4005424	test: 11.7138542	best: 11.7130764 (1475)	total: 46.7s	remaining: 1m 19s
    1477:	learn: 10.4000317	test: 11.7146709	best: 11.7130764 (1475)	total: 46.7s	remaining: 1m 19s
    1478:	learn: 10.3987157	test: 11.7139698	best: 11.7130764 (1475)	total: 46.7s	remaining: 1m 19s
    1479:	learn: 10.3976694	test: 11.7150088	best: 11.7130764 (1475)	total: 46.7s	remaining: 1m 19s
    1480:	learn: 10.3964442	test: 11.7150347	best: 11.7130764 (1475)	total: 46.8s	remaining: 1m 19s
    1481:	learn: 10.3949781	test: 11.7152357	best: 11.7130764 (1475)	total: 46.8s	remaining: 1m 19s
    1482:	learn: 10.3940896	test: 11.7136055	best: 11.7130764 (1475)	total: 46.8s	remaining: 1m 19s
    1483:	learn: 10.3921629	test: 11.7131103	best: 11.7130764 (1475)	total: 46.9s	remaining: 1m 19s
    1484:	learn: 10.3911193	test: 11.7123076	best: 11.7123076 (1484)	total: 46.9s	remaining: 1m 19s
    1485:	learn: 10.3903571	test: 11.7129529	best: 11.7123076 (1484)	total: 46.9s	remaining: 1m 19s
    1486:	learn: 10.3897696	test: 11.7137026	best: 11.7123076 (1484)	total: 46.9s	remaining: 1m 19s
    1487:	learn: 10.3876974	test: 11.7139502	best: 11.7123076 (1484)	total: 47s	remaining: 1m 19s
    1488:	learn: 10.3859311	test: 11.7134121	best: 11.7123076 (1484)	total: 47s	remaining: 1m 19s
    1489:	learn: 10.3849365	test: 11.7132450	best: 11.7123076 (1484)	total: 47s	remaining: 1m 19s
    1490:	learn: 10.3817887	test: 11.7119675	best: 11.7119675 (1490)	total: 47.1s	remaining: 1m 19s
    1491:	learn: 10.3803298	test: 11.7130093	best: 11.7119675 (1490)	total: 47.1s	remaining: 1m 19s
    1492:	learn: 10.3791378	test: 11.7123992	best: 11.7119675 (1490)	total: 47.1s	remaining: 1m 19s
    1493:	learn: 10.3784963	test: 11.7118872	best: 11.7118872 (1493)	total: 47.1s	remaining: 1m 19s
    1494:	learn: 10.3771543	test: 11.7116894	best: 11.7116894 (1494)	total: 47.2s	remaining: 1m 19s
    1495:	learn: 10.3741258	test: 11.7124125	best: 11.7116894 (1494)	total: 47.2s	remaining: 1m 18s
    1496:	learn: 10.3730211	test: 11.7126215	best: 11.7116894 (1494)	total: 47.2s	remaining: 1m 18s
    1497:	learn: 10.3710786	test: 11.7110703	best: 11.7110703 (1497)	total: 47.2s	remaining: 1m 18s
    1498:	learn: 10.3691960	test: 11.7091105	best: 11.7091105 (1498)	total: 47.3s	remaining: 1m 18s
    1499:	learn: 10.3670928	test: 11.7081468	best: 11.7081468 (1499)	total: 47.3s	remaining: 1m 18s
    1500:	learn: 10.3650936	test: 11.7098332	best: 11.7081468 (1499)	total: 47.3s	remaining: 1m 18s
    1501:	learn: 10.3640181	test: 11.7088152	best: 11.7081468 (1499)	total: 47.4s	remaining: 1m 18s
    1502:	learn: 10.3627298	test: 11.7103839	best: 11.7081468 (1499)	total: 47.4s	remaining: 1m 18s
    1503:	learn: 10.3594314	test: 11.7089524	best: 11.7081468 (1499)	total: 47.4s	remaining: 1m 18s
    1504:	learn: 10.3574761	test: 11.7077404	best: 11.7077404 (1504)	total: 47.4s	remaining: 1m 18s
    1505:	learn: 10.3557602	test: 11.7055654	best: 11.7055654 (1505)	total: 47.5s	remaining: 1m 18s
    1506:	learn: 10.3550904	test: 11.7051204	best: 11.7051204 (1506)	total: 47.5s	remaining: 1m 18s
    1507:	learn: 10.3544913	test: 11.7055306	best: 11.7051204 (1506)	total: 47.5s	remaining: 1m 18s
    1508:	learn: 10.3526521	test: 11.7048795	best: 11.7048795 (1508)	total: 47.6s	remaining: 1m 18s
    1509:	learn: 10.3513216	test: 11.7031692	best: 11.7031692 (1509)	total: 47.6s	remaining: 1m 18s
    1510:	learn: 10.3486285	test: 11.7065883	best: 11.7031692 (1509)	total: 47.6s	remaining: 1m 18s
    1511:	learn: 10.3478068	test: 11.7045956	best: 11.7031692 (1509)	total: 47.6s	remaining: 1m 18s
    1512:	learn: 10.3450172	test: 11.7045827	best: 11.7031692 (1509)	total: 47.7s	remaining: 1m 18s
    1513:	learn: 10.3442460	test: 11.7053376	best: 11.7031692 (1509)	total: 47.7s	remaining: 1m 18s
    1514:	learn: 10.3401927	test: 11.7047409	best: 11.7031692 (1509)	total: 47.7s	remaining: 1m 18s
    1515:	learn: 10.3387829	test: 11.7058740	best: 11.7031692 (1509)	total: 47.8s	remaining: 1m 18s
    1516:	learn: 10.3365933	test: 11.7046614	best: 11.7031692 (1509)	total: 47.8s	remaining: 1m 18s
    1517:	learn: 10.3355482	test: 11.7056293	best: 11.7031692 (1509)	total: 47.8s	remaining: 1m 18s
    1518:	learn: 10.3344546	test: 11.7057275	best: 11.7031692 (1509)	total: 47.8s	remaining: 1m 18s
    1519:	learn: 10.3335580	test: 11.7045427	best: 11.7031692 (1509)	total: 47.9s	remaining: 1m 18s
    1520:	learn: 10.3317732	test: 11.7041175	best: 11.7031692 (1509)	total: 47.9s	remaining: 1m 18s
    1521:	learn: 10.3295849	test: 11.7044537	best: 11.7031692 (1509)	total: 47.9s	remaining: 1m 18s
    1522:	learn: 10.3281528	test: 11.7025846	best: 11.7025846 (1522)	total: 48s	remaining: 1m 17s
    1523:	learn: 10.3260905	test: 11.7004837	best: 11.7004837 (1523)	total: 48s	remaining: 1m 17s
    1524:	learn: 10.3232317	test: 11.6997618	best: 11.6997618 (1524)	total: 48s	remaining: 1m 17s
    1525:	learn: 10.3218213	test: 11.7009670	best: 11.6997618 (1524)	total: 48s	remaining: 1m 17s
    1526:	learn: 10.3202739	test: 11.7010613	best: 11.6997618 (1524)	total: 48.1s	remaining: 1m 17s
    1527:	learn: 10.3190053	test: 11.7011300	best: 11.6997618 (1524)	total: 48.1s	remaining: 1m 17s
    1528:	learn: 10.3183938	test: 11.6998884	best: 11.6997618 (1524)	total: 48.1s	remaining: 1m 17s
    1529:	learn: 10.3156450	test: 11.6979212	best: 11.6979212 (1529)	total: 48.2s	remaining: 1m 17s
    1530:	learn: 10.3137832	test: 11.6959832	best: 11.6959832 (1530)	total: 48.2s	remaining: 1m 17s
    1531:	learn: 10.3110878	test: 11.6944637	best: 11.6944637 (1531)	total: 48.2s	remaining: 1m 17s
    1532:	learn: 10.3107562	test: 11.6942779	best: 11.6942779 (1532)	total: 48.2s	remaining: 1m 17s
    1533:	learn: 10.3105406	test: 11.6944111	best: 11.6942779 (1532)	total: 48.3s	remaining: 1m 17s
    1534:	learn: 10.3095317	test: 11.6948388	best: 11.6942779 (1532)	total: 48.3s	remaining: 1m 17s
    1535:	learn: 10.3088364	test: 11.6946485	best: 11.6942779 (1532)	total: 48.3s	remaining: 1m 17s
    1536:	learn: 10.3082788	test: 11.6932017	best: 11.6932017 (1536)	total: 48.4s	remaining: 1m 17s
    1537:	learn: 10.3066158	test: 11.6916139	best: 11.6916139 (1537)	total: 48.4s	remaining: 1m 17s
    1538:	learn: 10.3047781	test: 11.6917675	best: 11.6916139 (1537)	total: 48.5s	remaining: 1m 17s
    1539:	learn: 10.3034748	test: 11.6919181	best: 11.6916139 (1537)	total: 48.5s	remaining: 1m 17s
    1540:	learn: 10.3025322	test: 11.6934530	best: 11.6916139 (1537)	total: 48.5s	remaining: 1m 17s
    1541:	learn: 10.2993498	test: 11.6936829	best: 11.6916139 (1537)	total: 48.5s	remaining: 1m 17s
    1542:	learn: 10.2971506	test: 11.6920517	best: 11.6916139 (1537)	total: 48.6s	remaining: 1m 17s
    1543:	learn: 10.2968152	test: 11.6914181	best: 11.6914181 (1543)	total: 48.6s	remaining: 1m 17s
    1544:	learn: 10.2934053	test: 11.6909140	best: 11.6909140 (1544)	total: 48.6s	remaining: 1m 17s
    1545:	learn: 10.2912350	test: 11.6895431	best: 11.6895431 (1545)	total: 48.7s	remaining: 1m 17s
    1546:	learn: 10.2907057	test: 11.6896527	best: 11.6895431 (1545)	total: 48.7s	remaining: 1m 17s
    1547:	learn: 10.2901373	test: 11.6895737	best: 11.6895431 (1545)	total: 48.7s	remaining: 1m 17s
    1548:	learn: 10.2889462	test: 11.6898422	best: 11.6895431 (1545)	total: 48.7s	remaining: 1m 17s
    1549:	learn: 10.2887594	test: 11.6905552	best: 11.6895431 (1545)	total: 48.8s	remaining: 1m 17s
    1550:	learn: 10.2881266	test: 11.6900820	best: 11.6895431 (1545)	total: 48.8s	remaining: 1m 17s
    1551:	learn: 10.2853803	test: 11.6889499	best: 11.6889499 (1551)	total: 48.8s	remaining: 1m 17s
    1552:	learn: 10.2837297	test: 11.6883971	best: 11.6883971 (1552)	total: 48.9s	remaining: 1m 16s
    1553:	learn: 10.2819342	test: 11.6868645	best: 11.6868645 (1553)	total: 48.9s	remaining: 1m 16s
    1554:	learn: 10.2809777	test: 11.6894443	best: 11.6868645 (1553)	total: 48.9s	remaining: 1m 16s
    1555:	learn: 10.2801308	test: 11.6903241	best: 11.6868645 (1553)	total: 48.9s	remaining: 1m 16s
    1556:	learn: 10.2779512	test: 11.6904279	best: 11.6868645 (1553)	total: 49s	remaining: 1m 16s
    1557:	learn: 10.2757923	test: 11.6915143	best: 11.6868645 (1553)	total: 49s	remaining: 1m 16s
    1558:	learn: 10.2725796	test: 11.6920673	best: 11.6868645 (1553)	total: 49s	remaining: 1m 16s
    1559:	learn: 10.2710210	test: 11.6911029	best: 11.6868645 (1553)	total: 49s	remaining: 1m 16s
    1560:	learn: 10.2680050	test: 11.6891252	best: 11.6868645 (1553)	total: 49.1s	remaining: 1m 16s
    1561:	learn: 10.2679035	test: 11.6891138	best: 11.6868645 (1553)	total: 49.1s	remaining: 1m 16s
    1562:	learn: 10.2665533	test: 11.6889701	best: 11.6868645 (1553)	total: 49.1s	remaining: 1m 16s
    1563:	learn: 10.2658190	test: 11.6883866	best: 11.6868645 (1553)	total: 49.2s	remaining: 1m 16s
    1564:	learn: 10.2638497	test: 11.6893266	best: 11.6868645 (1553)	total: 49.2s	remaining: 1m 16s
    1565:	learn: 10.2599788	test: 11.6854300	best: 11.6854300 (1565)	total: 49.2s	remaining: 1m 16s
    1566:	learn: 10.2585673	test: 11.6855256	best: 11.6854300 (1565)	total: 49.2s	remaining: 1m 16s
    1567:	learn: 10.2577869	test: 11.6856758	best: 11.6854300 (1565)	total: 49.3s	remaining: 1m 16s
    1568:	learn: 10.2570956	test: 11.6846483	best: 11.6846483 (1568)	total: 49.3s	remaining: 1m 16s
    1569:	learn: 10.2537445	test: 11.6819010	best: 11.6819010 (1569)	total: 49.3s	remaining: 1m 16s
    1570:	learn: 10.2535328	test: 11.6825714	best: 11.6819010 (1569)	total: 49.4s	remaining: 1m 16s
    1571:	learn: 10.2534611	test: 11.6825610	best: 11.6819010 (1569)	total: 49.4s	remaining: 1m 16s
    1572:	learn: 10.2518568	test: 11.6836213	best: 11.6819010 (1569)	total: 49.4s	remaining: 1m 16s
    1573:	learn: 10.2494479	test: 11.6823199	best: 11.6819010 (1569)	total: 49.4s	remaining: 1m 16s
    1574:	learn: 10.2471134	test: 11.6803762	best: 11.6803762 (1574)	total: 49.5s	remaining: 1m 16s
    1575:	learn: 10.2455102	test: 11.6796131	best: 11.6796131 (1575)	total: 49.5s	remaining: 1m 16s
    1576:	learn: 10.2441116	test: 11.6810127	best: 11.6796131 (1575)	total: 49.5s	remaining: 1m 16s
    1577:	learn: 10.2434998	test: 11.6820580	best: 11.6796131 (1575)	total: 49.5s	remaining: 1m 16s
    1578:	learn: 10.2426240	test: 11.6813832	best: 11.6796131 (1575)	total: 49.6s	remaining: 1m 16s
    1579:	learn: 10.2419293	test: 11.6810355	best: 11.6796131 (1575)	total: 49.6s	remaining: 1m 15s
    1580:	learn: 10.2403164	test: 11.6810335	best: 11.6796131 (1575)	total: 49.6s	remaining: 1m 15s
    1581:	learn: 10.2392136	test: 11.6816655	best: 11.6796131 (1575)	total: 49.7s	remaining: 1m 15s
    1582:	learn: 10.2360146	test: 11.6808754	best: 11.6796131 (1575)	total: 49.7s	remaining: 1m 15s
    1583:	learn: 10.2353726	test: 11.6808211	best: 11.6796131 (1575)	total: 49.7s	remaining: 1m 15s
    1584:	learn: 10.2346242	test: 11.6800847	best: 11.6796131 (1575)	total: 49.7s	remaining: 1m 15s
    1585:	learn: 10.2324469	test: 11.6795629	best: 11.6795629 (1585)	total: 49.8s	remaining: 1m 15s
    1586:	learn: 10.2313149	test: 11.6798345	best: 11.6795629 (1585)	total: 49.8s	remaining: 1m 15s
    1587:	learn: 10.2275711	test: 11.6789449	best: 11.6789449 (1587)	total: 49.8s	remaining: 1m 15s
    1588:	learn: 10.2263426	test: 11.6775318	best: 11.6775318 (1588)	total: 49.8s	remaining: 1m 15s
    1589:	learn: 10.2249514	test: 11.6783658	best: 11.6775318 (1588)	total: 49.9s	remaining: 1m 15s
    1590:	learn: 10.2220229	test: 11.6772335	best: 11.6772335 (1590)	total: 49.9s	remaining: 1m 15s
    1591:	learn: 10.2208044	test: 11.6759532	best: 11.6759532 (1591)	total: 49.9s	remaining: 1m 15s
    1592:	learn: 10.2198222	test: 11.6757532	best: 11.6757532 (1592)	total: 50s	remaining: 1m 15s
    1593:	learn: 10.2189993	test: 11.6752949	best: 11.6752949 (1593)	total: 50s	remaining: 1m 15s
    1594:	learn: 10.2175551	test: 11.6756013	best: 11.6752949 (1593)	total: 50s	remaining: 1m 15s
    1595:	learn: 10.2161918	test: 11.6737084	best: 11.6737084 (1595)	total: 50s	remaining: 1m 15s
    1596:	learn: 10.2157366	test: 11.6744194	best: 11.6737084 (1595)	total: 50.1s	remaining: 1m 15s
    1597:	learn: 10.2154003	test: 11.6747921	best: 11.6737084 (1595)	total: 50.1s	remaining: 1m 15s
    1598:	learn: 10.2144229	test: 11.6756111	best: 11.6737084 (1595)	total: 50.1s	remaining: 1m 15s
    1599:	learn: 10.2137257	test: 11.6746338	best: 11.6737084 (1595)	total: 50.2s	remaining: 1m 15s
    1600:	learn: 10.2126207	test: 11.6738269	best: 11.6737084 (1595)	total: 50.2s	remaining: 1m 15s
    1601:	learn: 10.2116799	test: 11.6738081	best: 11.6737084 (1595)	total: 50.2s	remaining: 1m 15s
    1602:	learn: 10.2102766	test: 11.6747521	best: 11.6737084 (1595)	total: 50.2s	remaining: 1m 15s
    1603:	learn: 10.2079715	test: 11.6725698	best: 11.6725698 (1603)	total: 50.3s	remaining: 1m 15s
    1604:	learn: 10.2060763	test: 11.6718652	best: 11.6718652 (1604)	total: 50.3s	remaining: 1m 15s
    1605:	learn: 10.2042889	test: 11.6720874	best: 11.6718652 (1604)	total: 50.3s	remaining: 1m 15s
    1606:	learn: 10.2031184	test: 11.6729438	best: 11.6718652 (1604)	total: 50.4s	remaining: 1m 14s
    1607:	learn: 10.2012557	test: 11.6742532	best: 11.6718652 (1604)	total: 50.4s	remaining: 1m 14s
    1608:	learn: 10.2005581	test: 11.6727356	best: 11.6718652 (1604)	total: 50.4s	remaining: 1m 14s
    1609:	learn: 10.1984218	test: 11.6736420	best: 11.6718652 (1604)	total: 50.4s	remaining: 1m 14s
    1610:	learn: 10.1964303	test: 11.6745125	best: 11.6718652 (1604)	total: 50.5s	remaining: 1m 14s
    1611:	learn: 10.1954591	test: 11.6748183	best: 11.6718652 (1604)	total: 50.5s	remaining: 1m 14s
    1612:	learn: 10.1950723	test: 11.6736974	best: 11.6718652 (1604)	total: 50.5s	remaining: 1m 14s
    1613:	learn: 10.1929162	test: 11.6728741	best: 11.6718652 (1604)	total: 50.5s	remaining: 1m 14s
    1614:	learn: 10.1917954	test: 11.6730610	best: 11.6718652 (1604)	total: 50.6s	remaining: 1m 14s
    1615:	learn: 10.1907648	test: 11.6720692	best: 11.6718652 (1604)	total: 50.6s	remaining: 1m 14s
    1616:	learn: 10.1898770	test: 11.6708842	best: 11.6708842 (1616)	total: 50.6s	remaining: 1m 14s
    1617:	learn: 10.1894681	test: 11.6702716	best: 11.6702716 (1617)	total: 50.6s	remaining: 1m 14s
    1618:	learn: 10.1886579	test: 11.6699423	best: 11.6699423 (1618)	total: 50.7s	remaining: 1m 14s
    1619:	learn: 10.1882008	test: 11.6697763	best: 11.6697763 (1619)	total: 50.7s	remaining: 1m 14s
    1620:	learn: 10.1868746	test: 11.6703686	best: 11.6697763 (1619)	total: 50.7s	remaining: 1m 14s
    1621:	learn: 10.1863997	test: 11.6700799	best: 11.6697763 (1619)	total: 50.8s	remaining: 1m 14s
    1622:	learn: 10.1859498	test: 11.6704338	best: 11.6697763 (1619)	total: 50.8s	remaining: 1m 14s
    1623:	learn: 10.1847585	test: 11.6692099	best: 11.6692099 (1623)	total: 50.8s	remaining: 1m 14s
    1624:	learn: 10.1836591	test: 11.6675026	best: 11.6675026 (1624)	total: 50.8s	remaining: 1m 14s
    1625:	learn: 10.1829724	test: 11.6671059	best: 11.6671059 (1625)	total: 50.9s	remaining: 1m 14s
    1626:	learn: 10.1808741	test: 11.6681991	best: 11.6671059 (1625)	total: 50.9s	remaining: 1m 14s
    1627:	learn: 10.1795101	test: 11.6686180	best: 11.6671059 (1625)	total: 50.9s	remaining: 1m 14s
    1628:	learn: 10.1793243	test: 11.6692195	best: 11.6671059 (1625)	total: 50.9s	remaining: 1m 14s
    1629:	learn: 10.1783088	test: 11.6697134	best: 11.6671059 (1625)	total: 51s	remaining: 1m 14s
    1630:	learn: 10.1776843	test: 11.6701379	best: 11.6671059 (1625)	total: 51s	remaining: 1m 14s
    1631:	learn: 10.1763845	test: 11.6716014	best: 11.6671059 (1625)	total: 51s	remaining: 1m 14s
    1632:	learn: 10.1732689	test: 11.6719747	best: 11.6671059 (1625)	total: 51s	remaining: 1m 13s
    1633:	learn: 10.1716594	test: 11.6718226	best: 11.6671059 (1625)	total: 51.1s	remaining: 1m 13s
    1634:	learn: 10.1680828	test: 11.6709739	best: 11.6671059 (1625)	total: 51.1s	remaining: 1m 13s
    1635:	learn: 10.1668831	test: 11.6705188	best: 11.6671059 (1625)	total: 51.1s	remaining: 1m 13s
    1636:	learn: 10.1649740	test: 11.6689654	best: 11.6671059 (1625)	total: 51.2s	remaining: 1m 13s
    1637:	learn: 10.1640836	test: 11.6689266	best: 11.6671059 (1625)	total: 51.2s	remaining: 1m 13s
    1638:	learn: 10.1637348	test: 11.6689914	best: 11.6671059 (1625)	total: 51.2s	remaining: 1m 13s
    1639:	learn: 10.1635655	test: 11.6699348	best: 11.6671059 (1625)	total: 51.2s	remaining: 1m 13s
    1640:	learn: 10.1627622	test: 11.6689490	best: 11.6671059 (1625)	total: 51.3s	remaining: 1m 13s
    1641:	learn: 10.1602116	test: 11.6689169	best: 11.6671059 (1625)	total: 51.3s	remaining: 1m 13s
    1642:	learn: 10.1589054	test: 11.6681426	best: 11.6671059 (1625)	total: 51.3s	remaining: 1m 13s
    1643:	learn: 10.1563692	test: 11.6706524	best: 11.6671059 (1625)	total: 51.4s	remaining: 1m 13s
    1644:	learn: 10.1551787	test: 11.6694295	best: 11.6671059 (1625)	total: 51.4s	remaining: 1m 13s
    1645:	learn: 10.1542228	test: 11.6680750	best: 11.6671059 (1625)	total: 51.4s	remaining: 1m 13s
    1646:	learn: 10.1525770	test: 11.6668521	best: 11.6668521 (1646)	total: 51.4s	remaining: 1m 13s
    1647:	learn: 10.1517828	test: 11.6671496	best: 11.6668521 (1646)	total: 51.5s	remaining: 1m 13s
    1648:	learn: 10.1515355	test: 11.6673773	best: 11.6668521 (1646)	total: 51.5s	remaining: 1m 13s
    1649:	learn: 10.1495229	test: 11.6678658	best: 11.6668521 (1646)	total: 51.5s	remaining: 1m 13s
    1650:	learn: 10.1489729	test: 11.6693285	best: 11.6668521 (1646)	total: 51.5s	remaining: 1m 13s
    1651:	learn: 10.1450086	test: 11.6654567	best: 11.6654567 (1651)	total: 51.6s	remaining: 1m 13s
    1652:	learn: 10.1444921	test: 11.6658269	best: 11.6654567 (1651)	total: 51.6s	remaining: 1m 13s
    1653:	learn: 10.1409248	test: 11.6633041	best: 11.6633041 (1653)	total: 51.6s	remaining: 1m 13s
    1654:	learn: 10.1398871	test: 11.6642252	best: 11.6633041 (1653)	total: 51.7s	remaining: 1m 13s
    1655:	learn: 10.1380140	test: 11.6640665	best: 11.6633041 (1653)	total: 51.7s	remaining: 1m 13s
    1656:	learn: 10.1376629	test: 11.6640826	best: 11.6633041 (1653)	total: 51.7s	remaining: 1m 13s
    1657:	learn: 10.1343612	test: 11.6602843	best: 11.6602843 (1657)	total: 51.7s	remaining: 1m 13s
    1658:	learn: 10.1310045	test: 11.6604153	best: 11.6602843 (1657)	total: 51.8s	remaining: 1m 13s
    1659:	learn: 10.1298748	test: 11.6607068	best: 11.6602843 (1657)	total: 51.8s	remaining: 1m 13s
    1660:	learn: 10.1268359	test: 11.6591397	best: 11.6591397 (1660)	total: 51.8s	remaining: 1m 12s
    1661:	learn: 10.1258580	test: 11.6587333	best: 11.6587333 (1661)	total: 51.8s	remaining: 1m 12s
    1662:	learn: 10.1236903	test: 11.6560413	best: 11.6560413 (1662)	total: 51.9s	remaining: 1m 12s
    1663:	learn: 10.1228752	test: 11.6547398	best: 11.6547398 (1663)	total: 51.9s	remaining: 1m 12s
    1664:	learn: 10.1204701	test: 11.6539058	best: 11.6539058 (1664)	total: 51.9s	remaining: 1m 12s
    1665:	learn: 10.1187022	test: 11.6540584	best: 11.6539058 (1664)	total: 51.9s	remaining: 1m 12s
    1666:	learn: 10.1175430	test: 11.6544848	best: 11.6539058 (1664)	total: 52s	remaining: 1m 12s
    1667:	learn: 10.1171183	test: 11.6550003	best: 11.6539058 (1664)	total: 52s	remaining: 1m 12s
    1668:	learn: 10.1167367	test: 11.6555790	best: 11.6539058 (1664)	total: 52s	remaining: 1m 12s
    1669:	learn: 10.1152101	test: 11.6545547	best: 11.6539058 (1664)	total: 52.1s	remaining: 1m 12s
    1670:	learn: 10.1140726	test: 11.6543361	best: 11.6539058 (1664)	total: 52.1s	remaining: 1m 12s
    1671:	learn: 10.1135492	test: 11.6538035	best: 11.6538035 (1671)	total: 52.1s	remaining: 1m 12s
    1672:	learn: 10.1124032	test: 11.6524747	best: 11.6524747 (1672)	total: 52.1s	remaining: 1m 12s
    1673:	learn: 10.1119701	test: 11.6529158	best: 11.6524747 (1672)	total: 52.2s	remaining: 1m 12s
    1674:	learn: 10.1115702	test: 11.6516314	best: 11.6516314 (1674)	total: 52.2s	remaining: 1m 12s
    1675:	learn: 10.1100163	test: 11.6526243	best: 11.6516314 (1674)	total: 52.2s	remaining: 1m 12s
    1676:	learn: 10.1084874	test: 11.6512649	best: 11.6512649 (1676)	total: 52.3s	remaining: 1m 12s
    1677:	learn: 10.1075479	test: 11.6511549	best: 11.6511549 (1677)	total: 52.3s	remaining: 1m 12s
    1678:	learn: 10.1057124	test: 11.6511991	best: 11.6511549 (1677)	total: 52.3s	remaining: 1m 12s
    1679:	learn: 10.1049411	test: 11.6522196	best: 11.6511549 (1677)	total: 52.3s	remaining: 1m 12s
    1680:	learn: 10.1030239	test: 11.6519095	best: 11.6511549 (1677)	total: 52.4s	remaining: 1m 12s
    1681:	learn: 10.1001243	test: 11.6517610	best: 11.6511549 (1677)	total: 52.4s	remaining: 1m 12s
    1682:	learn: 10.0988031	test: 11.6524286	best: 11.6511549 (1677)	total: 52.4s	remaining: 1m 12s
    1683:	learn: 10.0966938	test: 11.6514579	best: 11.6511549 (1677)	total: 52.5s	remaining: 1m 12s
    1684:	learn: 10.0950823	test: 11.6506515	best: 11.6506515 (1684)	total: 52.5s	remaining: 1m 12s
    1685:	learn: 10.0936661	test: 11.6513180	best: 11.6506515 (1684)	total: 52.5s	remaining: 1m 12s
    1686:	learn: 10.0926584	test: 11.6516059	best: 11.6506515 (1684)	total: 52.5s	remaining: 1m 12s
    1687:	learn: 10.0910476	test: 11.6511418	best: 11.6506515 (1684)	total: 52.6s	remaining: 1m 12s
    1688:	learn: 10.0903070	test: 11.6514239	best: 11.6506515 (1684)	total: 52.6s	remaining: 1m 11s
    1689:	learn: 10.0886862	test: 11.6506361	best: 11.6506361 (1689)	total: 52.7s	remaining: 1m 11s
    1690:	learn: 10.0883517	test: 11.6495368	best: 11.6495368 (1690)	total: 52.7s	remaining: 1m 11s
    1691:	learn: 10.0879265	test: 11.6484487	best: 11.6484487 (1691)	total: 52.7s	remaining: 1m 11s
    1692:	learn: 10.0859210	test: 11.6480057	best: 11.6480057 (1692)	total: 52.8s	remaining: 1m 11s
    1693:	learn: 10.0824201	test: 11.6478377	best: 11.6478377 (1693)	total: 52.8s	remaining: 1m 11s
    1694:	learn: 10.0816542	test: 11.6475026	best: 11.6475026 (1694)	total: 52.8s	remaining: 1m 11s
    1695:	learn: 10.0785482	test: 11.6464607	best: 11.6464607 (1695)	total: 52.8s	remaining: 1m 11s
    1696:	learn: 10.0778898	test: 11.6450740	best: 11.6450740 (1696)	total: 52.9s	remaining: 1m 11s
    1697:	learn: 10.0748912	test: 11.6458368	best: 11.6450740 (1696)	total: 52.9s	remaining: 1m 11s
    1698:	learn: 10.0737512	test: 11.6450123	best: 11.6450123 (1698)	total: 52.9s	remaining: 1m 11s
    1699:	learn: 10.0733703	test: 11.6443413	best: 11.6443413 (1699)	total: 53s	remaining: 1m 11s
    1700:	learn: 10.0721200	test: 11.6442798	best: 11.6442798 (1700)	total: 53s	remaining: 1m 11s
    1701:	learn: 10.0701768	test: 11.6471417	best: 11.6442798 (1700)	total: 53s	remaining: 1m 11s
    1702:	learn: 10.0686277	test: 11.6466949	best: 11.6442798 (1700)	total: 53s	remaining: 1m 11s
    1703:	learn: 10.0676458	test: 11.6453946	best: 11.6442798 (1700)	total: 53.1s	remaining: 1m 11s
    1704:	learn: 10.0661731	test: 11.6469621	best: 11.6442798 (1700)	total: 53.1s	remaining: 1m 11s
    1705:	learn: 10.0653763	test: 11.6469191	best: 11.6442798 (1700)	total: 53.1s	remaining: 1m 11s
    1706:	learn: 10.0642014	test: 11.6466017	best: 11.6442798 (1700)	total: 53.2s	remaining: 1m 11s
    1707:	learn: 10.0616047	test: 11.6456841	best: 11.6442798 (1700)	total: 53.2s	remaining: 1m 11s
    1708:	learn: 10.0604299	test: 11.6450125	best: 11.6442798 (1700)	total: 53.2s	remaining: 1m 11s
    1709:	learn: 10.0581021	test: 11.6448306	best: 11.6442798 (1700)	total: 53.3s	remaining: 1m 11s
    1710:	learn: 10.0564818	test: 11.6442556	best: 11.6442556 (1710)	total: 53.3s	remaining: 1m 11s
    1711:	learn: 10.0566012	test: 11.6441577	best: 11.6441577 (1711)	total: 53.3s	remaining: 1m 11s
    1712:	learn: 10.0554583	test: 11.6448684	best: 11.6441577 (1711)	total: 53.3s	remaining: 1m 11s
    1713:	learn: 10.0545774	test: 11.6465020	best: 11.6441577 (1711)	total: 53.4s	remaining: 1m 11s
    1714:	learn: 10.0537999	test: 11.6470296	best: 11.6441577 (1711)	total: 53.4s	remaining: 1m 11s
    1715:	learn: 10.0534004	test: 11.6470860	best: 11.6441577 (1711)	total: 53.4s	remaining: 1m 11s
    1716:	learn: 10.0517345	test: 11.6455492	best: 11.6441577 (1711)	total: 53.4s	remaining: 1m 11s
    1717:	learn: 10.0503339	test: 11.6441845	best: 11.6441577 (1711)	total: 53.5s	remaining: 1m 11s
    1718:	learn: 10.0486883	test: 11.6444258	best: 11.6441577 (1711)	total: 53.5s	remaining: 1m 10s
    1719:	learn: 10.0484637	test: 11.6434938	best: 11.6434938 (1719)	total: 53.5s	remaining: 1m 10s
    1720:	learn: 10.0460371	test: 11.6431686	best: 11.6431686 (1720)	total: 53.6s	remaining: 1m 10s
    1721:	learn: 10.0425687	test: 11.6423106	best: 11.6423106 (1721)	total: 53.6s	remaining: 1m 10s
    1722:	learn: 10.0407549	test: 11.6419967	best: 11.6419967 (1722)	total: 53.6s	remaining: 1m 10s
    1723:	learn: 10.0393867	test: 11.6421759	best: 11.6419967 (1722)	total: 53.6s	remaining: 1m 10s
    1724:	learn: 10.0386443	test: 11.6420762	best: 11.6419967 (1722)	total: 53.7s	remaining: 1m 10s
    1725:	learn: 10.0362913	test: 11.6447223	best: 11.6419967 (1722)	total: 53.7s	remaining: 1m 10s
    1726:	learn: 10.0337346	test: 11.6447973	best: 11.6419967 (1722)	total: 53.7s	remaining: 1m 10s
    1727:	learn: 10.0324222	test: 11.6453553	best: 11.6419967 (1722)	total: 53.8s	remaining: 1m 10s
    1728:	learn: 10.0322032	test: 11.6435350	best: 11.6419967 (1722)	total: 53.8s	remaining: 1m 10s
    1729:	learn: 10.0299480	test: 11.6442943	best: 11.6419967 (1722)	total: 53.8s	remaining: 1m 10s
    1730:	learn: 10.0290595	test: 11.6421635	best: 11.6419967 (1722)	total: 53.8s	remaining: 1m 10s
    1731:	learn: 10.0272989	test: 11.6441877	best: 11.6419967 (1722)	total: 53.9s	remaining: 1m 10s
    1732:	learn: 10.0252026	test: 11.6438708	best: 11.6419967 (1722)	total: 53.9s	remaining: 1m 10s
    1733:	learn: 10.0222264	test: 11.6407797	best: 11.6407797 (1733)	total: 53.9s	remaining: 1m 10s
    1734:	learn: 10.0214793	test: 11.6423429	best: 11.6407797 (1733)	total: 53.9s	remaining: 1m 10s
    1735:	learn: 10.0203488	test: 11.6421564	best: 11.6407797 (1733)	total: 54s	remaining: 1m 10s
    1736:	learn: 10.0186249	test: 11.6438150	best: 11.6407797 (1733)	total: 54s	remaining: 1m 10s
    1737:	learn: 10.0167096	test: 11.6432244	best: 11.6407797 (1733)	total: 54s	remaining: 1m 10s
    1738:	learn: 10.0153191	test: 11.6427962	best: 11.6407797 (1733)	total: 54.1s	remaining: 1m 10s
    1739:	learn: 10.0128156	test: 11.6439531	best: 11.6407797 (1733)	total: 54.1s	remaining: 1m 10s
    1740:	learn: 10.0121882	test: 11.6437836	best: 11.6407797 (1733)	total: 54.1s	remaining: 1m 10s
    1741:	learn: 10.0115212	test: 11.6439018	best: 11.6407797 (1733)	total: 54.1s	remaining: 1m 10s
    1742:	learn: 10.0095938	test: 11.6416393	best: 11.6407797 (1733)	total: 54.2s	remaining: 1m 10s
    1743:	learn: 10.0074441	test: 11.6395034	best: 11.6395034 (1743)	total: 54.2s	remaining: 1m 10s
    1744:	learn: 10.0061923	test: 11.6378110	best: 11.6378110 (1744)	total: 54.2s	remaining: 1m 10s
    1745:	learn: 10.0054465	test: 11.6374143	best: 11.6374143 (1745)	total: 54.3s	remaining: 1m 10s
    1746:	learn: 10.0037281	test: 11.6387114	best: 11.6374143 (1745)	total: 54.3s	remaining: 1m 10s
    1747:	learn: 10.0012018	test: 11.6393271	best: 11.6374143 (1745)	total: 54.3s	remaining: 1m 9s
    1748:	learn: 10.0001516	test: 11.6407876	best: 11.6374143 (1745)	total: 54.3s	remaining: 1m 9s
    1749:	learn: 9.9993477	test: 11.6408124	best: 11.6374143 (1745)	total: 54.4s	remaining: 1m 9s
    1750:	learn: 9.9985754	test: 11.6406734	best: 11.6374143 (1745)	total: 54.4s	remaining: 1m 9s
    1751:	learn: 9.9966852	test: 11.6413524	best: 11.6374143 (1745)	total: 54.4s	remaining: 1m 9s
    1752:	learn: 9.9956141	test: 11.6424279	best: 11.6374143 (1745)	total: 54.4s	remaining: 1m 9s
    1753:	learn: 9.9955374	test: 11.6417121	best: 11.6374143 (1745)	total: 54.5s	remaining: 1m 9s
    1754:	learn: 9.9948915	test: 11.6412706	best: 11.6374143 (1745)	total: 54.5s	remaining: 1m 9s
    1755:	learn: 9.9940649	test: 11.6410654	best: 11.6374143 (1745)	total: 54.5s	remaining: 1m 9s
    1756:	learn: 9.9903785	test: 11.6407181	best: 11.6374143 (1745)	total: 54.5s	remaining: 1m 9s
    1757:	learn: 9.9879610	test: 11.6412967	best: 11.6374143 (1745)	total: 54.6s	remaining: 1m 9s
    1758:	learn: 9.9875825	test: 11.6411711	best: 11.6374143 (1745)	total: 54.6s	remaining: 1m 9s
    1759:	learn: 9.9862348	test: 11.6420553	best: 11.6374143 (1745)	total: 54.6s	remaining: 1m 9s
    1760:	learn: 9.9843293	test: 11.6413589	best: 11.6374143 (1745)	total: 54.7s	remaining: 1m 9s
    1761:	learn: 9.9824132	test: 11.6415385	best: 11.6374143 (1745)	total: 54.7s	remaining: 1m 9s
    1762:	learn: 9.9818094	test: 11.6419350	best: 11.6374143 (1745)	total: 54.7s	remaining: 1m 9s
    1763:	learn: 9.9788782	test: 11.6402953	best: 11.6374143 (1745)	total: 54.7s	remaining: 1m 9s
    1764:	learn: 9.9780878	test: 11.6395405	best: 11.6374143 (1745)	total: 54.8s	remaining: 1m 9s
    1765:	learn: 9.9775305	test: 11.6406320	best: 11.6374143 (1745)	total: 54.8s	remaining: 1m 9s
    1766:	learn: 9.9750127	test: 11.6380896	best: 11.6374143 (1745)	total: 54.8s	remaining: 1m 9s
    1767:	learn: 9.9730683	test: 11.6361091	best: 11.6361091 (1767)	total: 54.8s	remaining: 1m 9s
    1768:	learn: 9.9695397	test: 11.6341722	best: 11.6341722 (1768)	total: 54.9s	remaining: 1m 9s
    1769:	learn: 9.9677253	test: 11.6321864	best: 11.6321864 (1769)	total: 54.9s	remaining: 1m 9s
    1770:	learn: 9.9665445	test: 11.6331481	best: 11.6321864 (1769)	total: 54.9s	remaining: 1m 9s
    1771:	learn: 9.9645472	test: 11.6326842	best: 11.6321864 (1769)	total: 55s	remaining: 1m 9s
    1772:	learn: 9.9630836	test: 11.6330494	best: 11.6321864 (1769)	total: 55s	remaining: 1m 9s
    1773:	learn: 9.9606582	test: 11.6329804	best: 11.6321864 (1769)	total: 55s	remaining: 1m 9s
    1774:	learn: 9.9592698	test: 11.6334633	best: 11.6321864 (1769)	total: 55s	remaining: 1m 9s
    1775:	learn: 9.9579500	test: 11.6317086	best: 11.6317086 (1775)	total: 55.1s	remaining: 1m 8s
    1776:	learn: 9.9560904	test: 11.6287537	best: 11.6287537 (1776)	total: 55.1s	remaining: 1m 8s
    1777:	learn: 9.9549425	test: 11.6299160	best: 11.6287537 (1776)	total: 55.1s	remaining: 1m 8s
    1778:	learn: 9.9528785	test: 11.6297376	best: 11.6287537 (1776)	total: 55.2s	remaining: 1m 8s
    1779:	learn: 9.9518586	test: 11.6300565	best: 11.6287537 (1776)	total: 55.2s	remaining: 1m 8s
    1780:	learn: 9.9512275	test: 11.6305575	best: 11.6287537 (1776)	total: 55.2s	remaining: 1m 8s
    1781:	learn: 9.9507448	test: 11.6312143	best: 11.6287537 (1776)	total: 55.2s	remaining: 1m 8s
    1782:	learn: 9.9490227	test: 11.6329613	best: 11.6287537 (1776)	total: 55.3s	remaining: 1m 8s
    1783:	learn: 9.9471360	test: 11.6331550	best: 11.6287537 (1776)	total: 55.3s	remaining: 1m 8s
    1784:	learn: 9.9460473	test: 11.6337556	best: 11.6287537 (1776)	total: 55.3s	remaining: 1m 8s
    1785:	learn: 9.9455563	test: 11.6343542	best: 11.6287537 (1776)	total: 55.3s	remaining: 1m 8s
    1786:	learn: 9.9431622	test: 11.6344650	best: 11.6287537 (1776)	total: 55.4s	remaining: 1m 8s
    1787:	learn: 9.9414754	test: 11.6346522	best: 11.6287537 (1776)	total: 55.4s	remaining: 1m 8s
    1788:	learn: 9.9402809	test: 11.6326455	best: 11.6287537 (1776)	total: 55.4s	remaining: 1m 8s
    1789:	learn: 9.9371362	test: 11.6352295	best: 11.6287537 (1776)	total: 55.5s	remaining: 1m 8s
    1790:	learn: 9.9367507	test: 11.6358858	best: 11.6287537 (1776)	total: 55.5s	remaining: 1m 8s
    1791:	learn: 9.9360680	test: 11.6349452	best: 11.6287537 (1776)	total: 55.5s	remaining: 1m 8s
    1792:	learn: 9.9350073	test: 11.6361001	best: 11.6287537 (1776)	total: 55.5s	remaining: 1m 8s
    1793:	learn: 9.9338328	test: 11.6360284	best: 11.6287537 (1776)	total: 55.6s	remaining: 1m 8s
    1794:	learn: 9.9326383	test: 11.6356616	best: 11.6287537 (1776)	total: 55.6s	remaining: 1m 8s
    1795:	learn: 9.9317156	test: 11.6376019	best: 11.6287537 (1776)	total: 55.6s	remaining: 1m 8s
    1796:	learn: 9.9304071	test: 11.6382136	best: 11.6287537 (1776)	total: 55.7s	remaining: 1m 8s
    1797:	learn: 9.9288325	test: 11.6372998	best: 11.6287537 (1776)	total: 55.7s	remaining: 1m 8s
    1798:	learn: 9.9283517	test: 11.6366051	best: 11.6287537 (1776)	total: 55.7s	remaining: 1m 8s
    1799:	learn: 9.9281284	test: 11.6360643	best: 11.6287537 (1776)	total: 55.7s	remaining: 1m 8s
    1800:	learn: 9.9272271	test: 11.6344332	best: 11.6287537 (1776)	total: 55.8s	remaining: 1m 8s
    1801:	learn: 9.9271142	test: 11.6350833	best: 11.6287537 (1776)	total: 55.8s	remaining: 1m 8s
    1802:	learn: 9.9263963	test: 11.6352951	best: 11.6287537 (1776)	total: 55.8s	remaining: 1m 8s
    1803:	learn: 9.9255542	test: 11.6357903	best: 11.6287537 (1776)	total: 55.9s	remaining: 1m 7s
    1804:	learn: 9.9236338	test: 11.6356399	best: 11.6287537 (1776)	total: 55.9s	remaining: 1m 7s
    1805:	learn: 9.9230997	test: 11.6345083	best: 11.6287537 (1776)	total: 55.9s	remaining: 1m 7s
    1806:	learn: 9.9224014	test: 11.6350732	best: 11.6287537 (1776)	total: 55.9s	remaining: 1m 7s
    1807:	learn: 9.9208870	test: 11.6353863	best: 11.6287537 (1776)	total: 56s	remaining: 1m 7s
    1808:	learn: 9.9200224	test: 11.6368940	best: 11.6287537 (1776)	total: 56s	remaining: 1m 7s
    1809:	learn: 9.9192479	test: 11.6361829	best: 11.6287537 (1776)	total: 56s	remaining: 1m 7s
    1810:	learn: 9.9178171	test: 11.6356230	best: 11.6287537 (1776)	total: 56s	remaining: 1m 7s
    1811:	learn: 9.9168272	test: 11.6353666	best: 11.6287537 (1776)	total: 56.1s	remaining: 1m 7s
    1812:	learn: 9.9146391	test: 11.6355638	best: 11.6287537 (1776)	total: 56.1s	remaining: 1m 7s
    1813:	learn: 9.9140844	test: 11.6342260	best: 11.6287537 (1776)	total: 56.1s	remaining: 1m 7s
    1814:	learn: 9.9138675	test: 11.6347809	best: 11.6287537 (1776)	total: 56.2s	remaining: 1m 7s
    1815:	learn: 9.9128602	test: 11.6352266	best: 11.6287537 (1776)	total: 56.2s	remaining: 1m 7s
    1816:	learn: 9.9113586	test: 11.6343636	best: 11.6287537 (1776)	total: 56.2s	remaining: 1m 7s
    1817:	learn: 9.9105486	test: 11.6336832	best: 11.6287537 (1776)	total: 56.2s	remaining: 1m 7s
    1818:	learn: 9.9099588	test: 11.6344652	best: 11.6287537 (1776)	total: 56.3s	remaining: 1m 7s
    1819:	learn: 9.9090534	test: 11.6340591	best: 11.6287537 (1776)	total: 56.3s	remaining: 1m 7s
    1820:	learn: 9.9083084	test: 11.6347878	best: 11.6287537 (1776)	total: 56.3s	remaining: 1m 7s
    1821:	learn: 9.9057288	test: 11.6355450	best: 11.6287537 (1776)	total: 56.4s	remaining: 1m 7s
    1822:	learn: 9.9036275	test: 11.6349482	best: 11.6287537 (1776)	total: 56.4s	remaining: 1m 7s
    1823:	learn: 9.9030030	test: 11.6343937	best: 11.6287537 (1776)	total: 56.4s	remaining: 1m 7s
    1824:	learn: 9.9016179	test: 11.6350083	best: 11.6287537 (1776)	total: 56.4s	remaining: 1m 7s
    1825:	learn: 9.9002883	test: 11.6349934	best: 11.6287537 (1776)	total: 56.5s	remaining: 1m 7s
    1826:	learn: 9.8991860	test: 11.6332545	best: 11.6287537 (1776)	total: 56.5s	remaining: 1m 7s
    1827:	learn: 9.8982544	test: 11.6311759	best: 11.6287537 (1776)	total: 56.5s	remaining: 1m 7s
    1828:	learn: 9.8967594	test: 11.6316370	best: 11.6287537 (1776)	total: 56.5s	remaining: 1m 7s
    1829:	learn: 9.8951675	test: 11.6313985	best: 11.6287537 (1776)	total: 56.6s	remaining: 1m 7s
    1830:	learn: 9.8938623	test: 11.6321345	best: 11.6287537 (1776)	total: 56.6s	remaining: 1m 7s
    1831:	learn: 9.8926717	test: 11.6327985	best: 11.6287537 (1776)	total: 56.6s	remaining: 1m 7s
    1832:	learn: 9.8921164	test: 11.6327211	best: 11.6287537 (1776)	total: 56.7s	remaining: 1m 6s
    1833:	learn: 9.8915943	test: 11.6329886	best: 11.6287537 (1776)	total: 56.7s	remaining: 1m 6s
    1834:	learn: 9.8908507	test: 11.6338973	best: 11.6287537 (1776)	total: 56.7s	remaining: 1m 6s
    1835:	learn: 9.8901046	test: 11.6336284	best: 11.6287537 (1776)	total: 56.7s	remaining: 1m 6s
    1836:	learn: 9.8896149	test: 11.6337037	best: 11.6287537 (1776)	total: 56.8s	remaining: 1m 6s
    1837:	learn: 9.8888127	test: 11.6333251	best: 11.6287537 (1776)	total: 56.8s	remaining: 1m 6s
    1838:	learn: 9.8876988	test: 11.6346840	best: 11.6287537 (1776)	total: 56.8s	remaining: 1m 6s
    1839:	learn: 9.8871031	test: 11.6351100	best: 11.6287537 (1776)	total: 56.8s	remaining: 1m 6s
    1840:	learn: 9.8862143	test: 11.6364302	best: 11.6287537 (1776)	total: 56.9s	remaining: 1m 6s
    1841:	learn: 9.8847552	test: 11.6360447	best: 11.6287537 (1776)	total: 56.9s	remaining: 1m 6s
    1842:	learn: 9.8839742	test: 11.6350025	best: 11.6287537 (1776)	total: 56.9s	remaining: 1m 6s
    1843:	learn: 9.8832756	test: 11.6339922	best: 11.6287537 (1776)	total: 57s	remaining: 1m 6s
    1844:	learn: 9.8822187	test: 11.6317498	best: 11.6287537 (1776)	total: 57s	remaining: 1m 6s
    1845:	learn: 9.8809807	test: 11.6304264	best: 11.6287537 (1776)	total: 57s	remaining: 1m 6s
    1846:	learn: 9.8794785	test: 11.6312680	best: 11.6287537 (1776)	total: 57s	remaining: 1m 6s
    1847:	learn: 9.8779339	test: 11.6307912	best: 11.6287537 (1776)	total: 57.1s	remaining: 1m 6s
    1848:	learn: 9.8753967	test: 11.6307256	best: 11.6287537 (1776)	total: 57.1s	remaining: 1m 6s
    1849:	learn: 9.8749413	test: 11.6323522	best: 11.6287537 (1776)	total: 57.1s	remaining: 1m 6s
    1850:	learn: 9.8741360	test: 11.6315120	best: 11.6287537 (1776)	total: 57.2s	remaining: 1m 6s
    1851:	learn: 9.8737761	test: 11.6301429	best: 11.6287537 (1776)	total: 57.2s	remaining: 1m 6s
    1852:	learn: 9.8710434	test: 11.6291260	best: 11.6287537 (1776)	total: 57.2s	remaining: 1m 6s
    1853:	learn: 9.8691814	test: 11.6298593	best: 11.6287537 (1776)	total: 57.2s	remaining: 1m 6s
    1854:	learn: 9.8683621	test: 11.6292348	best: 11.6287537 (1776)	total: 57.3s	remaining: 1m 6s
    1855:	learn: 9.8671408	test: 11.6269848	best: 11.6269848 (1855)	total: 57.3s	remaining: 1m 6s
    1856:	learn: 9.8664061	test: 11.6263452	best: 11.6263452 (1856)	total: 57.3s	remaining: 1m 6s
    1857:	learn: 9.8661854	test: 11.6275033	best: 11.6263452 (1856)	total: 57.4s	remaining: 1m 6s
    1858:	learn: 9.8636843	test: 11.6265562	best: 11.6263452 (1856)	total: 57.4s	remaining: 1m 6s
    1859:	learn: 9.8618521	test: 11.6237773	best: 11.6237773 (1859)	total: 57.4s	remaining: 1m 6s
    1860:	learn: 9.8598230	test: 11.6253184	best: 11.6237773 (1859)	total: 57.4s	remaining: 1m 6s
    1861:	learn: 9.8590509	test: 11.6230242	best: 11.6230242 (1861)	total: 57.5s	remaining: 1m 5s
    1862:	learn: 9.8588755	test: 11.6223853	best: 11.6223853 (1862)	total: 57.5s	remaining: 1m 5s
    1863:	learn: 9.8581819	test: 11.6223786	best: 11.6223786 (1863)	total: 57.5s	remaining: 1m 5s
    1864:	learn: 9.8584456	test: 11.6228773	best: 11.6223786 (1863)	total: 57.5s	remaining: 1m 5s
    1865:	learn: 9.8579036	test: 11.6238298	best: 11.6223786 (1863)	total: 57.6s	remaining: 1m 5s
    1866:	learn: 9.8560789	test: 11.6259436	best: 11.6223786 (1863)	total: 57.6s	remaining: 1m 5s
    1867:	learn: 9.8556917	test: 11.6242869	best: 11.6223786 (1863)	total: 57.6s	remaining: 1m 5s
    1868:	learn: 9.8532380	test: 11.6254460	best: 11.6223786 (1863)	total: 57.7s	remaining: 1m 5s
    1869:	learn: 9.8519034	test: 11.6265240	best: 11.6223786 (1863)	total: 57.7s	remaining: 1m 5s
    1870:	learn: 9.8515166	test: 11.6271710	best: 11.6223786 (1863)	total: 57.7s	remaining: 1m 5s
    1871:	learn: 9.8503893	test: 11.6265215	best: 11.6223786 (1863)	total: 57.7s	remaining: 1m 5s
    1872:	learn: 9.8499059	test: 11.6263294	best: 11.6223786 (1863)	total: 57.8s	remaining: 1m 5s
    1873:	learn: 9.8489771	test: 11.6269551	best: 11.6223786 (1863)	total: 57.8s	remaining: 1m 5s
    1874:	learn: 9.8488599	test: 11.6268161	best: 11.6223786 (1863)	total: 57.8s	remaining: 1m 5s
    1875:	learn: 9.8484697	test: 11.6270632	best: 11.6223786 (1863)	total: 57.8s	remaining: 1m 5s
    1876:	learn: 9.8477644	test: 11.6269271	best: 11.6223786 (1863)	total: 57.9s	remaining: 1m 5s
    1877:	learn: 9.8471942	test: 11.6279074	best: 11.6223786 (1863)	total: 57.9s	remaining: 1m 5s
    1878:	learn: 9.8459072	test: 11.6267734	best: 11.6223786 (1863)	total: 57.9s	remaining: 1m 5s
    1879:	learn: 9.8450451	test: 11.6283055	best: 11.6223786 (1863)	total: 58s	remaining: 1m 5s
    1880:	learn: 9.8425691	test: 11.6278090	best: 11.6223786 (1863)	total: 58s	remaining: 1m 5s
    1881:	learn: 9.8415058	test: 11.6277640	best: 11.6223786 (1863)	total: 58s	remaining: 1m 5s
    1882:	learn: 9.8405294	test: 11.6278018	best: 11.6223786 (1863)	total: 58s	remaining: 1m 5s
    1883:	learn: 9.8376085	test: 11.6280261	best: 11.6223786 (1863)	total: 58.1s	remaining: 1m 5s
    1884:	learn: 9.8366576	test: 11.6295621	best: 11.6223786 (1863)	total: 58.1s	remaining: 1m 5s
    1885:	learn: 9.8355130	test: 11.6292508	best: 11.6223786 (1863)	total: 58.1s	remaining: 1m 5s
    1886:	learn: 9.8349040	test: 11.6287922	best: 11.6223786 (1863)	total: 58.2s	remaining: 1m 5s
    1887:	learn: 9.8342958	test: 11.6296966	best: 11.6223786 (1863)	total: 58.2s	remaining: 1m 5s
    1888:	learn: 9.8326597	test: 11.6297085	best: 11.6223786 (1863)	total: 58.2s	remaining: 1m 5s
    1889:	learn: 9.8313811	test: 11.6314910	best: 11.6223786 (1863)	total: 58.2s	remaining: 1m 5s
    1890:	learn: 9.8302226	test: 11.6320992	best: 11.6223786 (1863)	total: 58.3s	remaining: 1m 4s
    1891:	learn: 9.8285748	test: 11.6300664	best: 11.6223786 (1863)	total: 58.3s	remaining: 1m 4s
    1892:	learn: 9.8279784	test: 11.6313056	best: 11.6223786 (1863)	total: 58.3s	remaining: 1m 4s
    1893:	learn: 9.8265732	test: 11.6319307	best: 11.6223786 (1863)	total: 58.3s	remaining: 1m 4s
    1894:	learn: 9.8255754	test: 11.6322956	best: 11.6223786 (1863)	total: 58.4s	remaining: 1m 4s
    1895:	learn: 9.8240791	test: 11.6300999	best: 11.6223786 (1863)	total: 58.4s	remaining: 1m 4s
    1896:	learn: 9.8233689	test: 11.6299671	best: 11.6223786 (1863)	total: 58.4s	remaining: 1m 4s
    1897:	learn: 9.8225194	test: 11.6311859	best: 11.6223786 (1863)	total: 58.4s	remaining: 1m 4s
    1898:	learn: 9.8223256	test: 11.6289445	best: 11.6223786 (1863)	total: 58.5s	remaining: 1m 4s
    1899:	learn: 9.8202252	test: 11.6290760	best: 11.6223786 (1863)	total: 58.5s	remaining: 1m 4s
    1900:	learn: 9.8190880	test: 11.6273563	best: 11.6223786 (1863)	total: 58.5s	remaining: 1m 4s
    1901:	learn: 9.8184348	test: 11.6268442	best: 11.6223786 (1863)	total: 58.6s	remaining: 1m 4s
    1902:	learn: 9.8179642	test: 11.6273499	best: 11.6223786 (1863)	total: 58.6s	remaining: 1m 4s
    1903:	learn: 9.8173185	test: 11.6273577	best: 11.6223786 (1863)	total: 58.6s	remaining: 1m 4s
    1904:	learn: 9.8156676	test: 11.6269586	best: 11.6223786 (1863)	total: 58.6s	remaining: 1m 4s
    1905:	learn: 9.8149884	test: 11.6272647	best: 11.6223786 (1863)	total: 58.7s	remaining: 1m 4s
    1906:	learn: 9.8133772	test: 11.6298475	best: 11.6223786 (1863)	total: 58.7s	remaining: 1m 4s
    1907:	learn: 9.8116384	test: 11.6299127	best: 11.6223786 (1863)	total: 58.7s	remaining: 1m 4s
    1908:	learn: 9.8106210	test: 11.6293583	best: 11.6223786 (1863)	total: 58.8s	remaining: 1m 4s
    1909:	learn: 9.8101943	test: 11.6307701	best: 11.6223786 (1863)	total: 58.8s	remaining: 1m 4s
    1910:	learn: 9.8099017	test: 11.6310818	best: 11.6223786 (1863)	total: 58.8s	remaining: 1m 4s
    1911:	learn: 9.8091629	test: 11.6318679	best: 11.6223786 (1863)	total: 58.8s	remaining: 1m 4s
    1912:	learn: 9.8082735	test: 11.6321437	best: 11.6223786 (1863)	total: 58.9s	remaining: 1m 4s
    1913:	learn: 9.8065511	test: 11.6310082	best: 11.6223786 (1863)	total: 58.9s	remaining: 1m 4s
    1914:	learn: 9.8060797	test: 11.6319113	best: 11.6223786 (1863)	total: 58.9s	remaining: 1m 4s
    1915:	learn: 9.8042923	test: 11.6325801	best: 11.6223786 (1863)	total: 58.9s	remaining: 1m 4s
    1916:	learn: 9.8034938	test: 11.6325842	best: 11.6223786 (1863)	total: 59s	remaining: 1m 4s
    1917:	learn: 9.8031241	test: 11.6318723	best: 11.6223786 (1863)	total: 59s	remaining: 1m 4s
    1918:	learn: 9.8027898	test: 11.6305135	best: 11.6223786 (1863)	total: 59s	remaining: 1m 4s
    1919:	learn: 9.8023285	test: 11.6293750	best: 11.6223786 (1863)	total: 59.1s	remaining: 1m 3s
    1920:	learn: 9.8014633	test: 11.6266409	best: 11.6223786 (1863)	total: 59.1s	remaining: 1m 3s
    1921:	learn: 9.8010304	test: 11.6273358	best: 11.6223786 (1863)	total: 59.1s	remaining: 1m 3s
    1922:	learn: 9.8001104	test: 11.6279023	best: 11.6223786 (1863)	total: 59.1s	remaining: 1m 3s
    1923:	learn: 9.7960757	test: 11.6281338	best: 11.6223786 (1863)	total: 59.2s	remaining: 1m 3s
    1924:	learn: 9.7942585	test: 11.6274695	best: 11.6223786 (1863)	total: 59.2s	remaining: 1m 3s
    1925:	learn: 9.7929113	test: 11.6297516	best: 11.6223786 (1863)	total: 59.2s	remaining: 1m 3s
    1926:	learn: 9.7922962	test: 11.6306367	best: 11.6223786 (1863)	total: 59.3s	remaining: 1m 3s
    1927:	learn: 9.7915691	test: 11.6298738	best: 11.6223786 (1863)	total: 59.3s	remaining: 1m 3s
    1928:	learn: 9.7891853	test: 11.6278378	best: 11.6223786 (1863)	total: 59.3s	remaining: 1m 3s
    1929:	learn: 9.7885324	test: 11.6287017	best: 11.6223786 (1863)	total: 59.3s	remaining: 1m 3s
    1930:	learn: 9.7876795	test: 11.6293569	best: 11.6223786 (1863)	total: 59.4s	remaining: 1m 3s
    1931:	learn: 9.7874471	test: 11.6300257	best: 11.6223786 (1863)	total: 59.4s	remaining: 1m 3s
    1932:	learn: 9.7860052	test: 11.6301256	best: 11.6223786 (1863)	total: 59.4s	remaining: 1m 3s
    1933:	learn: 9.7840005	test: 11.6310970	best: 11.6223786 (1863)	total: 59.4s	remaining: 1m 3s
    1934:	learn: 9.7830687	test: 11.6316278	best: 11.6223786 (1863)	total: 59.5s	remaining: 1m 3s
    1935:	learn: 9.7806195	test: 11.6310455	best: 11.6223786 (1863)	total: 59.5s	remaining: 1m 3s
    1936:	learn: 9.7796669	test: 11.6312575	best: 11.6223786 (1863)	total: 59.5s	remaining: 1m 3s
    1937:	learn: 9.7791098	test: 11.6306161	best: 11.6223786 (1863)	total: 59.5s	remaining: 1m 3s
    1938:	learn: 9.7789654	test: 11.6311896	best: 11.6223786 (1863)	total: 59.6s	remaining: 1m 3s
    1939:	learn: 9.7769353	test: 11.6300189	best: 11.6223786 (1863)	total: 59.6s	remaining: 1m 3s
    1940:	learn: 9.7748940	test: 11.6324006	best: 11.6223786 (1863)	total: 59.6s	remaining: 1m 3s
    1941:	learn: 9.7742330	test: 11.6332215	best: 11.6223786 (1863)	total: 59.7s	remaining: 1m 3s
    1942:	learn: 9.7728163	test: 11.6335424	best: 11.6223786 (1863)	total: 59.7s	remaining: 1m 3s
    1943:	learn: 9.7718217	test: 11.6330215	best: 11.6223786 (1863)	total: 59.7s	remaining: 1m 3s
    1944:	learn: 9.7712373	test: 11.6344979	best: 11.6223786 (1863)	total: 59.7s	remaining: 1m 3s
    1945:	learn: 9.7706197	test: 11.6343318	best: 11.6223786 (1863)	total: 59.8s	remaining: 1m 3s
    1946:	learn: 9.7691038	test: 11.6355824	best: 11.6223786 (1863)	total: 59.8s	remaining: 1m 3s
    1947:	learn: 9.7645265	test: 11.6319951	best: 11.6223786 (1863)	total: 59.8s	remaining: 1m 3s
    1948:	learn: 9.7631387	test: 11.6360539	best: 11.6223786 (1863)	total: 59.8s	remaining: 1m 2s
    1949:	learn: 9.7621788	test: 11.6343354	best: 11.6223786 (1863)	total: 59.9s	remaining: 1m 2s
    1950:	learn: 9.7611828	test: 11.6336207	best: 11.6223786 (1863)	total: 59.9s	remaining: 1m 2s
    1951:	learn: 9.7604547	test: 11.6329558	best: 11.6223786 (1863)	total: 59.9s	remaining: 1m 2s
    1952:	learn: 9.7600852	test: 11.6323180	best: 11.6223786 (1863)	total: 60s	remaining: 1m 2s
    1953:	learn: 9.7583951	test: 11.6312958	best: 11.6223786 (1863)	total: 60s	remaining: 1m 2s
    1954:	learn: 9.7573856	test: 11.6308368	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1955:	learn: 9.7558614	test: 11.6289391	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1956:	learn: 9.7552861	test: 11.6271142	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1957:	learn: 9.7520234	test: 11.6274626	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1958:	learn: 9.7510254	test: 11.6265261	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1959:	learn: 9.7502411	test: 11.6264731	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1960:	learn: 9.7479763	test: 11.6259425	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1961:	learn: 9.7455141	test: 11.6254051	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1962:	learn: 9.7428910	test: 11.6261257	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1963:	learn: 9.7407553	test: 11.6252077	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1964:	learn: 9.7386083	test: 11.6224426	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1965:	learn: 9.7373072	test: 11.6231860	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1966:	learn: 9.7366913	test: 11.6235461	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1967:	learn: 9.7353114	test: 11.6231698	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1968:	learn: 9.7345140	test: 11.6238924	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1969:	learn: 9.7340338	test: 11.6247371	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1970:	learn: 9.7329522	test: 11.6241615	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1971:	learn: 9.7319798	test: 11.6257089	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1972:	learn: 9.7315328	test: 11.6247297	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1973:	learn: 9.7281467	test: 11.6230606	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1974:	learn: 9.7261399	test: 11.6237577	best: 11.6223786 (1863)	total: 1m	remaining: 1m 2s
    1975:	learn: 9.7243407	test: 11.6222675	best: 11.6222675 (1975)	total: 1m	remaining: 1m 2s
    1976:	learn: 9.7237385	test: 11.6228125	best: 11.6222675 (1975)	total: 1m	remaining: 1m 2s
    1977:	learn: 9.7221480	test: 11.6212688	best: 11.6212688 (1977)	total: 1m	remaining: 1m 2s
    1978:	learn: 9.7201979	test: 11.6201228	best: 11.6201228 (1978)	total: 1m	remaining: 1m 1s
    1979:	learn: 9.7195568	test: 11.6195691	best: 11.6195691 (1979)	total: 1m	remaining: 1m 1s
    1980:	learn: 9.7183812	test: 11.6219612	best: 11.6195691 (1979)	total: 1m	remaining: 1m 1s
    1981:	learn: 9.7175706	test: 11.6220849	best: 11.6195691 (1979)	total: 1m	remaining: 1m 1s
    1982:	learn: 9.7152884	test: 11.6213521	best: 11.6195691 (1979)	total: 1m	remaining: 1m 1s
    1983:	learn: 9.7148053	test: 11.6210736	best: 11.6195691 (1979)	total: 1m	remaining: 1m 1s
    1984:	learn: 9.7127431	test: 11.6221401	best: 11.6195691 (1979)	total: 1m	remaining: 1m 1s
    1985:	learn: 9.7111046	test: 11.6225318	best: 11.6195691 (1979)	total: 1m	remaining: 1m 1s
    1986:	learn: 9.7110539	test: 11.6227256	best: 11.6195691 (1979)	total: 1m	remaining: 1m 1s
    1987:	learn: 9.7094527	test: 11.6228524	best: 11.6195691 (1979)	total: 1m	remaining: 1m 1s
    1988:	learn: 9.7086808	test: 11.6236871	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1989:	learn: 9.7068895	test: 11.6230604	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1990:	learn: 9.7057354	test: 11.6260163	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1991:	learn: 9.7048718	test: 11.6259508	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1992:	learn: 9.7040718	test: 11.6269993	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1993:	learn: 9.7033427	test: 11.6256766	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1994:	learn: 9.7029664	test: 11.6254736	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1995:	learn: 9.7008792	test: 11.6253120	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1996:	learn: 9.7005509	test: 11.6244921	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1997:	learn: 9.6990296	test: 11.6247793	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1998:	learn: 9.6978330	test: 11.6256899	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    1999:	learn: 9.6968613	test: 11.6266753	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2000:	learn: 9.6964566	test: 11.6247706	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2001:	learn: 9.6958067	test: 11.6246940	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2002:	learn: 9.6952256	test: 11.6240192	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2003:	learn: 9.6933482	test: 11.6252374	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2004:	learn: 9.6915232	test: 11.6238486	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2005:	learn: 9.6908422	test: 11.6242772	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2006:	learn: 9.6897793	test: 11.6237606	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2007:	learn: 9.6882965	test: 11.6241525	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2008:	learn: 9.6866465	test: 11.6233317	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2009:	learn: 9.6847233	test: 11.6245106	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2010:	learn: 9.6840795	test: 11.6253043	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2011:	learn: 9.6829915	test: 11.6265656	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2012:	learn: 9.6823301	test: 11.6268227	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m 1s
    2013:	learn: 9.6813843	test: 11.6274882	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m
    2014:	learn: 9.6782503	test: 11.6239496	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m
    2015:	learn: 9.6780819	test: 11.6237016	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m
    2016:	learn: 9.6753545	test: 11.6232970	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m
    2017:	learn: 9.6744698	test: 11.6235921	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m
    2018:	learn: 9.6741333	test: 11.6241286	best: 11.6195691 (1979)	total: 1m 1s	remaining: 1m
    2019:	learn: 9.6740035	test: 11.6233804	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2020:	learn: 9.6735531	test: 11.6238468	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2021:	learn: 9.6722040	test: 11.6234036	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2022:	learn: 9.6698732	test: 11.6247915	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2023:	learn: 9.6675825	test: 11.6233733	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2024:	learn: 9.6671460	test: 11.6244406	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2025:	learn: 9.6667723	test: 11.6240364	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2026:	learn: 9.6661162	test: 11.6242405	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2027:	learn: 9.6646913	test: 11.6248923	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2028:	learn: 9.6634638	test: 11.6244646	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2029:	learn: 9.6599424	test: 11.6240443	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2030:	learn: 9.6580758	test: 11.6247120	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2031:	learn: 9.6561414	test: 11.6206435	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2032:	learn: 9.6555984	test: 11.6213829	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2033:	learn: 9.6551654	test: 11.6215396	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2034:	learn: 9.6544667	test: 11.6209369	best: 11.6195691 (1979)	total: 1m 2s	remaining: 1m
    2035:	learn: 9.6530201	test: 11.6193975	best: 11.6193975 (2035)	total: 1m 2s	remaining: 1m
    2036:	learn: 9.6512811	test: 11.6180592	best: 11.6180592 (2036)	total: 1m 2s	remaining: 1m
    2037:	learn: 9.6501874	test: 11.6183815	best: 11.6180592 (2036)	total: 1m 2s	remaining: 1m
    2038:	learn: 9.6478439	test: 11.6156236	best: 11.6156236 (2038)	total: 1m 2s	remaining: 1m
    2039:	learn: 9.6464442	test: 11.6156559	best: 11.6156236 (2038)	total: 1m 2s	remaining: 1m
    2040:	learn: 9.6440415	test: 11.6137565	best: 11.6137565 (2040)	total: 1m 2s	remaining: 1m
    2041:	learn: 9.6433701	test: 11.6146961	best: 11.6137565 (2040)	total: 1m 2s	remaining: 1m
    2042:	learn: 9.6431193	test: 11.6147784	best: 11.6137565 (2040)	total: 1m 2s	remaining: 1m
    2043:	learn: 9.6416501	test: 11.6154494	best: 11.6137565 (2040)	total: 1m 2s	remaining: 1m
    2044:	learn: 9.6406349	test: 11.6153726	best: 11.6137565 (2040)	total: 1m 2s	remaining: 1m
    2045:	learn: 9.6389287	test: 11.6153386	best: 11.6137565 (2040)	total: 1m 2s	remaining: 1m
    2046:	learn: 9.6381859	test: 11.6137051	best: 11.6137051 (2046)	total: 1m 2s	remaining: 1m
    2047:	learn: 9.6357064	test: 11.6127102	best: 11.6127102 (2047)	total: 1m 2s	remaining: 1m
    2048:	learn: 9.6344899	test: 11.6155449	best: 11.6127102 (2047)	total: 1m 3s	remaining: 60s
    2049:	learn: 9.6330831	test: 11.6135488	best: 11.6127102 (2047)	total: 1m 3s	remaining: 60s
    2050:	learn: 9.6320259	test: 11.6130988	best: 11.6127102 (2047)	total: 1m 3s	remaining: 59.9s
    2051:	learn: 9.6311122	test: 11.6140779	best: 11.6127102 (2047)	total: 1m 3s	remaining: 59.9s
    2052:	learn: 9.6299690	test: 11.6136813	best: 11.6127102 (2047)	total: 1m 3s	remaining: 59.9s
    2053:	learn: 9.6294613	test: 11.6132689	best: 11.6127102 (2047)	total: 1m 3s	remaining: 59.9s
    2054:	learn: 9.6292092	test: 11.6122310	best: 11.6122310 (2054)	total: 1m 3s	remaining: 59.8s
    2055:	learn: 9.6262951	test: 11.6127152	best: 11.6122310 (2054)	total: 1m 3s	remaining: 59.8s
    2056:	learn: 9.6243368	test: 11.6131753	best: 11.6122310 (2054)	total: 1m 3s	remaining: 59.8s
    2057:	learn: 9.6232232	test: 11.6121185	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.7s
    2058:	learn: 9.6227202	test: 11.6135036	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.7s
    2059:	learn: 9.6203556	test: 11.6156323	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.7s
    2060:	learn: 9.6202848	test: 11.6149909	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.7s
    2061:	learn: 9.6185475	test: 11.6137376	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.6s
    2062:	learn: 9.6165930	test: 11.6127241	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.6s
    2063:	learn: 9.6145665	test: 11.6145564	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.6s
    2064:	learn: 9.6137801	test: 11.6150570	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.5s
    2065:	learn: 9.6131505	test: 11.6144781	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.5s
    2066:	learn: 9.6117121	test: 11.6143835	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.5s
    2067:	learn: 9.6099293	test: 11.6136413	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.5s
    2068:	learn: 9.6063359	test: 11.6142824	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.4s
    2069:	learn: 9.6056687	test: 11.6136709	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.4s
    2070:	learn: 9.6050890	test: 11.6135908	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.4s
    2071:	learn: 9.6045462	test: 11.6155130	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.3s
    2072:	learn: 9.6034460	test: 11.6169500	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.3s
    2073:	learn: 9.6023980	test: 11.6161167	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.3s
    2074:	learn: 9.6011875	test: 11.6171850	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.2s
    2075:	learn: 9.5997228	test: 11.6170715	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.2s
    2076:	learn: 9.5980338	test: 11.6196146	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.2s
    2077:	learn: 9.5969628	test: 11.6184372	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.2s
    2078:	learn: 9.5959852	test: 11.6184250	best: 11.6121185 (2057)	total: 1m 3s	remaining: 59.1s
    2079:	learn: 9.5948926	test: 11.6176402	best: 11.6121185 (2057)	total: 1m 4s	remaining: 59.1s
    2080:	learn: 9.5929347	test: 11.6196486	best: 11.6121185 (2057)	total: 1m 4s	remaining: 59.1s
    2081:	learn: 9.5922998	test: 11.6200969	best: 11.6121185 (2057)	total: 1m 4s	remaining: 59s
    2082:	learn: 9.5898335	test: 11.6168921	best: 11.6121185 (2057)	total: 1m 4s	remaining: 59s
    2083:	learn: 9.5887493	test: 11.6182279	best: 11.6121185 (2057)	total: 1m 4s	remaining: 59s
    2084:	learn: 9.5876637	test: 11.6170749	best: 11.6121185 (2057)	total: 1m 4s	remaining: 58.9s
    2085:	learn: 9.5870605	test: 11.6167416	best: 11.6121185 (2057)	total: 1m 4s	remaining: 58.9s
    2086:	learn: 9.5865100	test: 11.6148781	best: 11.6121185 (2057)	total: 1m 4s	remaining: 58.9s
    2087:	learn: 9.5843134	test: 11.6131953	best: 11.6121185 (2057)	total: 1m 4s	remaining: 58.9s
    2088:	learn: 9.5834244	test: 11.6119908	best: 11.6119908 (2088)	total: 1m 4s	remaining: 58.8s
    2089:	learn: 9.5816335	test: 11.6136775	best: 11.6119908 (2088)	total: 1m 4s	remaining: 58.8s
    2090:	learn: 9.5804730	test: 11.6135589	best: 11.6119908 (2088)	total: 1m 4s	remaining: 58.8s
    2091:	learn: 9.5798449	test: 11.6149033	best: 11.6119908 (2088)	total: 1m 4s	remaining: 58.7s
    2092:	learn: 9.5780225	test: 11.6136407	best: 11.6119908 (2088)	total: 1m 4s	remaining: 58.7s
    2093:	learn: 9.5770337	test: 11.6137974	best: 11.6119908 (2088)	total: 1m 4s	remaining: 58.7s
    2094:	learn: 9.5761030	test: 11.6136331	best: 11.6119908 (2088)	total: 1m 4s	remaining: 58.6s
    2095:	learn: 9.5733553	test: 11.6116969	best: 11.6116969 (2095)	total: 1m 4s	remaining: 58.6s
    2096:	learn: 9.5729963	test: 11.6128196	best: 11.6116969 (2095)	total: 1m 4s	remaining: 58.6s
    2097:	learn: 9.5722786	test: 11.6125718	best: 11.6116969 (2095)	total: 1m 4s	remaining: 58.6s
    2098:	learn: 9.5715070	test: 11.6126463	best: 11.6116969 (2095)	total: 1m 4s	remaining: 58.5s
    2099:	learn: 9.5704844	test: 11.6116110	best: 11.6116110 (2099)	total: 1m 4s	remaining: 58.5s
    2100:	learn: 9.5696774	test: 11.6100348	best: 11.6100348 (2100)	total: 1m 4s	remaining: 58.5s
    2101:	learn: 9.5688333	test: 11.6102143	best: 11.6100348 (2100)	total: 1m 4s	remaining: 58.4s
    2102:	learn: 9.5681329	test: 11.6117893	best: 11.6100348 (2100)	total: 1m 4s	remaining: 58.4s
    2103:	learn: 9.5666812	test: 11.6143411	best: 11.6100348 (2100)	total: 1m 4s	remaining: 58.4s
    2104:	learn: 9.5656231	test: 11.6132225	best: 11.6100348 (2100)	total: 1m 4s	remaining: 58.3s
    2105:	learn: 9.5653914	test: 11.6123827	best: 11.6100348 (2100)	total: 1m 4s	remaining: 58.3s
    2106:	learn: 9.5632218	test: 11.6100637	best: 11.6100348 (2100)	total: 1m 4s	remaining: 58.3s
    2107:	learn: 9.5626477	test: 11.6108691	best: 11.6100348 (2100)	total: 1m 4s	remaining: 58.3s
    2108:	learn: 9.5613643	test: 11.6096905	best: 11.6096905 (2108)	total: 1m 4s	remaining: 58.2s
    2109:	learn: 9.5612147	test: 11.6099688	best: 11.6096905 (2108)	total: 1m 4s	remaining: 58.2s
    2110:	learn: 9.5608300	test: 11.6112652	best: 11.6096905 (2108)	total: 1m 5s	remaining: 58.2s
    2111:	learn: 9.5604737	test: 11.6106110	best: 11.6096905 (2108)	total: 1m 5s	remaining: 58.1s
    2112:	learn: 9.5596704	test: 11.6111252	best: 11.6096905 (2108)	total: 1m 5s	remaining: 58.1s
    2113:	learn: 9.5576727	test: 11.6121605	best: 11.6096905 (2108)	total: 1m 5s	remaining: 58.1s
    2114:	learn: 9.5561133	test: 11.6102958	best: 11.6096905 (2108)	total: 1m 5s	remaining: 58.1s
    2115:	learn: 9.5543365	test: 11.6108569	best: 11.6096905 (2108)	total: 1m 5s	remaining: 58s
    2116:	learn: 9.5541061	test: 11.6109588	best: 11.6096905 (2108)	total: 1m 5s	remaining: 58s
    2117:	learn: 9.5534640	test: 11.6112967	best: 11.6096905 (2108)	total: 1m 5s	remaining: 58s
    2118:	learn: 9.5533796	test: 11.6111055	best: 11.6096905 (2108)	total: 1m 5s	remaining: 57.9s
    2119:	learn: 9.5521259	test: 11.6100214	best: 11.6096905 (2108)	total: 1m 5s	remaining: 57.9s
    2120:	learn: 9.5514140	test: 11.6080193	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.9s
    2121:	learn: 9.5503896	test: 11.6095093	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.8s
    2122:	learn: 9.5498350	test: 11.6085108	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.8s
    2123:	learn: 9.5495298	test: 11.6092027	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.8s
    2124:	learn: 9.5477732	test: 11.6102423	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.8s
    2125:	learn: 9.5469119	test: 11.6095924	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.7s
    2126:	learn: 9.5465153	test: 11.6106604	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.7s
    2127:	learn: 9.5443215	test: 11.6123200	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.7s
    2128:	learn: 9.5440863	test: 11.6116492	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.6s
    2129:	learn: 9.5417599	test: 11.6129129	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.6s
    2130:	learn: 9.5415590	test: 11.6127184	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.6s
    2131:	learn: 9.5414703	test: 11.6129194	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.5s
    2132:	learn: 9.5405661	test: 11.6134463	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.5s
    2133:	learn: 9.5386035	test: 11.6126126	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.5s
    2134:	learn: 9.5363003	test: 11.6107173	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.4s
    2135:	learn: 9.5343061	test: 11.6114979	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.4s
    2136:	learn: 9.5333659	test: 11.6115798	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.4s
    2137:	learn: 9.5323891	test: 11.6110484	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.3s
    2138:	learn: 9.5309207	test: 11.6110891	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.3s
    2139:	learn: 9.5301543	test: 11.6121008	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.3s
    2140:	learn: 9.5285238	test: 11.6111724	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.2s
    2141:	learn: 9.5282369	test: 11.6095300	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.2s
    2142:	learn: 9.5250198	test: 11.6096260	best: 11.6080193 (2120)	total: 1m 5s	remaining: 57.2s
    2143:	learn: 9.5240458	test: 11.6105626	best: 11.6080193 (2120)	total: 1m 6s	remaining: 57.1s
    2144:	learn: 9.5232850	test: 11.6110048	best: 11.6080193 (2120)	total: 1m 6s	remaining: 57.1s
    2145:	learn: 9.5216683	test: 11.6095580	best: 11.6080193 (2120)	total: 1m 6s	remaining: 57.1s
    2146:	learn: 9.5203631	test: 11.6086679	best: 11.6080193 (2120)	total: 1m 6s	remaining: 57s
    2147:	learn: 9.5193840	test: 11.6067963	best: 11.6067963 (2147)	total: 1m 6s	remaining: 57s
    2148:	learn: 9.5182718	test: 11.6071592	best: 11.6067963 (2147)	total: 1m 6s	remaining: 57s
    2149:	learn: 9.5177575	test: 11.6068860	best: 11.6067963 (2147)	total: 1m 6s	remaining: 56.9s
    2150:	learn: 9.5159449	test: 11.6068352	best: 11.6067963 (2147)	total: 1m 6s	remaining: 56.9s
    2151:	learn: 9.5139102	test: 11.6062184	best: 11.6062184 (2151)	total: 1m 6s	remaining: 56.9s
    2152:	learn: 9.5123768	test: 11.6043464	best: 11.6043464 (2152)	total: 1m 6s	remaining: 56.8s
    2153:	learn: 9.5105425	test: 11.6036488	best: 11.6036488 (2153)	total: 1m 6s	remaining: 56.8s
    2154:	learn: 9.5091596	test: 11.6039526	best: 11.6036488 (2153)	total: 1m 6s	remaining: 56.8s
    2155:	learn: 9.5086741	test: 11.6029722	best: 11.6029722 (2155)	total: 1m 6s	remaining: 56.7s
    2156:	learn: 9.5081648	test: 11.6026273	best: 11.6026273 (2156)	total: 1m 6s	remaining: 56.7s
    2157:	learn: 9.5074488	test: 11.6029082	best: 11.6026273 (2156)	total: 1m 6s	remaining: 56.7s
    2158:	learn: 9.5066085	test: 11.6026255	best: 11.6026255 (2158)	total: 1m 6s	remaining: 56.6s
    2159:	learn: 9.5060573	test: 11.6024944	best: 11.6024944 (2159)	total: 1m 6s	remaining: 56.6s
    2160:	learn: 9.5044537	test: 11.6019269	best: 11.6019269 (2160)	total: 1m 6s	remaining: 56.6s
    2161:	learn: 9.5021334	test: 11.6013061	best: 11.6013061 (2161)	total: 1m 6s	remaining: 56.5s
    2162:	learn: 9.5002655	test: 11.6016813	best: 11.6013061 (2161)	total: 1m 6s	remaining: 56.5s
    2163:	learn: 9.4984593	test: 11.6003463	best: 11.6003463 (2163)	total: 1m 6s	remaining: 56.5s
    2164:	learn: 9.4981358	test: 11.6010216	best: 11.6003463 (2163)	total: 1m 6s	remaining: 56.4s
    2165:	learn: 9.4968941	test: 11.6011135	best: 11.6003463 (2163)	total: 1m 6s	remaining: 56.4s
    2166:	learn: 9.4967335	test: 11.6011182	best: 11.6003463 (2163)	total: 1m 6s	remaining: 56.4s
    2167:	learn: 9.4947909	test: 11.6024350	best: 11.6003463 (2163)	total: 1m 6s	remaining: 56.3s
    2168:	learn: 9.4940866	test: 11.6014594	best: 11.6003463 (2163)	total: 1m 6s	remaining: 56.3s
    2169:	learn: 9.4934289	test: 11.6008727	best: 11.6003463 (2163)	total: 1m 6s	remaining: 56.3s
    2170:	learn: 9.4930410	test: 11.5997028	best: 11.5997028 (2170)	total: 1m 6s	remaining: 56.2s
    2171:	learn: 9.4918224	test: 11.5994715	best: 11.5994715 (2171)	total: 1m 6s	remaining: 56.2s
    2172:	learn: 9.4914068	test: 11.6003951	best: 11.5994715 (2171)	total: 1m 6s	remaining: 56.2s
    2173:	learn: 9.4899713	test: 11.6029129	best: 11.5994715 (2171)	total: 1m 6s	remaining: 56.1s
    2174:	learn: 9.4892195	test: 11.6027470	best: 11.5994715 (2171)	total: 1m 6s	remaining: 56.1s
    2175:	learn: 9.4882633	test: 11.6015654	best: 11.5994715 (2171)	total: 1m 6s	remaining: 56.1s
    2176:	learn: 9.4872114	test: 11.6028514	best: 11.5994715 (2171)	total: 1m 6s	remaining: 56s
    2177:	learn: 9.4866431	test: 11.6033978	best: 11.5994715 (2171)	total: 1m 6s	remaining: 56s
    2178:	learn: 9.4864643	test: 11.6035918	best: 11.5994715 (2171)	total: 1m 6s	remaining: 56s
    2179:	learn: 9.4856281	test: 11.6047387	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.9s
    2180:	learn: 9.4833715	test: 11.6052049	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.9s
    2181:	learn: 9.4829891	test: 11.6058029	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.9s
    2182:	learn: 9.4818932	test: 11.6066155	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.9s
    2183:	learn: 9.4811106	test: 11.6051420	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.8s
    2184:	learn: 9.4803870	test: 11.6071470	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.8s
    2185:	learn: 9.4791031	test: 11.6069441	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.8s
    2186:	learn: 9.4787936	test: 11.6049857	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.7s
    2187:	learn: 9.4780942	test: 11.6056902	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.7s
    2188:	learn: 9.4769212	test: 11.6043283	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.7s
    2189:	learn: 9.4760623	test: 11.6039140	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.7s
    2190:	learn: 9.4753398	test: 11.6026350	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.6s
    2191:	learn: 9.4743520	test: 11.6021322	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.6s
    2192:	learn: 9.4738610	test: 11.6030693	best: 11.5994715 (2171)	total: 1m 7s	remaining: 55.6s
    2193:	learn: 9.4698927	test: 11.5977314	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.5s
    2194:	learn: 9.4696320	test: 11.5991883	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.5s
    2195:	learn: 9.4692295	test: 11.5998249	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.5s
    2196:	learn: 9.4678008	test: 11.5996271	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.5s
    2197:	learn: 9.4664417	test: 11.6008965	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.4s
    2198:	learn: 9.4651395	test: 11.6013160	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.4s
    2199:	learn: 9.4633568	test: 11.5995578	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.4s
    2200:	learn: 9.4617425	test: 11.5980409	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.3s
    2201:	learn: 9.4604886	test: 11.5998207	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.3s
    2202:	learn: 9.4589175	test: 11.6010281	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.3s
    2203:	learn: 9.4585690	test: 11.6006035	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.3s
    2204:	learn: 9.4581867	test: 11.6010956	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.2s
    2205:	learn: 9.4561193	test: 11.6003207	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.2s
    2206:	learn: 9.4548208	test: 11.5994506	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.2s
    2207:	learn: 9.4538147	test: 11.5999155	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.1s
    2208:	learn: 9.4531644	test: 11.6013349	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.1s
    2209:	learn: 9.4505163	test: 11.6035815	best: 11.5977314 (2193)	total: 1m 7s	remaining: 55.1s
    2210:	learn: 9.4494563	test: 11.6043740	best: 11.5977314 (2193)	total: 1m 8s	remaining: 55s
    2211:	learn: 9.4490399	test: 11.6048972	best: 11.5977314 (2193)	total: 1m 8s	remaining: 55s
    2212:	learn: 9.4484060	test: 11.6046541	best: 11.5977314 (2193)	total: 1m 8s	remaining: 55s
    2213:	learn: 9.4476691	test: 11.6050018	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.9s
    2214:	learn: 9.4457816	test: 11.6036052	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.9s
    2215:	learn: 9.4429343	test: 11.6023494	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.9s
    2216:	learn: 9.4424224	test: 11.6029951	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.8s
    2217:	learn: 9.4411257	test: 11.6011406	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.8s
    2218:	learn: 9.4395440	test: 11.6007156	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.8s
    2219:	learn: 9.4380859	test: 11.6007452	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.7s
    2220:	learn: 9.4371785	test: 11.6010164	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.7s
    2221:	learn: 9.4362157	test: 11.6010021	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.7s
    2222:	learn: 9.4351243	test: 11.6001200	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.6s
    2223:	learn: 9.4343345	test: 11.5989736	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.6s
    2224:	learn: 9.4323556	test: 11.6000867	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.6s
    2225:	learn: 9.4315166	test: 11.5984287	best: 11.5977314 (2193)	total: 1m 8s	remaining: 54.6s
    2226:	learn: 9.4312160	test: 11.5977300	best: 11.5977300 (2226)	total: 1m 8s	remaining: 54.5s
    2227:	learn: 9.4308672	test: 11.5965769	best: 11.5965769 (2227)	total: 1m 8s	remaining: 54.5s
    2228:	learn: 9.4279624	test: 11.5938703	best: 11.5938703 (2228)	total: 1m 8s	remaining: 54.5s
    2229:	learn: 9.4278557	test: 11.5948047	best: 11.5938703 (2228)	total: 1m 8s	remaining: 54.5s
    2230:	learn: 9.4264715	test: 11.5953918	best: 11.5938703 (2228)	total: 1m 8s	remaining: 54.5s
    2231:	learn: 9.4245194	test: 11.5952574	best: 11.5938703 (2228)	total: 1m 8s	remaining: 54.4s
    2232:	learn: 9.4228462	test: 11.5942697	best: 11.5938703 (2228)	total: 1m 8s	remaining: 54.4s
    2233:	learn: 9.4221565	test: 11.5941734	best: 11.5938703 (2228)	total: 1m 8s	remaining: 54.4s
    2234:	learn: 9.4211932	test: 11.5921932	best: 11.5921932 (2234)	total: 1m 8s	remaining: 54.3s
    2235:	learn: 9.4205410	test: 11.5918559	best: 11.5918559 (2235)	total: 1m 8s	remaining: 54.3s
    2236:	learn: 9.4192921	test: 11.5915681	best: 11.5915681 (2236)	total: 1m 8s	remaining: 54.3s
    2237:	learn: 9.4187228	test: 11.5925595	best: 11.5915681 (2236)	total: 1m 8s	remaining: 54.2s
    2238:	learn: 9.4179829	test: 11.5915730	best: 11.5915681 (2236)	total: 1m 8s	remaining: 54.2s
    2239:	learn: 9.4177377	test: 11.5913613	best: 11.5913613 (2239)	total: 1m 8s	remaining: 54.2s
    2240:	learn: 9.4172851	test: 11.5899342	best: 11.5899342 (2240)	total: 1m 8s	remaining: 54.1s
    2241:	learn: 9.4149341	test: 11.5892206	best: 11.5892206 (2241)	total: 1m 8s	remaining: 54.1s
    2242:	learn: 9.4135707	test: 11.5889609	best: 11.5889609 (2242)	total: 1m 9s	remaining: 54.1s
    2243:	learn: 9.4119405	test: 11.5909681	best: 11.5889609 (2242)	total: 1m 9s	remaining: 54s
    2244:	learn: 9.4103338	test: 11.5899564	best: 11.5889609 (2242)	total: 1m 9s	remaining: 54s
    2245:	learn: 9.4074042	test: 11.5897646	best: 11.5889609 (2242)	total: 1m 9s	remaining: 54s
    2246:	learn: 9.4061262	test: 11.5894586	best: 11.5889609 (2242)	total: 1m 9s	remaining: 53.9s
    2247:	learn: 9.4050519	test: 11.5900165	best: 11.5889609 (2242)	total: 1m 9s	remaining: 53.9s
    2248:	learn: 9.4037407	test: 11.5897347	best: 11.5889609 (2242)	total: 1m 9s	remaining: 53.9s
    2249:	learn: 9.4031988	test: 11.5909444	best: 11.5889609 (2242)	total: 1m 9s	remaining: 53.8s
    2250:	learn: 9.4026251	test: 11.5905867	best: 11.5889609 (2242)	total: 1m 9s	remaining: 53.8s
    2251:	learn: 9.4019278	test: 11.5896597	best: 11.5889609 (2242)	total: 1m 9s	remaining: 53.8s
    2252:	learn: 9.4005904	test: 11.5894808	best: 11.5889609 (2242)	total: 1m 9s	remaining: 53.7s
    2253:	learn: 9.3993955	test: 11.5872606	best: 11.5872606 (2253)	total: 1m 9s	remaining: 53.7s
    2254:	learn: 9.3977553	test: 11.5876925	best: 11.5872606 (2253)	total: 1m 9s	remaining: 53.7s
    2255:	learn: 9.3966139	test: 11.5884405	best: 11.5872606 (2253)	total: 1m 9s	remaining: 53.6s
    2256:	learn: 9.3947274	test: 11.5894610	best: 11.5872606 (2253)	total: 1m 9s	remaining: 53.6s
    2257:	learn: 9.3946473	test: 11.5887013	best: 11.5872606 (2253)	total: 1m 9s	remaining: 53.6s
    2258:	learn: 9.3930488	test: 11.5895618	best: 11.5872606 (2253)	total: 1m 9s	remaining: 53.5s
    2259:	learn: 9.3910651	test: 11.5885680	best: 11.5872606 (2253)	total: 1m 9s	remaining: 53.5s
    2260:	learn: 9.3906469	test: 11.5883494	best: 11.5872606 (2253)	total: 1m 9s	remaining: 53.5s
    2261:	learn: 9.3901263	test: 11.5862073	best: 11.5862073 (2261)	total: 1m 9s	remaining: 53.4s
    2262:	learn: 9.3883772	test: 11.5845919	best: 11.5845919 (2262)	total: 1m 9s	remaining: 53.4s
    2263:	learn: 9.3879432	test: 11.5844389	best: 11.5844389 (2263)	total: 1m 9s	remaining: 53.4s
    2264:	learn: 9.3875387	test: 11.5837879	best: 11.5837879 (2264)	total: 1m 9s	remaining: 53.4s
    2265:	learn: 9.3860042	test: 11.5841900	best: 11.5837879 (2264)	total: 1m 9s	remaining: 53.3s
    2266:	learn: 9.3837383	test: 11.5796723	best: 11.5796723 (2266)	total: 1m 9s	remaining: 53.3s
    2267:	learn: 9.3817568	test: 11.5803235	best: 11.5796723 (2266)	total: 1m 9s	remaining: 53.3s
    2268:	learn: 9.3797677	test: 11.5811492	best: 11.5796723 (2266)	total: 1m 9s	remaining: 53.2s
    2269:	learn: 9.3793468	test: 11.5835320	best: 11.5796723 (2266)	total: 1m 9s	remaining: 53.2s
    2270:	learn: 9.3789596	test: 11.5841558	best: 11.5796723 (2266)	total: 1m 9s	remaining: 53.1s
    2271:	learn: 9.3786618	test: 11.5835661	best: 11.5796723 (2266)	total: 1m 9s	remaining: 53.1s
    2272:	learn: 9.3782964	test: 11.5820397	best: 11.5796723 (2266)	total: 1m 9s	remaining: 53.1s
    2273:	learn: 9.3775571	test: 11.5822709	best: 11.5796723 (2266)	total: 1m 9s	remaining: 53.1s
    2274:	learn: 9.3771727	test: 11.5819111	best: 11.5796723 (2266)	total: 1m 9s	remaining: 53s
    2275:	learn: 9.3752280	test: 11.5826354	best: 11.5796723 (2266)	total: 1m 10s	remaining: 53s
    2276:	learn: 9.3742129	test: 11.5818767	best: 11.5796723 (2266)	total: 1m 10s	remaining: 53s
    2277:	learn: 9.3726752	test: 11.5797327	best: 11.5796723 (2266)	total: 1m 10s	remaining: 53s
    2278:	learn: 9.3704727	test: 11.5790473	best: 11.5790473 (2278)	total: 1m 10s	remaining: 53s
    2279:	learn: 9.3697545	test: 11.5802210	best: 11.5790473 (2278)	total: 1m 10s	remaining: 52.9s
    2280:	learn: 9.3690902	test: 11.5808954	best: 11.5790473 (2278)	total: 1m 10s	remaining: 52.9s
    2281:	learn: 9.3679278	test: 11.5804709	best: 11.5790473 (2278)	total: 1m 10s	remaining: 52.9s
    2282:	learn: 9.3665417	test: 11.5826239	best: 11.5790473 (2278)	total: 1m 10s	remaining: 52.9s
    2283:	learn: 9.3658488	test: 11.5824663	best: 11.5790473 (2278)	total: 1m 10s	remaining: 52.8s
    2284:	learn: 9.3653075	test: 11.5831035	best: 11.5790473 (2278)	total: 1m 10s	remaining: 52.8s
    2285:	learn: 9.3649386	test: 11.5826454	best: 11.5790473 (2278)	total: 1m 10s	remaining: 52.8s
    2286:	learn: 9.3644979	test: 11.5800305	best: 11.5790473 (2278)	total: 1m 10s	remaining: 52.7s
    2287:	learn: 9.3629910	test: 11.5793078	best: 11.5790473 (2278)	total: 1m 10s	remaining: 52.7s
    2288:	learn: 9.3622638	test: 11.5785952	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.7s
    2289:	learn: 9.3616457	test: 11.5799159	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.7s
    2290:	learn: 9.3609710	test: 11.5798047	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.7s
    2291:	learn: 9.3599654	test: 11.5804136	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.7s
    2292:	learn: 9.3594000	test: 11.5795081	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.6s
    2293:	learn: 9.3587875	test: 11.5794707	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.6s
    2294:	learn: 9.3574731	test: 11.5808546	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.6s
    2295:	learn: 9.3567387	test: 11.5818049	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.5s
    2296:	learn: 9.3551586	test: 11.5824211	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.5s
    2297:	learn: 9.3549781	test: 11.5820127	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.5s
    2298:	learn: 9.3539047	test: 11.5829141	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.4s
    2299:	learn: 9.3532452	test: 11.5833132	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.4s
    2300:	learn: 9.3525853	test: 11.5831843	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.4s
    2301:	learn: 9.3519113	test: 11.5825154	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.3s
    2302:	learn: 9.3516814	test: 11.5833760	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.3s
    2303:	learn: 9.3506216	test: 11.5858126	best: 11.5785952 (2288)	total: 1m 10s	remaining: 52.2s
    2304:	learn: 9.3502931	test: 11.5868295	best: 11.5785952 (2288)	total: 1m 11s	remaining: 52.2s
    2305:	learn: 9.3498073	test: 11.5863370	best: 11.5785952 (2288)	total: 1m 11s	remaining: 52.2s
    2306:	learn: 9.3476465	test: 11.5842516	best: 11.5785952 (2288)	total: 1m 11s	remaining: 52.1s
    2307:	learn: 9.3468275	test: 11.5847880	best: 11.5785952 (2288)	total: 1m 11s	remaining: 52.1s
    2308:	learn: 9.3459607	test: 11.5845290	best: 11.5785952 (2288)	total: 1m 11s	remaining: 52.1s
    2309:	learn: 9.3441285	test: 11.5871005	best: 11.5785952 (2288)	total: 1m 11s	remaining: 52.1s
    2310:	learn: 9.3437637	test: 11.5866693	best: 11.5785952 (2288)	total: 1m 11s	remaining: 52s
    2311:	learn: 9.3432767	test: 11.5868910	best: 11.5785952 (2288)	total: 1m 11s	remaining: 52s
    2312:	learn: 9.3411544	test: 11.5858230	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.9s
    2313:	learn: 9.3405859	test: 11.5844926	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.9s
    2314:	learn: 9.3400421	test: 11.5848050	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.9s
    2315:	learn: 9.3389635	test: 11.5850044	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.8s
    2316:	learn: 9.3383206	test: 11.5840822	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.8s
    2317:	learn: 9.3349968	test: 11.5853440	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.8s
    2318:	learn: 9.3332010	test: 11.5858784	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.8s
    2319:	learn: 9.3321010	test: 11.5844213	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.7s
    2320:	learn: 9.3290572	test: 11.5843629	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.7s
    2321:	learn: 9.3285035	test: 11.5827052	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.6s
    2322:	learn: 9.3276174	test: 11.5835463	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.6s
    2323:	learn: 9.3257635	test: 11.5829256	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.6s
    2324:	learn: 9.3252378	test: 11.5841924	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.6s
    2325:	learn: 9.3239846	test: 11.5825347	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.5s
    2326:	learn: 9.3220256	test: 11.5805466	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.5s
    2327:	learn: 9.3209356	test: 11.5831559	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.5s
    2328:	learn: 9.3201085	test: 11.5830430	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.4s
    2329:	learn: 9.3197781	test: 11.5813213	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.4s
    2330:	learn: 9.3190535	test: 11.5821095	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.4s
    2331:	learn: 9.3178687	test: 11.5821078	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.3s
    2332:	learn: 9.3168647	test: 11.5830516	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.3s
    2333:	learn: 9.3156040	test: 11.5841853	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.3s
    2334:	learn: 9.3138011	test: 11.5824939	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.2s
    2335:	learn: 9.3132205	test: 11.5810315	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.2s
    2336:	learn: 9.3116071	test: 11.5809625	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.1s
    2337:	learn: 9.3108396	test: 11.5805566	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.1s
    2338:	learn: 9.3104260	test: 11.5817297	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51.1s
    2339:	learn: 9.3101886	test: 11.5820406	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51s
    2340:	learn: 9.3088144	test: 11.5800360	best: 11.5785952 (2288)	total: 1m 11s	remaining: 51s
    2341:	learn: 9.3081812	test: 11.5811009	best: 11.5785952 (2288)	total: 1m 12s	remaining: 51s
    2342:	learn: 9.3072877	test: 11.5793550	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.9s
    2343:	learn: 9.3064824	test: 11.5811016	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.9s
    2344:	learn: 9.3057504	test: 11.5800236	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.9s
    2345:	learn: 9.3047654	test: 11.5804099	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.8s
    2346:	learn: 9.3048045	test: 11.5798996	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.8s
    2347:	learn: 9.3045298	test: 11.5796675	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.8s
    2348:	learn: 9.3040028	test: 11.5811328	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.8s
    2349:	learn: 9.3033507	test: 11.5816035	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.7s
    2350:	learn: 9.3027459	test: 11.5825149	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.7s
    2351:	learn: 9.3025774	test: 11.5828609	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.7s
    2352:	learn: 9.3023917	test: 11.5835816	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.6s
    2353:	learn: 9.3004471	test: 11.5824962	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.6s
    2354:	learn: 9.2995007	test: 11.5810110	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.6s
    2355:	learn: 9.2991867	test: 11.5809900	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.5s
    2356:	learn: 9.2982779	test: 11.5807916	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.5s
    2357:	learn: 9.2977441	test: 11.5807792	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.5s
    2358:	learn: 9.2971095	test: 11.5806356	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.4s
    2359:	learn: 9.2964201	test: 11.5803507	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.4s
    2360:	learn: 9.2959611	test: 11.5806665	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.4s
    2361:	learn: 9.2953327	test: 11.5803746	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.3s
    2362:	learn: 9.2953939	test: 11.5810090	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.3s
    2363:	learn: 9.2946503	test: 11.5812914	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.3s
    2364:	learn: 9.2933009	test: 11.5813306	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.2s
    2365:	learn: 9.2924898	test: 11.5801924	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.2s
    2366:	learn: 9.2920404	test: 11.5800612	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.2s
    2367:	learn: 9.2909901	test: 11.5792905	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.1s
    2368:	learn: 9.2906790	test: 11.5801894	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.1s
    2369:	learn: 9.2901404	test: 11.5808959	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50.1s
    2370:	learn: 9.2898459	test: 11.5816405	best: 11.5785952 (2288)	total: 1m 12s	remaining: 50s
    2371:	learn: 9.2886837	test: 11.5775773	best: 11.5775773 (2371)	total: 1m 12s	remaining: 50s
    2372:	learn: 9.2883369	test: 11.5792418	best: 11.5775773 (2371)	total: 1m 12s	remaining: 50s
    2373:	learn: 9.2876798	test: 11.5793547	best: 11.5775773 (2371)	total: 1m 12s	remaining: 49.9s
    2374:	learn: 9.2872152	test: 11.5779723	best: 11.5775773 (2371)	total: 1m 12s	remaining: 49.9s
    2375:	learn: 9.2863550	test: 11.5787432	best: 11.5775773 (2371)	total: 1m 12s	remaining: 49.9s
    2376:	learn: 9.2845902	test: 11.5792059	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.9s
    2377:	learn: 9.2835228	test: 11.5797580	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.8s
    2378:	learn: 9.2819751	test: 11.5801747	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.8s
    2379:	learn: 9.2810585	test: 11.5807583	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.8s
    2380:	learn: 9.2805957	test: 11.5818529	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.7s
    2381:	learn: 9.2795341	test: 11.5824079	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.7s
    2382:	learn: 9.2776954	test: 11.5792510	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.6s
    2383:	learn: 9.2762595	test: 11.5787226	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.6s
    2384:	learn: 9.2760959	test: 11.5795453	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.6s
    2385:	learn: 9.2757844	test: 11.5810301	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.6s
    2386:	learn: 9.2756984	test: 11.5804642	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.5s
    2387:	learn: 9.2743959	test: 11.5817072	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.5s
    2388:	learn: 9.2739863	test: 11.5810787	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.5s
    2389:	learn: 9.2731309	test: 11.5804575	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.4s
    2390:	learn: 9.2718461	test: 11.5798548	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.4s
    2391:	learn: 9.2713996	test: 11.5797861	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.3s
    2392:	learn: 9.2710394	test: 11.5800126	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.3s
    2393:	learn: 9.2709803	test: 11.5800561	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.3s
    2394:	learn: 9.2697697	test: 11.5808137	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.3s
    2395:	learn: 9.2684090	test: 11.5811525	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.2s
    2396:	learn: 9.2681733	test: 11.5817476	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.2s
    2397:	learn: 9.2670517	test: 11.5811559	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.1s
    2398:	learn: 9.2664432	test: 11.5808644	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.1s
    2399:	learn: 9.2656582	test: 11.5808204	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49.1s
    2400:	learn: 9.2642852	test: 11.5833341	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49s
    2401:	learn: 9.2621704	test: 11.5807115	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49s
    2402:	learn: 9.2605122	test: 11.5803887	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49s
    2403:	learn: 9.2598926	test: 11.5791501	best: 11.5775773 (2371)	total: 1m 13s	remaining: 49s
    2404:	learn: 9.2589260	test: 11.5793607	best: 11.5775773 (2371)	total: 1m 13s	remaining: 48.9s
    2405:	learn: 9.2575246	test: 11.5817121	best: 11.5775773 (2371)	total: 1m 13s	remaining: 48.9s
    2406:	learn: 9.2567683	test: 11.5807233	best: 11.5775773 (2371)	total: 1m 13s	remaining: 48.9s
    2407:	learn: 9.2565275	test: 11.5795543	best: 11.5775773 (2371)	total: 1m 13s	remaining: 48.8s
    2408:	learn: 9.2556797	test: 11.5788936	best: 11.5775773 (2371)	total: 1m 13s	remaining: 48.8s
    2409:	learn: 9.2553400	test: 11.5791940	best: 11.5775773 (2371)	total: 1m 13s	remaining: 48.8s
    2410:	learn: 9.2544593	test: 11.5791163	best: 11.5775773 (2371)	total: 1m 13s	remaining: 48.7s
    2411:	learn: 9.2535454	test: 11.5807606	best: 11.5775773 (2371)	total: 1m 13s	remaining: 48.7s
    2412:	learn: 9.2520446	test: 11.5807233	best: 11.5775773 (2371)	total: 1m 13s	remaining: 48.7s
    2413:	learn: 9.2516675	test: 11.5816234	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.6s
    2414:	learn: 9.2513048	test: 11.5805546	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.6s
    2415:	learn: 9.2505540	test: 11.5813389	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.6s
    2416:	learn: 9.2495772	test: 11.5820042	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.5s
    2417:	learn: 9.2484077	test: 11.5835218	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.5s
    2418:	learn: 9.2470037	test: 11.5849238	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.5s
    2419:	learn: 9.2464335	test: 11.5846458	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.5s
    2420:	learn: 9.2453575	test: 11.5856114	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.4s
    2421:	learn: 9.2429468	test: 11.5843718	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.4s
    2422:	learn: 9.2421861	test: 11.5844983	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.4s
    2423:	learn: 9.2418236	test: 11.5831405	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.4s
    2424:	learn: 9.2413821	test: 11.5819828	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.3s
    2425:	learn: 9.2406054	test: 11.5810368	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.3s
    2426:	learn: 9.2400959	test: 11.5807065	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.3s
    2427:	learn: 9.2390815	test: 11.5807203	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.2s
    2428:	learn: 9.2382502	test: 11.5814836	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.2s
    2429:	learn: 9.2376385	test: 11.5809279	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.2s
    2430:	learn: 9.2370296	test: 11.5815151	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.1s
    2431:	learn: 9.2366657	test: 11.5809143	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.1s
    2432:	learn: 9.2353527	test: 11.5824886	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48.1s
    2433:	learn: 9.2325048	test: 11.5841845	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48s
    2434:	learn: 9.2306736	test: 11.5823225	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48s
    2435:	learn: 9.2298585	test: 11.5824528	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48s
    2436:	learn: 9.2295949	test: 11.5829465	best: 11.5775773 (2371)	total: 1m 14s	remaining: 48s
    2437:	learn: 9.2285738	test: 11.5813096	best: 11.5775773 (2371)	total: 1m 14s	remaining: 47.9s
    2438:	learn: 9.2273462	test: 11.5813733	best: 11.5775773 (2371)	total: 1m 14s	remaining: 47.9s
    2439:	learn: 9.2268185	test: 11.5805360	best: 11.5775773 (2371)	total: 1m 14s	remaining: 47.9s
    2440:	learn: 9.2265971	test: 11.5804968	best: 11.5775773 (2371)	total: 1m 14s	remaining: 47.8s
    2441:	learn: 9.2257237	test: 11.5804097	best: 11.5775773 (2371)	total: 1m 14s	remaining: 47.8s
    2442:	learn: 9.2250833	test: 11.5778573	best: 11.5775773 (2371)	total: 1m 14s	remaining: 47.8s
    2443:	learn: 9.2238677	test: 11.5801058	best: 11.5775773 (2371)	total: 1m 14s	remaining: 47.7s
    2444:	learn: 9.2222500	test: 11.5806591	best: 11.5775773 (2371)	total: 1m 14s	remaining: 47.7s
    2445:	learn: 9.2213275	test: 11.5811899	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.7s
    2446:	learn: 9.2209152	test: 11.5809708	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.6s
    2447:	learn: 9.2204130	test: 11.5806244	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.6s
    2448:	learn: 9.2198063	test: 11.5795923	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.6s
    2449:	learn: 9.2190487	test: 11.5810804	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.5s
    2450:	learn: 9.2164047	test: 11.5819412	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.5s
    2451:	learn: 9.2161780	test: 11.5824172	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.5s
    2452:	learn: 9.2149786	test: 11.5829767	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.4s
    2453:	learn: 9.2133946	test: 11.5835530	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.4s
    2454:	learn: 9.2128640	test: 11.5822979	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.4s
    2455:	learn: 9.2123880	test: 11.5825830	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.3s
    2456:	learn: 9.2116736	test: 11.5827714	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.3s
    2457:	learn: 9.2105916	test: 11.5809840	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.3s
    2458:	learn: 9.2093742	test: 11.5800200	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.2s
    2459:	learn: 9.2081688	test: 11.5813626	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.2s
    2460:	learn: 9.2055437	test: 11.5808290	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.2s
    2461:	learn: 9.2044824	test: 11.5815954	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.1s
    2462:	learn: 9.2039988	test: 11.5798152	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.1s
    2463:	learn: 9.2027592	test: 11.5800503	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47.1s
    2464:	learn: 9.2016072	test: 11.5798066	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47s
    2465:	learn: 9.1994198	test: 11.5797502	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47s
    2466:	learn: 9.1987917	test: 11.5792016	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47s
    2467:	learn: 9.1977473	test: 11.5789380	best: 11.5775773 (2371)	total: 1m 15s	remaining: 47s
    2468:	learn: 9.1977285	test: 11.5792886	best: 11.5775773 (2371)	total: 1m 15s	remaining: 46.9s
    2469:	learn: 9.1953438	test: 11.5792901	best: 11.5775773 (2371)	total: 1m 15s	remaining: 46.9s
    2470:	learn: 9.1948214	test: 11.5788475	best: 11.5775773 (2371)	total: 1m 15s	remaining: 46.9s
    2471:	learn: 9.1936307	test: 11.5775811	best: 11.5775773 (2371)	total: 1m 15s	remaining: 46.9s
    2472:	learn: 9.1927082	test: 11.5770579	best: 11.5770579 (2472)	total: 1m 15s	remaining: 46.9s
    2473:	learn: 9.1922235	test: 11.5756979	best: 11.5756979 (2473)	total: 1m 15s	remaining: 46.8s
    2474:	learn: 9.1913762	test: 11.5762953	best: 11.5756979 (2473)	total: 1m 15s	remaining: 46.8s
    2475:	learn: 9.1910348	test: 11.5770437	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.8s
    2476:	learn: 9.1907089	test: 11.5772358	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.7s
    2477:	learn: 9.1897881	test: 11.5780467	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.7s
    2478:	learn: 9.1889415	test: 11.5791721	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.7s
    2479:	learn: 9.1874703	test: 11.5792014	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.6s
    2480:	learn: 9.1875878	test: 11.5799431	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.6s
    2481:	learn: 9.1864848	test: 11.5798522	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.6s
    2482:	learn: 9.1835349	test: 11.5774380	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.6s
    2483:	learn: 9.1827709	test: 11.5766483	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.5s
    2484:	learn: 9.1815929	test: 11.5770544	best: 11.5756979 (2473)	total: 1m 16s	remaining: 46.5s
    2485:	learn: 9.1803882	test: 11.5747431	best: 11.5747431 (2485)	total: 1m 16s	remaining: 46.5s
    2486:	learn: 9.1790400	test: 11.5760955	best: 11.5747431 (2485)	total: 1m 16s	remaining: 46.4s
    2487:	learn: 9.1789074	test: 11.5766703	best: 11.5747431 (2485)	total: 1m 16s	remaining: 46.4s
    2488:	learn: 9.1787741	test: 11.5762441	best: 11.5747431 (2485)	total: 1m 16s	remaining: 46.4s
    2489:	learn: 9.1765891	test: 11.5751956	best: 11.5747431 (2485)	total: 1m 16s	remaining: 46.4s
    2490:	learn: 9.1761816	test: 11.5722521	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46.3s
    2491:	learn: 9.1740393	test: 11.5732551	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46.3s
    2492:	learn: 9.1733821	test: 11.5742676	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46.3s
    2493:	learn: 9.1721383	test: 11.5738004	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46.2s
    2494:	learn: 9.1707299	test: 11.5736225	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46.2s
    2495:	learn: 9.1689460	test: 11.5728073	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46.2s
    2496:	learn: 9.1685674	test: 11.5733947	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46.1s
    2497:	learn: 9.1683866	test: 11.5728403	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46.1s
    2498:	learn: 9.1680244	test: 11.5726881	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46.1s
    2499:	learn: 9.1677326	test: 11.5731185	best: 11.5722521 (2490)	total: 1m 16s	remaining: 46s
    2500:	learn: 9.1658406	test: 11.5709402	best: 11.5709402 (2500)	total: 1m 16s	remaining: 46s
    2501:	learn: 9.1655956	test: 11.5704310	best: 11.5704310 (2501)	total: 1m 16s	remaining: 46s
    2502:	learn: 9.1654285	test: 11.5708447	best: 11.5704310 (2501)	total: 1m 16s	remaining: 45.9s
    2503:	learn: 9.1635036	test: 11.5722059	best: 11.5704310 (2501)	total: 1m 16s	remaining: 45.9s
    2504:	learn: 9.1628362	test: 11.5718388	best: 11.5704310 (2501)	total: 1m 16s	remaining: 45.9s
    2505:	learn: 9.1622740	test: 11.5724361	best: 11.5704310 (2501)	total: 1m 16s	remaining: 45.9s
    2506:	learn: 9.1618592	test: 11.5724963	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.9s
    2507:	learn: 9.1606046	test: 11.5716897	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.8s
    2508:	learn: 9.1601576	test: 11.5718242	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.8s
    2509:	learn: 9.1592419	test: 11.5712190	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.8s
    2510:	learn: 9.1588591	test: 11.5710516	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.8s
    2511:	learn: 9.1581773	test: 11.5713562	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.7s
    2512:	learn: 9.1576335	test: 11.5704611	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.7s
    2513:	learn: 9.1570187	test: 11.5711682	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.7s
    2514:	learn: 9.1565569	test: 11.5714468	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.6s
    2515:	learn: 9.1552032	test: 11.5729293	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.6s
    2516:	learn: 9.1548475	test: 11.5733096	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.6s
    2517:	learn: 9.1534032	test: 11.5728549	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.5s
    2518:	learn: 9.1524980	test: 11.5723099	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.5s
    2519:	learn: 9.1520077	test: 11.5713586	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.5s
    2520:	learn: 9.1504779	test: 11.5705043	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.4s
    2521:	learn: 9.1493635	test: 11.5707218	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.4s
    2522:	learn: 9.1486416	test: 11.5709570	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.4s
    2523:	learn: 9.1462585	test: 11.5722217	best: 11.5704310 (2501)	total: 1m 17s	remaining: 45.3s
    2524:	learn: 9.1458760	test: 11.5704016	best: 11.5704016 (2524)	total: 1m 17s	remaining: 45.3s
    2525:	learn: 9.1453452	test: 11.5712821	best: 11.5704016 (2524)	total: 1m 17s	remaining: 45.3s
    2526:	learn: 9.1446810	test: 11.5709537	best: 11.5704016 (2524)	total: 1m 17s	remaining: 45.2s
    2527:	learn: 9.1432531	test: 11.5699865	best: 11.5699865 (2527)	total: 1m 17s	remaining: 45.2s
    2528:	learn: 9.1429872	test: 11.5702473	best: 11.5699865 (2527)	total: 1m 17s	remaining: 45.2s
    2529:	learn: 9.1429312	test: 11.5696450	best: 11.5696450 (2529)	total: 1m 17s	remaining: 45.1s
    2530:	learn: 9.1413209	test: 11.5715728	best: 11.5696450 (2529)	total: 1m 17s	remaining: 45.1s
    2531:	learn: 9.1393203	test: 11.5709021	best: 11.5696450 (2529)	total: 1m 17s	remaining: 45.1s
    2532:	learn: 9.1386274	test: 11.5719369	best: 11.5696450 (2529)	total: 1m 17s	remaining: 45.1s
    2533:	learn: 9.1385543	test: 11.5718385	best: 11.5696450 (2529)	total: 1m 17s	remaining: 45s
    2534:	learn: 9.1375501	test: 11.5721837	best: 11.5696450 (2529)	total: 1m 17s	remaining: 45s
    2535:	learn: 9.1357931	test: 11.5720825	best: 11.5696450 (2529)	total: 1m 17s	remaining: 45s
    2536:	learn: 9.1345176	test: 11.5720985	best: 11.5696450 (2529)	total: 1m 17s	remaining: 44.9s
    2537:	learn: 9.1332153	test: 11.5714763	best: 11.5696450 (2529)	total: 1m 17s	remaining: 44.9s
    2538:	learn: 9.1326019	test: 11.5696920	best: 11.5696450 (2529)	total: 1m 17s	remaining: 44.9s
    2539:	learn: 9.1314967	test: 11.5676197	best: 11.5676197 (2539)	total: 1m 17s	remaining: 44.8s
    2540:	learn: 9.1310796	test: 11.5687862	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.8s
    2541:	learn: 9.1304343	test: 11.5693498	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.8s
    2542:	learn: 9.1300869	test: 11.5695438	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.7s
    2543:	learn: 9.1287103	test: 11.5701573	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.7s
    2544:	learn: 9.1275129	test: 11.5698698	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.7s
    2545:	learn: 9.1271519	test: 11.5701162	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.6s
    2546:	learn: 9.1266680	test: 11.5706586	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.6s
    2547:	learn: 9.1259527	test: 11.5712270	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.6s
    2548:	learn: 9.1252399	test: 11.5703067	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.5s
    2549:	learn: 9.1235047	test: 11.5704729	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.5s
    2550:	learn: 9.1217237	test: 11.5690442	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.5s
    2551:	learn: 9.1206930	test: 11.5693268	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.4s
    2552:	learn: 9.1199076	test: 11.5698786	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.4s
    2553:	learn: 9.1192268	test: 11.5701277	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.4s
    2554:	learn: 9.1186128	test: 11.5681274	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.3s
    2555:	learn: 9.1177845	test: 11.5692636	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.3s
    2556:	learn: 9.1171012	test: 11.5676305	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.3s
    2557:	learn: 9.1142698	test: 11.5691697	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.2s
    2558:	learn: 9.1137395	test: 11.5682157	best: 11.5676197 (2539)	total: 1m 18s	remaining: 44.2s
    2559:	learn: 9.1103241	test: 11.5675143	best: 11.5675143 (2559)	total: 1m 18s	remaining: 44.2s
    2560:	learn: 9.1099238	test: 11.5672088	best: 11.5672088 (2560)	total: 1m 18s	remaining: 44.1s
    2561:	learn: 9.1085673	test: 11.5659734	best: 11.5659734 (2561)	total: 1m 18s	remaining: 44.1s
    2562:	learn: 9.1077008	test: 11.5641552	best: 11.5641552 (2562)	total: 1m 18s	remaining: 44.1s
    2563:	learn: 9.1077904	test: 11.5643159	best: 11.5641552 (2562)	total: 1m 18s	remaining: 44s
    2564:	learn: 9.1054395	test: 11.5637322	best: 11.5637322 (2564)	total: 1m 18s	remaining: 44s
    2565:	learn: 9.1052267	test: 11.5629538	best: 11.5629538 (2565)	total: 1m 18s	remaining: 44s
    2566:	learn: 9.1050516	test: 11.5637639	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.9s
    2567:	learn: 9.1027514	test: 11.5641561	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.9s
    2568:	learn: 9.1021265	test: 11.5655289	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.9s
    2569:	learn: 9.1021822	test: 11.5653765	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.8s
    2570:	learn: 9.1018838	test: 11.5662473	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.8s
    2571:	learn: 9.1009754	test: 11.5654102	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.8s
    2572:	learn: 9.0999508	test: 11.5669011	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.7s
    2573:	learn: 9.0990230	test: 11.5659336	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.7s
    2574:	learn: 9.0986376	test: 11.5655034	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.7s
    2575:	learn: 9.0980356	test: 11.5653679	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.6s
    2576:	learn: 9.0960881	test: 11.5668424	best: 11.5629538 (2565)	total: 1m 18s	remaining: 43.6s
    2577:	learn: 9.0946407	test: 11.5672381	best: 11.5629538 (2565)	total: 1m 19s	remaining: 43.6s
    2578:	learn: 9.0942716	test: 11.5663702	best: 11.5629538 (2565)	total: 1m 19s	remaining: 43.5s
    2579:	learn: 9.0936978	test: 11.5662024	best: 11.5629538 (2565)	total: 1m 19s	remaining: 43.5s
    2580:	learn: 9.0931549	test: 11.5658516	best: 11.5629538 (2565)	total: 1m 19s	remaining: 43.5s
    2581:	learn: 9.0924824	test: 11.5639188	best: 11.5629538 (2565)	total: 1m 19s	remaining: 43.4s
    2582:	learn: 9.0921093	test: 11.5644711	best: 11.5629538 (2565)	total: 1m 19s	remaining: 43.4s
    2583:	learn: 9.0915742	test: 11.5644538	best: 11.5629538 (2565)	total: 1m 19s	remaining: 43.4s
    2584:	learn: 9.0906926	test: 11.5624928	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.4s
    2585:	learn: 9.0891336	test: 11.5631017	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.3s
    2586:	learn: 9.0885115	test: 11.5634789	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.3s
    2587:	learn: 9.0874704	test: 11.5655109	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.3s
    2588:	learn: 9.0862288	test: 11.5639200	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.2s
    2589:	learn: 9.0854758	test: 11.5639747	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.2s
    2590:	learn: 9.0851736	test: 11.5647135	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.1s
    2591:	learn: 9.0851663	test: 11.5662139	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.1s
    2592:	learn: 9.0844167	test: 11.5670034	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.1s
    2593:	learn: 9.0836874	test: 11.5680482	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43.1s
    2594:	learn: 9.0827273	test: 11.5668707	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43s
    2595:	learn: 9.0821921	test: 11.5687921	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43s
    2596:	learn: 9.0805885	test: 11.5700044	best: 11.5624928 (2584)	total: 1m 19s	remaining: 43s
    2597:	learn: 9.0804963	test: 11.5706421	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.9s
    2598:	learn: 9.0795163	test: 11.5707683	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.9s
    2599:	learn: 9.0796967	test: 11.5713225	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.9s
    2600:	learn: 9.0794528	test: 11.5718060	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.8s
    2601:	learn: 9.0790676	test: 11.5722746	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.8s
    2602:	learn: 9.0786879	test: 11.5731216	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.8s
    2603:	learn: 9.0775432	test: 11.5719078	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.7s
    2604:	learn: 9.0763609	test: 11.5714818	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.7s
    2605:	learn: 9.0760543	test: 11.5708461	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.7s
    2606:	learn: 9.0753280	test: 11.5710765	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.6s
    2607:	learn: 9.0739922	test: 11.5722908	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.6s
    2608:	learn: 9.0722314	test: 11.5718543	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.6s
    2609:	learn: 9.0718132	test: 11.5713893	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.5s
    2610:	learn: 9.0703115	test: 11.5719753	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.5s
    2611:	learn: 9.0697114	test: 11.5718152	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.5s
    2612:	learn: 9.0682798	test: 11.5680769	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.4s
    2613:	learn: 9.0679337	test: 11.5693547	best: 11.5624928 (2584)	total: 1m 19s	remaining: 42.4s
    2614:	learn: 9.0669627	test: 11.5696561	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.4s
    2615:	learn: 9.0665063	test: 11.5692633	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.3s
    2616:	learn: 9.0659992	test: 11.5703293	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.3s
    2617:	learn: 9.0643577	test: 11.5700116	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.3s
    2618:	learn: 9.0637349	test: 11.5698109	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.2s
    2619:	learn: 9.0632585	test: 11.5669058	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.2s
    2620:	learn: 9.0623829	test: 11.5674406	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.2s
    2621:	learn: 9.0605669	test: 11.5660008	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.1s
    2622:	learn: 9.0600789	test: 11.5664198	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.1s
    2623:	learn: 9.0583140	test: 11.5663412	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42.1s
    2624:	learn: 9.0578214	test: 11.5648032	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42s
    2625:	learn: 9.0569794	test: 11.5652564	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42s
    2626:	learn: 9.0562190	test: 11.5648974	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42s
    2627:	learn: 9.0559114	test: 11.5641289	best: 11.5624928 (2584)	total: 1m 20s	remaining: 42s
    2628:	learn: 9.0553239	test: 11.5656490	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.9s
    2629:	learn: 9.0539761	test: 11.5649558	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.9s
    2630:	learn: 9.0532158	test: 11.5658605	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.9s
    2631:	learn: 9.0510867	test: 11.5640343	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.8s
    2632:	learn: 9.0501714	test: 11.5643151	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.8s
    2633:	learn: 9.0494578	test: 11.5634267	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.8s
    2634:	learn: 9.0489946	test: 11.5648912	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.7s
    2635:	learn: 9.0480312	test: 11.5653255	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.7s
    2636:	learn: 9.0473944	test: 11.5638112	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.7s
    2637:	learn: 9.0470798	test: 11.5651147	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.6s
    2638:	learn: 9.0464822	test: 11.5662565	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.6s
    2639:	learn: 9.0457433	test: 11.5675746	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.6s
    2640:	learn: 9.0451999	test: 11.5660697	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.5s
    2641:	learn: 9.0437897	test: 11.5664156	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.5s
    2642:	learn: 9.0413841	test: 11.5653030	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.5s
    2643:	learn: 9.0412417	test: 11.5654226	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.4s
    2644:	learn: 9.0398316	test: 11.5650192	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.4s
    2645:	learn: 9.0391490	test: 11.5645368	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.4s
    2646:	learn: 9.0385368	test: 11.5648937	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.3s
    2647:	learn: 9.0380022	test: 11.5654324	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.3s
    2648:	learn: 9.0377781	test: 11.5661475	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.3s
    2649:	learn: 9.0369484	test: 11.5663914	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.2s
    2650:	learn: 9.0348584	test: 11.5675585	best: 11.5624928 (2584)	total: 1m 20s	remaining: 41.2s
    2651:	learn: 9.0344525	test: 11.5678181	best: 11.5624928 (2584)	total: 1m 21s	remaining: 41.2s
    2652:	learn: 9.0330248	test: 11.5679537	best: 11.5624928 (2584)	total: 1m 21s	remaining: 41.1s
    2653:	learn: 9.0321592	test: 11.5685909	best: 11.5624928 (2584)	total: 1m 21s	remaining: 41.1s
    2654:	learn: 9.0310936	test: 11.5689228	best: 11.5624928 (2584)	total: 1m 21s	remaining: 41.1s
    2655:	learn: 9.0300202	test: 11.5701743	best: 11.5624928 (2584)	total: 1m 21s	remaining: 41s
    2656:	learn: 9.0288480	test: 11.5688777	best: 11.5624928 (2584)	total: 1m 21s	remaining: 41s
    2657:	learn: 9.0282631	test: 11.5685246	best: 11.5624928 (2584)	total: 1m 21s	remaining: 41s
    2658:	learn: 9.0275580	test: 11.5696378	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.9s
    2659:	learn: 9.0269030	test: 11.5698757	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.9s
    2660:	learn: 9.0256151	test: 11.5674047	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.9s
    2661:	learn: 9.0252895	test: 11.5673354	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.8s
    2662:	learn: 9.0247896	test: 11.5664738	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.8s
    2663:	learn: 9.0238670	test: 11.5680232	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.8s
    2664:	learn: 9.0226797	test: 11.5653468	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.8s
    2665:	learn: 9.0215466	test: 11.5693007	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.7s
    2666:	learn: 9.0210176	test: 11.5694934	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.7s
    2667:	learn: 9.0200572	test: 11.5699473	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.7s
    2668:	learn: 9.0191957	test: 11.5706537	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.6s
    2669:	learn: 9.0165910	test: 11.5698279	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.6s
    2670:	learn: 9.0158706	test: 11.5711542	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.6s
    2671:	learn: 9.0152896	test: 11.5704046	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.5s
    2672:	learn: 9.0139734	test: 11.5698865	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.5s
    2673:	learn: 9.0125914	test: 11.5717287	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.5s
    2674:	learn: 9.0124025	test: 11.5723406	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.4s
    2675:	learn: 9.0124805	test: 11.5728403	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.4s
    2676:	learn: 9.0119900	test: 11.5726507	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.4s
    2677:	learn: 9.0100678	test: 11.5714036	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.3s
    2678:	learn: 9.0100870	test: 11.5708551	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.3s
    2679:	learn: 9.0087178	test: 11.5713813	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.3s
    2680:	learn: 9.0079009	test: 11.5711434	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.2s
    2681:	learn: 9.0061013	test: 11.5710738	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.2s
    2682:	learn: 9.0057465	test: 11.5713160	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.2s
    2683:	learn: 9.0056106	test: 11.5715770	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.1s
    2684:	learn: 9.0051863	test: 11.5722053	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.1s
    2685:	learn: 9.0047897	test: 11.5724890	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.1s
    2686:	learn: 9.0047569	test: 11.5722113	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40.1s
    2687:	learn: 9.0036220	test: 11.5712682	best: 11.5624928 (2584)	total: 1m 21s	remaining: 40s
    2688:	learn: 9.0026345	test: 11.5728657	best: 11.5624928 (2584)	total: 1m 22s	remaining: 40s
    2689:	learn: 9.0020986	test: 11.5740138	best: 11.5624928 (2584)	total: 1m 22s	remaining: 40s
    2690:	learn: 9.0020194	test: 11.5750459	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.9s
    2691:	learn: 9.0004461	test: 11.5739193	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.9s
    2692:	learn: 8.9990879	test: 11.5710973	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.9s
    2693:	learn: 8.9985519	test: 11.5701962	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.8s
    2694:	learn: 8.9965101	test: 11.5704004	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.8s
    2695:	learn: 8.9953307	test: 11.5715517	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.8s
    2696:	learn: 8.9948529	test: 11.5712528	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.7s
    2697:	learn: 8.9943916	test: 11.5710153	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.7s
    2698:	learn: 8.9941126	test: 11.5707881	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.7s
    2699:	learn: 8.9936723	test: 11.5714429	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.6s
    2700:	learn: 8.9923044	test: 11.5678182	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.6s
    2701:	learn: 8.9916939	test: 11.5693982	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.6s
    2702:	learn: 8.9906296	test: 11.5671588	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.5s
    2703:	learn: 8.9904278	test: 11.5670733	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.5s
    2704:	learn: 8.9894190	test: 11.5688572	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.5s
    2705:	learn: 8.9888294	test: 11.5687440	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.4s
    2706:	learn: 8.9877559	test: 11.5680262	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.4s
    2707:	learn: 8.9875349	test: 11.5688921	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.4s
    2708:	learn: 8.9870275	test: 11.5683723	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.3s
    2709:	learn: 8.9865420	test: 11.5694356	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.3s
    2710:	learn: 8.9859108	test: 11.5707102	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.3s
    2711:	learn: 8.9846028	test: 11.5694164	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.2s
    2712:	learn: 8.9836959	test: 11.5686678	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.2s
    2713:	learn: 8.9815276	test: 11.5702925	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.2s
    2714:	learn: 8.9808330	test: 11.5687288	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.2s
    2715:	learn: 8.9804004	test: 11.5691034	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.1s
    2716:	learn: 8.9797595	test: 11.5691822	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.1s
    2717:	learn: 8.9783866	test: 11.5683021	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39.1s
    2718:	learn: 8.9771503	test: 11.5675019	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39s
    2719:	learn: 8.9762382	test: 11.5674137	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39s
    2720:	learn: 8.9754578	test: 11.5665466	best: 11.5624928 (2584)	total: 1m 22s	remaining: 39s
    2721:	learn: 8.9750763	test: 11.5672602	best: 11.5624928 (2584)	total: 1m 22s	remaining: 38.9s
    2722:	learn: 8.9734917	test: 11.5673430	best: 11.5624928 (2584)	total: 1m 22s	remaining: 38.9s
    2723:	learn: 8.9720045	test: 11.5670104	best: 11.5624928 (2584)	total: 1m 22s	remaining: 38.9s
    2724:	learn: 8.9698984	test: 11.5671941	best: 11.5624928 (2584)	total: 1m 22s	remaining: 38.8s
    2725:	learn: 8.9678789	test: 11.5676166	best: 11.5624928 (2584)	total: 1m 23s	remaining: 38.8s
    2726:	learn: 8.9666027	test: 11.5669472	best: 11.5624928 (2584)	total: 1m 23s	remaining: 38.8s
    2727:	learn: 8.9660963	test: 11.5675316	best: 11.5624928 (2584)	total: 1m 23s	remaining: 38.7s
    2728:	learn: 8.9661279	test: 11.5668213	best: 11.5624928 (2584)	total: 1m 23s	remaining: 38.7s
    2729:	learn: 8.9652993	test: 11.5664509	best: 11.5624928 (2584)	total: 1m 23s	remaining: 38.7s
    2730:	learn: 8.9642446	test: 11.5639807	best: 11.5624928 (2584)	total: 1m 23s	remaining: 38.6s
    2731:	learn: 8.9609789	test: 11.5609459	best: 11.5609459 (2731)	total: 1m 23s	remaining: 38.6s
    2732:	learn: 8.9602189	test: 11.5608821	best: 11.5608821 (2732)	total: 1m 23s	remaining: 38.6s
    2733:	learn: 8.9596741	test: 11.5601820	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.5s
    2734:	learn: 8.9587545	test: 11.5612440	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.5s
    2735:	learn: 8.9582881	test: 11.5611275	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.5s
    2736:	learn: 8.9574278	test: 11.5624052	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.4s
    2737:	learn: 8.9566791	test: 11.5629251	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.4s
    2738:	learn: 8.9557675	test: 11.5624739	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.4s
    2739:	learn: 8.9554127	test: 11.5638946	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.4s
    2740:	learn: 8.9545225	test: 11.5640363	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.3s
    2741:	learn: 8.9543294	test: 11.5645973	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.3s
    2742:	learn: 8.9538213	test: 11.5651596	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.3s
    2743:	learn: 8.9528755	test: 11.5670452	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.2s
    2744:	learn: 8.9523555	test: 11.5664943	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.2s
    2745:	learn: 8.9515495	test: 11.5646144	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.2s
    2746:	learn: 8.9497866	test: 11.5639866	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.1s
    2747:	learn: 8.9489159	test: 11.5626401	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.1s
    2748:	learn: 8.9483165	test: 11.5617262	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38.1s
    2749:	learn: 8.9474881	test: 11.5620479	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38s
    2750:	learn: 8.9474666	test: 11.5617102	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38s
    2751:	learn: 8.9473544	test: 11.5615410	best: 11.5601820 (2733)	total: 1m 23s	remaining: 38s
    2752:	learn: 8.9467901	test: 11.5606876	best: 11.5601820 (2733)	total: 1m 23s	remaining: 37.9s
    2753:	learn: 8.9455974	test: 11.5609020	best: 11.5601820 (2733)	total: 1m 23s	remaining: 37.9s
    2754:	learn: 8.9452055	test: 11.5621494	best: 11.5601820 (2733)	total: 1m 23s	remaining: 37.9s
    2755:	learn: 8.9447112	test: 11.5622212	best: 11.5601820 (2733)	total: 1m 23s	remaining: 37.8s
    2756:	learn: 8.9442399	test: 11.5626071	best: 11.5601820 (2733)	total: 1m 23s	remaining: 37.8s
    2757:	learn: 8.9429722	test: 11.5624541	best: 11.5601820 (2733)	total: 1m 23s	remaining: 37.8s
    2758:	learn: 8.9413747	test: 11.5588449	best: 11.5588449 (2758)	total: 1m 23s	remaining: 37.8s
    2759:	learn: 8.9409935	test: 11.5584656	best: 11.5584656 (2759)	total: 1m 23s	remaining: 37.7s
    2760:	learn: 8.9407556	test: 11.5582484	best: 11.5582484 (2760)	total: 1m 23s	remaining: 37.7s
    2761:	learn: 8.9400942	test: 11.5573893	best: 11.5573893 (2761)	total: 1m 24s	remaining: 37.7s
    2762:	learn: 8.9384771	test: 11.5578880	best: 11.5573893 (2761)	total: 1m 24s	remaining: 37.6s
    2763:	learn: 8.9381204	test: 11.5586039	best: 11.5573893 (2761)	total: 1m 24s	remaining: 37.6s
    2764:	learn: 8.9373392	test: 11.5576012	best: 11.5573893 (2761)	total: 1m 24s	remaining: 37.6s
    2765:	learn: 8.9366595	test: 11.5583453	best: 11.5573893 (2761)	total: 1m 24s	remaining: 37.5s
    2766:	learn: 8.9358449	test: 11.5570638	best: 11.5570638 (2766)	total: 1m 24s	remaining: 37.5s
    2767:	learn: 8.9347701	test: 11.5579260	best: 11.5570638 (2766)	total: 1m 24s	remaining: 37.5s
    2768:	learn: 8.9332156	test: 11.5563269	best: 11.5563269 (2768)	total: 1m 24s	remaining: 37.4s
    2769:	learn: 8.9311545	test: 11.5556979	best: 11.5556979 (2769)	total: 1m 24s	remaining: 37.4s
    2770:	learn: 8.9296433	test: 11.5572425	best: 11.5556979 (2769)	total: 1m 24s	remaining: 37.4s
    2771:	learn: 8.9279287	test: 11.5553505	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.4s
    2772:	learn: 8.9277814	test: 11.5561071	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.3s
    2773:	learn: 8.9269314	test: 11.5577073	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.3s
    2774:	learn: 8.9252605	test: 11.5589207	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.3s
    2775:	learn: 8.9241997	test: 11.5606380	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.2s
    2776:	learn: 8.9231366	test: 11.5617539	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.2s
    2777:	learn: 8.9231769	test: 11.5624266	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.2s
    2778:	learn: 8.9226910	test: 11.5641004	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.1s
    2779:	learn: 8.9218305	test: 11.5630117	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.1s
    2780:	learn: 8.9208933	test: 11.5643077	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37.1s
    2781:	learn: 8.9204996	test: 11.5649225	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37s
    2782:	learn: 8.9186632	test: 11.5645803	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37s
    2783:	learn: 8.9176294	test: 11.5649914	best: 11.5553505 (2771)	total: 1m 24s	remaining: 37s
    2784:	learn: 8.9169383	test: 11.5652379	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.9s
    2785:	learn: 8.9169838	test: 11.5639043	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.9s
    2786:	learn: 8.9158927	test: 11.5663504	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.9s
    2787:	learn: 8.9159283	test: 11.5662574	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.8s
    2788:	learn: 8.9145037	test: 11.5645036	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.8s
    2789:	learn: 8.9131235	test: 11.5623854	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.8s
    2790:	learn: 8.9123229	test: 11.5626054	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.8s
    2791:	learn: 8.9110438	test: 11.5600999	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.7s
    2792:	learn: 8.9098836	test: 11.5595752	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.7s
    2793:	learn: 8.9094845	test: 11.5595957	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.7s
    2794:	learn: 8.9086394	test: 11.5599226	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.6s
    2795:	learn: 8.9078960	test: 11.5601279	best: 11.5553505 (2771)	total: 1m 24s	remaining: 36.6s
    2796:	learn: 8.9075803	test: 11.5586411	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.6s
    2797:	learn: 8.9069548	test: 11.5591271	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.5s
    2798:	learn: 8.9064310	test: 11.5581785	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.5s
    2799:	learn: 8.9061649	test: 11.5584648	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.5s
    2800:	learn: 8.9055901	test: 11.5576744	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.4s
    2801:	learn: 8.9035229	test: 11.5589569	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.4s
    2802:	learn: 8.9032504	test: 11.5595808	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.4s
    2803:	learn: 8.9029647	test: 11.5580408	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.3s
    2804:	learn: 8.9015520	test: 11.5589744	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.3s
    2805:	learn: 8.9011825	test: 11.5590192	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.3s
    2806:	learn: 8.9003982	test: 11.5599749	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.2s
    2807:	learn: 8.8991523	test: 11.5607230	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.2s
    2808:	learn: 8.8985981	test: 11.5598405	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.2s
    2809:	learn: 8.8986914	test: 11.5605447	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.1s
    2810:	learn: 8.8977967	test: 11.5601829	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.1s
    2811:	learn: 8.8974205	test: 11.5597738	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.1s
    2812:	learn: 8.8956580	test: 11.5603460	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36.1s
    2813:	learn: 8.8951724	test: 11.5594373	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36s
    2814:	learn: 8.8944364	test: 11.5594145	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36s
    2815:	learn: 8.8938940	test: 11.5594550	best: 11.5553505 (2771)	total: 1m 25s	remaining: 36s
    2816:	learn: 8.8929518	test: 11.5596497	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.9s
    2817:	learn: 8.8919778	test: 11.5601489	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.9s
    2818:	learn: 8.8915678	test: 11.5631680	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.9s
    2819:	learn: 8.8912327	test: 11.5626543	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.8s
    2820:	learn: 8.8880962	test: 11.5571294	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.8s
    2821:	learn: 8.8862414	test: 11.5577010	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.8s
    2822:	learn: 8.8851966	test: 11.5585784	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.7s
    2823:	learn: 8.8838934	test: 11.5612360	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.7s
    2824:	learn: 8.8824466	test: 11.5616325	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.7s
    2825:	learn: 8.8817796	test: 11.5628907	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.6s
    2826:	learn: 8.8805869	test: 11.5620718	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.6s
    2827:	learn: 8.8802420	test: 11.5626168	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.6s
    2828:	learn: 8.8792589	test: 11.5613917	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.5s
    2829:	learn: 8.8786431	test: 11.5611216	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.5s
    2830:	learn: 8.8770445	test: 11.5603997	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.5s
    2831:	learn: 8.8767604	test: 11.5597903	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.5s
    2832:	learn: 8.8756395	test: 11.5618992	best: 11.5553505 (2771)	total: 1m 25s	remaining: 35.4s
    2833:	learn: 8.8745855	test: 11.5603232	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.4s
    2834:	learn: 8.8735083	test: 11.5643857	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.4s
    2835:	learn: 8.8730396	test: 11.5650541	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.3s
    2836:	learn: 8.8715912	test: 11.5646296	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.3s
    2837:	learn: 8.8709791	test: 11.5631527	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.3s
    2838:	learn: 8.8691932	test: 11.5632665	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.2s
    2839:	learn: 8.8685708	test: 11.5624047	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.2s
    2840:	learn: 8.8669468	test: 11.5612391	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.2s
    2841:	learn: 8.8660598	test: 11.5613040	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.1s
    2842:	learn: 8.8658715	test: 11.5615489	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.1s
    2843:	learn: 8.8652716	test: 11.5619342	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35.1s
    2844:	learn: 8.8644725	test: 11.5620252	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35s
    2845:	learn: 8.8642448	test: 11.5622562	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35s
    2846:	learn: 8.8644556	test: 11.5617147	best: 11.5553505 (2771)	total: 1m 26s	remaining: 35s
    2847:	learn: 8.8629762	test: 11.5621821	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.9s
    2848:	learn: 8.8628075	test: 11.5626282	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.9s
    2849:	learn: 8.8620624	test: 11.5628833	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.9s
    2850:	learn: 8.8616104	test: 11.5636784	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.8s
    2851:	learn: 8.8600717	test: 11.5623774	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.8s
    2852:	learn: 8.8586189	test: 11.5626707	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.8s
    2853:	learn: 8.8583225	test: 11.5617845	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.8s
    2854:	learn: 8.8578068	test: 11.5614118	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.7s
    2855:	learn: 8.8575889	test: 11.5623400	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.7s
    2856:	learn: 8.8562170	test: 11.5605613	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.7s
    2857:	learn: 8.8554047	test: 11.5599703	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.6s
    2858:	learn: 8.8545916	test: 11.5598031	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.6s
    2859:	learn: 8.8541601	test: 11.5589173	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.6s
    2860:	learn: 8.8541231	test: 11.5567515	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.5s
    2861:	learn: 8.8532480	test: 11.5563790	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.5s
    2862:	learn: 8.8522197	test: 11.5578078	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.5s
    2863:	learn: 8.8516679	test: 11.5593550	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.4s
    2864:	learn: 8.8508750	test: 11.5624786	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.4s
    2865:	learn: 8.8507727	test: 11.5610720	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.4s
    2866:	learn: 8.8498133	test: 11.5605100	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.3s
    2867:	learn: 8.8496552	test: 11.5589591	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.3s
    2868:	learn: 8.8490829	test: 11.5599779	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.3s
    2869:	learn: 8.8485651	test: 11.5606346	best: 11.5553505 (2771)	total: 1m 26s	remaining: 34.2s
    2870:	learn: 8.8481792	test: 11.5622842	best: 11.5553505 (2771)	total: 1m 27s	remaining: 34.2s
    2871:	learn: 8.8479994	test: 11.5633409	best: 11.5553505 (2771)	total: 1m 27s	remaining: 34.2s
    2872:	learn: 8.8451390	test: 11.5618727	best: 11.5553505 (2771)	total: 1m 27s	remaining: 34.1s
    2873:	learn: 8.8442613	test: 11.5629151	best: 11.5553505 (2771)	total: 1m 27s	remaining: 34.1s
    2874:	learn: 8.8432437	test: 11.5622235	best: 11.5553505 (2771)	total: 1m 27s	remaining: 34.1s
    2875:	learn: 8.8430048	test: 11.5619408	best: 11.5553505 (2771)	total: 1m 27s	remaining: 34.1s
    2876:	learn: 8.8420048	test: 11.5626836	best: 11.5553505 (2771)	total: 1m 27s	remaining: 34s
    2877:	learn: 8.8392261	test: 11.5591606	best: 11.5553505 (2771)	total: 1m 27s	remaining: 34s
    2878:	learn: 8.8372439	test: 11.5602195	best: 11.5553505 (2771)	total: 1m 27s	remaining: 34s
    2879:	learn: 8.8367387	test: 11.5603066	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.9s
    2880:	learn: 8.8355775	test: 11.5614874	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.9s
    2881:	learn: 8.8351833	test: 11.5642864	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.9s
    2882:	learn: 8.8343622	test: 11.5633721	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.8s
    2883:	learn: 8.8338357	test: 11.5623291	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.8s
    2884:	learn: 8.8327591	test: 11.5639362	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.8s
    2885:	learn: 8.8321514	test: 11.5615247	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.7s
    2886:	learn: 8.8317463	test: 11.5620229	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.7s
    2887:	learn: 8.8313593	test: 11.5621708	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.7s
    2888:	learn: 8.8299107	test: 11.5613101	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.6s
    2889:	learn: 8.8289030	test: 11.5601853	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.6s
    2890:	learn: 8.8280454	test: 11.5608293	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.6s
    2891:	learn: 8.8276479	test: 11.5598609	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.6s
    2892:	learn: 8.8276062	test: 11.5598345	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.5s
    2893:	learn: 8.8271164	test: 11.5595887	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.5s
    2894:	learn: 8.8262585	test: 11.5605080	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.5s
    2895:	learn: 8.8256986	test: 11.5601722	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.4s
    2896:	learn: 8.8255599	test: 11.5605099	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.4s
    2897:	learn: 8.8242380	test: 11.5605830	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.4s
    2898:	learn: 8.8236564	test: 11.5625415	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.3s
    2899:	learn: 8.8227731	test: 11.5627342	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.3s
    2900:	learn: 8.8226591	test: 11.5615679	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.3s
    2901:	learn: 8.8222469	test: 11.5613223	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.2s
    2902:	learn: 8.8214416	test: 11.5615773	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.2s
    2903:	learn: 8.8204611	test: 11.5649682	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.2s
    2904:	learn: 8.8198372	test: 11.5659020	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.1s
    2905:	learn: 8.8184244	test: 11.5660315	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.1s
    2906:	learn: 8.8166070	test: 11.5657852	best: 11.5553505 (2771)	total: 1m 27s	remaining: 33.1s
    2907:	learn: 8.8154310	test: 11.5652602	best: 11.5553505 (2771)	total: 1m 28s	remaining: 33.1s
    2908:	learn: 8.8147107	test: 11.5659527	best: 11.5553505 (2771)	total: 1m 28s	remaining: 33s
    2909:	learn: 8.8136154	test: 11.5672978	best: 11.5553505 (2771)	total: 1m 28s	remaining: 33s
    2910:	learn: 8.8113560	test: 11.5646701	best: 11.5553505 (2771)	total: 1m 28s	remaining: 33s
    2911:	learn: 8.8110416	test: 11.5630827	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.9s
    2912:	learn: 8.8106612	test: 11.5638449	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.9s
    2913:	learn: 8.8101048	test: 11.5641767	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.9s
    2914:	learn: 8.8101575	test: 11.5645320	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.8s
    2915:	learn: 8.8093132	test: 11.5637196	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.8s
    2916:	learn: 8.8080802	test: 11.5632075	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.8s
    2917:	learn: 8.8076755	test: 11.5629796	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.7s
    2918:	learn: 8.8065640	test: 11.5615222	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.7s
    2919:	learn: 8.8055967	test: 11.5605666	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.7s
    2920:	learn: 8.8050438	test: 11.5617417	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.6s
    2921:	learn: 8.8038581	test: 11.5606941	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.6s
    2922:	learn: 8.8026540	test: 11.5617597	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.6s
    2923:	learn: 8.8021233	test: 11.5610170	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.6s
    2924:	learn: 8.8009982	test: 11.5611561	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.5s
    2925:	learn: 8.8000929	test: 11.5616457	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.5s
    2926:	learn: 8.7992479	test: 11.5628489	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.5s
    2927:	learn: 8.7982906	test: 11.5639777	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.4s
    2928:	learn: 8.7975476	test: 11.5630373	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.4s
    2929:	learn: 8.7966386	test: 11.5643978	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.4s
    2930:	learn: 8.7964766	test: 11.5652711	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.3s
    2931:	learn: 8.7952441	test: 11.5646692	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.3s
    2932:	learn: 8.7945602	test: 11.5649082	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.3s
    2933:	learn: 8.7942658	test: 11.5641238	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.2s
    2934:	learn: 8.7938847	test: 11.5652364	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.2s
    2935:	learn: 8.7932694	test: 11.5655459	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.2s
    2936:	learn: 8.7927806	test: 11.5659497	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.1s
    2937:	learn: 8.7920539	test: 11.5655610	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.1s
    2938:	learn: 8.7917766	test: 11.5641324	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32.1s
    2939:	learn: 8.7896073	test: 11.5626132	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32s
    2940:	learn: 8.7887076	test: 11.5647351	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32s
    2941:	learn: 8.7874587	test: 11.5632896	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32s
    2942:	learn: 8.7855046	test: 11.5633635	best: 11.5553505 (2771)	total: 1m 28s	remaining: 32s
    2943:	learn: 8.7845454	test: 11.5633014	best: 11.5553505 (2771)	total: 1m 28s	remaining: 31.9s
    2944:	learn: 8.7835852	test: 11.5636090	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.9s
    2945:	learn: 8.7827628	test: 11.5632173	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.9s
    2946:	learn: 8.7811028	test: 11.5633178	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.8s
    2947:	learn: 8.7792776	test: 11.5643230	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.8s
    2948:	learn: 8.7792963	test: 11.5638962	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.8s
    2949:	learn: 8.7774630	test: 11.5643801	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.7s
    2950:	learn: 8.7775367	test: 11.5634257	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.7s
    2951:	learn: 8.7765150	test: 11.5624759	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.7s
    2952:	learn: 8.7755957	test: 11.5627177	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.6s
    2953:	learn: 8.7750338	test: 11.5639005	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.6s
    2954:	learn: 8.7745713	test: 11.5632761	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.6s
    2955:	learn: 8.7740547	test: 11.5648865	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.6s
    2956:	learn: 8.7736947	test: 11.5650563	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.5s
    2957:	learn: 8.7724232	test: 11.5636020	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.5s
    2958:	learn: 8.7721017	test: 11.5651932	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.5s
    2959:	learn: 8.7712933	test: 11.5647108	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.4s
    2960:	learn: 8.7705746	test: 11.5638420	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.4s
    2961:	learn: 8.7702323	test: 11.5638592	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.4s
    2962:	learn: 8.7694109	test: 11.5626636	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.4s
    2963:	learn: 8.7688875	test: 11.5622702	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.3s
    2964:	learn: 8.7675869	test: 11.5643030	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.3s
    2965:	learn: 8.7657985	test: 11.5634650	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.3s
    2966:	learn: 8.7653718	test: 11.5638872	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.2s
    2967:	learn: 8.7644297	test: 11.5639696	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.2s
    2968:	learn: 8.7640179	test: 11.5643803	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.2s
    2969:	learn: 8.7631280	test: 11.5647526	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.1s
    2970:	learn: 8.7628074	test: 11.5655024	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.1s
    2971:	learn: 8.7608722	test: 11.5668461	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31.1s
    2972:	learn: 8.7601492	test: 11.5666888	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31s
    2973:	learn: 8.7594423	test: 11.5678641	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31s
    2974:	learn: 8.7594907	test: 11.5676090	best: 11.5553505 (2771)	total: 1m 29s	remaining: 31s
    2975:	learn: 8.7577776	test: 11.5669573	best: 11.5553505 (2771)	total: 1m 29s	remaining: 30.9s
    2976:	learn: 8.7574410	test: 11.5675163	best: 11.5553505 (2771)	total: 1m 29s	remaining: 30.9s
    2977:	learn: 8.7556689	test: 11.5668415	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.9s
    2978:	learn: 8.7552116	test: 11.5661679	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.9s
    2979:	learn: 8.7532460	test: 11.5680788	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.8s
    2980:	learn: 8.7523593	test: 11.5666448	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.8s
    2981:	learn: 8.7515476	test: 11.5663292	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.8s
    2982:	learn: 8.7509485	test: 11.5657819	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.7s
    2983:	learn: 8.7501161	test: 11.5651409	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.7s
    2984:	learn: 8.7499833	test: 11.5650702	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.7s
    2985:	learn: 8.7497209	test: 11.5647303	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.6s
    2986:	learn: 8.7489255	test: 11.5658232	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.6s
    2987:	learn: 8.7484869	test: 11.5661662	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.6s
    2988:	learn: 8.7482001	test: 11.5662490	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.5s
    2989:	learn: 8.7471498	test: 11.5652220	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.5s
    2990:	learn: 8.7462524	test: 11.5656563	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.5s
    2991:	learn: 8.7459543	test: 11.5663623	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.5s
    2992:	learn: 8.7458996	test: 11.5668159	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.4s
    2993:	learn: 8.7457584	test: 11.5658744	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.4s
    2994:	learn: 8.7449192	test: 11.5669746	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.4s
    2995:	learn: 8.7439617	test: 11.5660016	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.3s
    2996:	learn: 8.7431974	test: 11.5664310	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.3s
    2997:	learn: 8.7422649	test: 11.5660609	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.3s
    2998:	learn: 8.7411524	test: 11.5664652	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.2s
    2999:	learn: 8.7398796	test: 11.5662282	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.2s
    3000:	learn: 8.7395972	test: 11.5658501	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.2s
    3001:	learn: 8.7392813	test: 11.5652520	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.1s
    3002:	learn: 8.7386293	test: 11.5654471	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.1s
    3003:	learn: 8.7381128	test: 11.5652518	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.1s
    3004:	learn: 8.7370785	test: 11.5630889	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30.1s
    3005:	learn: 8.7363789	test: 11.5637852	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30s
    3006:	learn: 8.7359602	test: 11.5636167	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30s
    3007:	learn: 8.7350442	test: 11.5634292	best: 11.5553505 (2771)	total: 1m 30s	remaining: 30s
    3008:	learn: 8.7344333	test: 11.5642229	best: 11.5553505 (2771)	total: 1m 30s	remaining: 29.9s
    3009:	learn: 8.7343304	test: 11.5646637	best: 11.5553505 (2771)	total: 1m 30s	remaining: 29.9s
    3010:	learn: 8.7340078	test: 11.5642426	best: 11.5553505 (2771)	total: 1m 30s	remaining: 29.9s
    3011:	learn: 8.7332533	test: 11.5651299	best: 11.5553505 (2771)	total: 1m 30s	remaining: 29.8s
    3012:	learn: 8.7318793	test: 11.5663058	best: 11.5553505 (2771)	total: 1m 30s	remaining: 29.8s
    3013:	learn: 8.7310628	test: 11.5664667	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.8s
    3014:	learn: 8.7300248	test: 11.5675478	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.7s
    3015:	learn: 8.7293027	test: 11.5692281	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.7s
    3016:	learn: 8.7284000	test: 11.5677344	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.7s
    3017:	learn: 8.7270149	test: 11.5684104	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.6s
    3018:	learn: 8.7265312	test: 11.5675447	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.6s
    3019:	learn: 8.7264168	test: 11.5682950	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.6s
    3020:	learn: 8.7255426	test: 11.5691103	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.6s
    3021:	learn: 8.7252904	test: 11.5688404	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.5s
    3022:	learn: 8.7245426	test: 11.5686599	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.5s
    3023:	learn: 8.7236762	test: 11.5676410	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.5s
    3024:	learn: 8.7220774	test: 11.5683353	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.4s
    3025:	learn: 8.7210695	test: 11.5678463	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.4s
    3026:	learn: 8.7207103	test: 11.5687004	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.4s
    3027:	learn: 8.7200717	test: 11.5673051	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.3s
    3028:	learn: 8.7180857	test: 11.5680064	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.3s
    3029:	learn: 8.7174982	test: 11.5681595	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.3s
    3030:	learn: 8.7171975	test: 11.5679817	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.2s
    3031:	learn: 8.7159638	test: 11.5661297	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.2s
    3032:	learn: 8.7150154	test: 11.5656306	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.2s
    3033:	learn: 8.7147331	test: 11.5667733	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.2s
    3034:	learn: 8.7131397	test: 11.5681753	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.1s
    3035:	learn: 8.7129264	test: 11.5660243	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.1s
    3036:	learn: 8.7120446	test: 11.5671990	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29.1s
    3037:	learn: 8.7115676	test: 11.5681694	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29s
    3038:	learn: 8.7114643	test: 11.5693161	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29s
    3039:	learn: 8.7107144	test: 11.5679364	best: 11.5553505 (2771)	total: 1m 31s	remaining: 29s
    3040:	learn: 8.7104240	test: 11.5679729	best: 11.5553505 (2771)	total: 1m 31s	remaining: 28.9s
    3041:	learn: 8.7100926	test: 11.5659410	best: 11.5553505 (2771)	total: 1m 31s	remaining: 28.9s
    3042:	learn: 8.7094513	test: 11.5659796	best: 11.5553505 (2771)	total: 1m 31s	remaining: 28.9s
    3043:	learn: 8.7090504	test: 11.5664810	best: 11.5553505 (2771)	total: 1m 31s	remaining: 28.8s
    3044:	learn: 8.7077131	test: 11.5647699	best: 11.5553505 (2771)	total: 1m 31s	remaining: 28.8s
    3045:	learn: 8.7069852	test: 11.5660549	best: 11.5553505 (2771)	total: 1m 31s	remaining: 28.8s
    3046:	learn: 8.7062150	test: 11.5660564	best: 11.5553505 (2771)	total: 1m 31s	remaining: 28.8s
    3047:	learn: 8.7048402	test: 11.5664250	best: 11.5553505 (2771)	total: 1m 31s	remaining: 28.7s
    3048:	learn: 8.7039857	test: 11.5652535	best: 11.5553505 (2771)	total: 1m 31s	remaining: 28.7s
    3049:	learn: 8.7034431	test: 11.5652327	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.7s
    3050:	learn: 8.7032072	test: 11.5671713	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.6s
    3051:	learn: 8.7000539	test: 11.5666161	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.6s
    3052:	learn: 8.6995013	test: 11.5670010	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.6s
    3053:	learn: 8.6984218	test: 11.5663815	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.5s
    3054:	learn: 8.6981734	test: 11.5665300	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.5s
    3055:	learn: 8.6978451	test: 11.5687806	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.5s
    3056:	learn: 8.6967987	test: 11.5663653	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.4s
    3057:	learn: 8.6964761	test: 11.5658760	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.4s
    3058:	learn: 8.6960487	test: 11.5665401	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.4s
    3059:	learn: 8.6959308	test: 11.5663040	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.3s
    3060:	learn: 8.6954292	test: 11.5656379	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.3s
    3061:	learn: 8.6951122	test: 11.5658303	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.3s
    3062:	learn: 8.6946510	test: 11.5653100	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.3s
    3063:	learn: 8.6937651	test: 11.5647680	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.2s
    3064:	learn: 8.6933845	test: 11.5640749	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.2s
    3065:	learn: 8.6932295	test: 11.5643032	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.2s
    3066:	learn: 8.6910836	test: 11.5638549	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.1s
    3067:	learn: 8.6888837	test: 11.5626320	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.1s
    3068:	learn: 8.6888606	test: 11.5608070	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28.1s
    3069:	learn: 8.6879224	test: 11.5608561	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28s
    3070:	learn: 8.6876976	test: 11.5610911	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28s
    3071:	learn: 8.6869462	test: 11.5608129	best: 11.5553505 (2771)	total: 1m 32s	remaining: 28s
    3072:	learn: 8.6861651	test: 11.5603403	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.9s
    3073:	learn: 8.6860710	test: 11.5603084	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.9s
    3074:	learn: 8.6858586	test: 11.5604591	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.9s
    3075:	learn: 8.6844060	test: 11.5626490	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.9s
    3076:	learn: 8.6839504	test: 11.5629804	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.8s
    3077:	learn: 8.6828548	test: 11.5618441	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.8s
    3078:	learn: 8.6828920	test: 11.5629089	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.8s
    3079:	learn: 8.6822129	test: 11.5621101	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.7s
    3080:	learn: 8.6811021	test: 11.5624198	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.7s
    3081:	learn: 8.6807288	test: 11.5620996	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.7s
    3082:	learn: 8.6800880	test: 11.5627442	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.6s
    3083:	learn: 8.6791681	test: 11.5616665	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.6s
    3084:	learn: 8.6783552	test: 11.5609328	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.6s
    3085:	learn: 8.6777779	test: 11.5606072	best: 11.5553505 (2771)	total: 1m 32s	remaining: 27.5s
    3086:	learn: 8.6763517	test: 11.5602578	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.5s
    3087:	learn: 8.6759534	test: 11.5611703	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.5s
    3088:	learn: 8.6757490	test: 11.5605688	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.4s
    3089:	learn: 8.6751889	test: 11.5612063	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.4s
    3090:	learn: 8.6751724	test: 11.5614052	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.4s
    3091:	learn: 8.6739687	test: 11.5623129	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.4s
    3092:	learn: 8.6738549	test: 11.5636579	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.3s
    3093:	learn: 8.6731814	test: 11.5628262	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.3s
    3094:	learn: 8.6724140	test: 11.5622252	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.3s
    3095:	learn: 8.6719598	test: 11.5648487	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.2s
    3096:	learn: 8.6716871	test: 11.5635584	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.2s
    3097:	learn: 8.6713469	test: 11.5627660	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.2s
    3098:	learn: 8.6711545	test: 11.5622067	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.1s
    3099:	learn: 8.6706575	test: 11.5627184	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.1s
    3100:	learn: 8.6685108	test: 11.5620270	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27.1s
    3101:	learn: 8.6683355	test: 11.5614347	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27s
    3102:	learn: 8.6675947	test: 11.5626091	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27s
    3103:	learn: 8.6671251	test: 11.5633603	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27s
    3104:	learn: 8.6664186	test: 11.5602900	best: 11.5553505 (2771)	total: 1m 33s	remaining: 27s
    3105:	learn: 8.6659556	test: 11.5598244	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.9s
    3106:	learn: 8.6657679	test: 11.5587791	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.9s
    3107:	learn: 8.6650297	test: 11.5588698	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.9s
    3108:	learn: 8.6649102	test: 11.5588176	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.8s
    3109:	learn: 8.6645031	test: 11.5587759	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.8s
    3110:	learn: 8.6638641	test: 11.5592253	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.8s
    3111:	learn: 8.6637653	test: 11.5589006	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.7s
    3112:	learn: 8.6634236	test: 11.5583531	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.7s
    3113:	learn: 8.6631596	test: 11.5579342	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.7s
    3114:	learn: 8.6626638	test: 11.5572265	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.6s
    3115:	learn: 8.6626683	test: 11.5575063	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.6s
    3116:	learn: 8.6613184	test: 11.5598254	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.6s
    3117:	learn: 8.6607648	test: 11.5592006	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.5s
    3118:	learn: 8.6606023	test: 11.5596762	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.5s
    3119:	learn: 8.6600370	test: 11.5577781	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.5s
    3120:	learn: 8.6596535	test: 11.5587466	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.5s
    3121:	learn: 8.6582274	test: 11.5577957	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.4s
    3122:	learn: 8.6578102	test: 11.5602349	best: 11.5553505 (2771)	total: 1m 33s	remaining: 26.4s
    3123:	learn: 8.6569603	test: 11.5595603	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.4s
    3124:	learn: 8.6557068	test: 11.5575390	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.3s
    3125:	learn: 8.6553064	test: 11.5581356	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.3s
    3126:	learn: 8.6547235	test: 11.5575426	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.3s
    3127:	learn: 8.6528292	test: 11.5580252	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.2s
    3128:	learn: 8.6522512	test: 11.5576177	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.2s
    3129:	learn: 8.6516400	test: 11.5573302	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.2s
    3130:	learn: 8.6506659	test: 11.5575723	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.1s
    3131:	learn: 8.6496228	test: 11.5585745	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.1s
    3132:	learn: 8.6487599	test: 11.5603141	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.1s
    3133:	learn: 8.6476016	test: 11.5592957	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26.1s
    3134:	learn: 8.6469556	test: 11.5596209	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26s
    3135:	learn: 8.6468467	test: 11.5602192	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26s
    3136:	learn: 8.6462812	test: 11.5615213	best: 11.5553505 (2771)	total: 1m 34s	remaining: 26s
    3137:	learn: 8.6459596	test: 11.5619698	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.9s
    3138:	learn: 8.6451240	test: 11.5618419	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.9s
    3139:	learn: 8.6446642	test: 11.5629223	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.9s
    3140:	learn: 8.6442482	test: 11.5632027	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.8s
    3141:	learn: 8.6436383	test: 11.5643086	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.8s
    3142:	learn: 8.6433270	test: 11.5637013	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.8s
    3143:	learn: 8.6419753	test: 11.5613997	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.8s
    3144:	learn: 8.6421374	test: 11.5625595	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.7s
    3145:	learn: 8.6415733	test: 11.5661586	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.7s
    3146:	learn: 8.6415905	test: 11.5658293	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.7s
    3147:	learn: 8.6415155	test: 11.5674538	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.6s
    3148:	learn: 8.6413346	test: 11.5671879	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.6s
    3149:	learn: 8.6410942	test: 11.5678213	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.6s
    3150:	learn: 8.6408770	test: 11.5683189	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.6s
    3151:	learn: 8.6402012	test: 11.5670241	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.5s
    3152:	learn: 8.6393062	test: 11.5676017	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.5s
    3153:	learn: 8.6390197	test: 11.5700859	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.5s
    3154:	learn: 8.6373828	test: 11.5695968	best: 11.5553505 (2771)	total: 1m 34s	remaining: 25.4s
    3155:	learn: 8.6371632	test: 11.5696480	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.4s
    3156:	learn: 8.6368083	test: 11.5686739	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.4s
    3157:	learn: 8.6363952	test: 11.5678368	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.4s
    3158:	learn: 8.6357093	test: 11.5672039	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.3s
    3159:	learn: 8.6338657	test: 11.5630399	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.3s
    3160:	learn: 8.6325805	test: 11.5620311	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.3s
    3161:	learn: 8.6312232	test: 11.5635084	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.2s
    3162:	learn: 8.6294098	test: 11.5649824	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.2s
    3163:	learn: 8.6274060	test: 11.5636962	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.2s
    3164:	learn: 8.6264133	test: 11.5642021	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.1s
    3165:	learn: 8.6253071	test: 11.5642866	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.1s
    3166:	learn: 8.6248620	test: 11.5648913	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.1s
    3167:	learn: 8.6246287	test: 11.5635567	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25.1s
    3168:	learn: 8.6228209	test: 11.5628430	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25s
    3169:	learn: 8.6213694	test: 11.5614951	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25s
    3170:	learn: 8.6211651	test: 11.5618331	best: 11.5553505 (2771)	total: 1m 35s	remaining: 25s
    3171:	learn: 8.6204232	test: 11.5605728	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.9s
    3172:	learn: 8.6196238	test: 11.5614380	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.9s
    3173:	learn: 8.6185913	test: 11.5609095	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.9s
    3174:	learn: 8.6182510	test: 11.5609275	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.9s
    3175:	learn: 8.6178729	test: 11.5617807	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.8s
    3176:	learn: 8.6177605	test: 11.5626170	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.8s
    3177:	learn: 8.6169501	test: 11.5631061	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.8s
    3178:	learn: 8.6166536	test: 11.5628647	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.7s
    3179:	learn: 8.6162823	test: 11.5625290	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.7s
    3180:	learn: 8.6155757	test: 11.5632413	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.7s
    3181:	learn: 8.6144913	test: 11.5631755	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.6s
    3182:	learn: 8.6140823	test: 11.5621343	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.6s
    3183:	learn: 8.6122712	test: 11.5606892	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.6s
    3184:	learn: 8.6119375	test: 11.5601698	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.6s
    3185:	learn: 8.6110114	test: 11.5589495	best: 11.5553505 (2771)	total: 1m 35s	remaining: 24.5s
    3186:	learn: 8.6102159	test: 11.5595844	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.5s
    3187:	learn: 8.6103955	test: 11.5592716	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.5s
    3188:	learn: 8.6091793	test: 11.5594399	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.4s
    3189:	learn: 8.6089627	test: 11.5588391	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.4s
    3190:	learn: 8.6081623	test: 11.5599556	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.4s
    3191:	learn: 8.6078347	test: 11.5602030	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.3s
    3192:	learn: 8.6069999	test: 11.5594434	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.3s
    3193:	learn: 8.6055870	test: 11.5603685	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.3s
    3194:	learn: 8.6051170	test: 11.5610509	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.3s
    3195:	learn: 8.6046958	test: 11.5604596	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.2s
    3196:	learn: 8.6039522	test: 11.5586054	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.2s
    3197:	learn: 8.6038518	test: 11.5595454	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.2s
    3198:	learn: 8.6034901	test: 11.5581090	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.2s
    3199:	learn: 8.6024334	test: 11.5602566	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.1s
    3200:	learn: 8.6021632	test: 11.5590719	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.1s
    3201:	learn: 8.6019604	test: 11.5583269	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.1s
    3202:	learn: 8.6004798	test: 11.5559440	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24.1s
    3203:	learn: 8.6002537	test: 11.5570453	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24s
    3204:	learn: 8.5997108	test: 11.5584889	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24s
    3205:	learn: 8.5992727	test: 11.5579191	best: 11.5553505 (2771)	total: 1m 36s	remaining: 24s
    3206:	learn: 8.5984702	test: 11.5569221	best: 11.5553505 (2771)	total: 1m 36s	remaining: 23.9s
    3207:	learn: 8.5983302	test: 11.5563315	best: 11.5553505 (2771)	total: 1m 36s	remaining: 23.9s
    3208:	learn: 8.5983256	test: 11.5563602	best: 11.5553505 (2771)	total: 1m 36s	remaining: 23.9s
    3209:	learn: 8.5953795	test: 11.5541421	best: 11.5541421 (3209)	total: 1m 36s	remaining: 23.9s
    3210:	learn: 8.5938936	test: 11.5529418	best: 11.5529418 (3210)	total: 1m 36s	remaining: 23.8s
    3211:	learn: 8.5937242	test: 11.5523809	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.8s
    3212:	learn: 8.5935237	test: 11.5534260	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.8s
    3213:	learn: 8.5923002	test: 11.5536750	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.7s
    3214:	learn: 8.5916882	test: 11.5545423	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.7s
    3215:	learn: 8.5909876	test: 11.5531501	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.7s
    3216:	learn: 8.5910235	test: 11.5543551	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.7s
    3217:	learn: 8.5908132	test: 11.5564385	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.6s
    3218:	learn: 8.5904949	test: 11.5573514	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.6s
    3219:	learn: 8.5898757	test: 11.5561337	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.6s
    3220:	learn: 8.5888929	test: 11.5555650	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.5s
    3221:	learn: 8.5868904	test: 11.5570719	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.5s
    3222:	learn: 8.5863178	test: 11.5572657	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.5s
    3223:	learn: 8.5856906	test: 11.5561915	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.5s
    3224:	learn: 8.5849178	test: 11.5545313	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.4s
    3225:	learn: 8.5841249	test: 11.5549359	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.4s
    3226:	learn: 8.5825003	test: 11.5541165	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.4s
    3227:	learn: 8.5816574	test: 11.5534246	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.4s
    3228:	learn: 8.5814612	test: 11.5527420	best: 11.5523809 (3211)	total: 1m 37s	remaining: 23.3s
    3229:	learn: 8.5806963	test: 11.5522633	best: 11.5522633 (3229)	total: 1m 37s	remaining: 23.3s
    3230:	learn: 8.5802296	test: 11.5529971	best: 11.5522633 (3229)	total: 1m 37s	remaining: 23.3s
    3231:	learn: 8.5795047	test: 11.5516955	best: 11.5516955 (3231)	total: 1m 37s	remaining: 23.2s
    3232:	learn: 8.5779569	test: 11.5519568	best: 11.5516955 (3231)	total: 1m 37s	remaining: 23.2s
    3233:	learn: 8.5773125	test: 11.5517940	best: 11.5516955 (3231)	total: 1m 37s	remaining: 23.2s
    3234:	learn: 8.5765722	test: 11.5519527	best: 11.5516955 (3231)	total: 1m 37s	remaining: 23.2s
    3235:	learn: 8.5755097	test: 11.5519536	best: 11.5516955 (3231)	total: 1m 37s	remaining: 23.1s
    3236:	learn: 8.5743756	test: 11.5507200	best: 11.5507200 (3236)	total: 1m 38s	remaining: 23.1s
    3237:	learn: 8.5725789	test: 11.5486877	best: 11.5486877 (3237)	total: 1m 38s	remaining: 23.1s
    3238:	learn: 8.5721096	test: 11.5493801	best: 11.5486877 (3237)	total: 1m 38s	remaining: 23s
    3239:	learn: 8.5711288	test: 11.5482274	best: 11.5482274 (3239)	total: 1m 38s	remaining: 23s
    3240:	learn: 8.5706441	test: 11.5509135	best: 11.5482274 (3239)	total: 1m 38s	remaining: 23s
    3241:	learn: 8.5708806	test: 11.5491580	best: 11.5482274 (3239)	total: 1m 38s	remaining: 23s
    3242:	learn: 8.5689368	test: 11.5473829	best: 11.5473829 (3242)	total: 1m 38s	remaining: 22.9s
    3243:	learn: 8.5681715	test: 11.5470776	best: 11.5470776 (3243)	total: 1m 38s	remaining: 22.9s
    3244:	learn: 8.5678890	test: 11.5449708	best: 11.5449708 (3244)	total: 1m 38s	remaining: 22.9s
    3245:	learn: 8.5670109	test: 11.5455609	best: 11.5449708 (3244)	total: 1m 38s	remaining: 22.9s
    3246:	learn: 8.5665906	test: 11.5452044	best: 11.5449708 (3244)	total: 1m 38s	remaining: 22.8s
    3247:	learn: 8.5659182	test: 11.5434600	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.8s
    3248:	learn: 8.5653361	test: 11.5440826	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.8s
    3249:	learn: 8.5645817	test: 11.5444511	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.7s
    3250:	learn: 8.5636991	test: 11.5456126	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.7s
    3251:	learn: 8.5629600	test: 11.5449842	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.7s
    3252:	learn: 8.5622290	test: 11.5447408	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.7s
    3253:	learn: 8.5620867	test: 11.5454127	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.6s
    3254:	learn: 8.5608480	test: 11.5461828	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.6s
    3255:	learn: 8.5599698	test: 11.5464542	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.6s
    3256:	learn: 8.5590343	test: 11.5471207	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.5s
    3257:	learn: 8.5570993	test: 11.5470239	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.5s
    3258:	learn: 8.5563563	test: 11.5496272	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.5s
    3259:	learn: 8.5554646	test: 11.5506064	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.4s
    3260:	learn: 8.5548918	test: 11.5506199	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.4s
    3261:	learn: 8.5531507	test: 11.5520007	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.4s
    3262:	learn: 8.5520178	test: 11.5504708	best: 11.5434600 (3247)	total: 1m 38s	remaining: 22.4s
    3263:	learn: 8.5516174	test: 11.5513298	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.3s
    3264:	learn: 8.5508661	test: 11.5531935	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.3s
    3265:	learn: 8.5495821	test: 11.5527886	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.3s
    3266:	learn: 8.5485553	test: 11.5512584	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.2s
    3267:	learn: 8.5479192	test: 11.5512836	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.2s
    3268:	learn: 8.5467711	test: 11.5523085	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.2s
    3269:	learn: 8.5452930	test: 11.5509865	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.1s
    3270:	learn: 8.5450657	test: 11.5533052	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.1s
    3271:	learn: 8.5448567	test: 11.5532551	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.1s
    3272:	learn: 8.5447665	test: 11.5521029	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22.1s
    3273:	learn: 8.5446386	test: 11.5508976	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22s
    3274:	learn: 8.5442515	test: 11.5507060	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22s
    3275:	learn: 8.5435406	test: 11.5502316	best: 11.5434600 (3247)	total: 1m 39s	remaining: 22s
    3276:	learn: 8.5426285	test: 11.5499398	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.9s
    3277:	learn: 8.5421644	test: 11.5507217	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.9s
    3278:	learn: 8.5415838	test: 11.5509795	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.9s
    3279:	learn: 8.5410276	test: 11.5515423	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.9s
    3280:	learn: 8.5408032	test: 11.5524413	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.8s
    3281:	learn: 8.5400943	test: 11.5529265	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.8s
    3282:	learn: 8.5394276	test: 11.5553087	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.8s
    3283:	learn: 8.5383175	test: 11.5550015	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.7s
    3284:	learn: 8.5370209	test: 11.5530482	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.7s
    3285:	learn: 8.5365213	test: 11.5560339	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.7s
    3286:	learn: 8.5354217	test: 11.5554272	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.6s
    3287:	learn: 8.5343878	test: 11.5563107	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.6s
    3288:	learn: 8.5335673	test: 11.5577788	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.6s
    3289:	learn: 8.5320228	test: 11.5560467	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.6s
    3290:	learn: 8.5317350	test: 11.5563755	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.5s
    3291:	learn: 8.5308367	test: 11.5562478	best: 11.5434600 (3247)	total: 1m 39s	remaining: 21.5s
    3292:	learn: 8.5306057	test: 11.5572751	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.5s
    3293:	learn: 8.5300048	test: 11.5587183	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.4s
    3294:	learn: 8.5297481	test: 11.5588641	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.4s
    3295:	learn: 8.5292225	test: 11.5604041	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.4s
    3296:	learn: 8.5275270	test: 11.5612576	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.4s
    3297:	learn: 8.5272044	test: 11.5610732	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.3s
    3298:	learn: 8.5271935	test: 11.5614004	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.3s
    3299:	learn: 8.5255387	test: 11.5595196	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.3s
    3300:	learn: 8.5251419	test: 11.5595312	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.2s
    3301:	learn: 8.5249803	test: 11.5616368	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.2s
    3302:	learn: 8.5247726	test: 11.5622913	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.2s
    3303:	learn: 8.5243808	test: 11.5620389	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.1s
    3304:	learn: 8.5237110	test: 11.5619683	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.1s
    3305:	learn: 8.5231653	test: 11.5624287	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.1s
    3306:	learn: 8.5219885	test: 11.5630530	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21.1s
    3307:	learn: 8.5208585	test: 11.5615377	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21s
    3308:	learn: 8.5203928	test: 11.5606928	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21s
    3309:	learn: 8.5194756	test: 11.5612378	best: 11.5434600 (3247)	total: 1m 40s	remaining: 21s
    3310:	learn: 8.5183778	test: 11.5606890	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.9s
    3311:	learn: 8.5179429	test: 11.5598505	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.9s
    3312:	learn: 8.5175514	test: 11.5584522	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.9s
    3313:	learn: 8.5157491	test: 11.5559704	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.8s
    3314:	learn: 8.5140213	test: 11.5566420	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.8s
    3315:	learn: 8.5133662	test: 11.5551748	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.8s
    3316:	learn: 8.5130447	test: 11.5553626	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.8s
    3317:	learn: 8.5117081	test: 11.5565753	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.7s
    3318:	learn: 8.5098850	test: 11.5538626	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.7s
    3319:	learn: 8.5093329	test: 11.5538308	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.7s
    3320:	learn: 8.5078452	test: 11.5547720	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.6s
    3321:	learn: 8.5072158	test: 11.5550943	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.6s
    3322:	learn: 8.5059550	test: 11.5539773	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.6s
    3323:	learn: 8.5044413	test: 11.5561535	best: 11.5434600 (3247)	total: 1m 40s	remaining: 20.5s
    3324:	learn: 8.5041093	test: 11.5561045	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.5s
    3325:	learn: 8.5031435	test: 11.5548172	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.5s
    3326:	learn: 8.5018785	test: 11.5556228	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.4s
    3327:	learn: 8.5014172	test: 11.5559782	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.4s
    3328:	learn: 8.5006346	test: 11.5558733	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.4s
    3329:	learn: 8.5000373	test: 11.5552791	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.4s
    3330:	learn: 8.5003161	test: 11.5556686	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.3s
    3331:	learn: 8.4993589	test: 11.5586722	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.3s
    3332:	learn: 8.4986261	test: 11.5590585	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.3s
    3333:	learn: 8.4965140	test: 11.5587829	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.2s
    3334:	learn: 8.4956209	test: 11.5592335	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.2s
    3335:	learn: 8.4952054	test: 11.5573437	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.2s
    3336:	learn: 8.4938630	test: 11.5557620	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.2s
    3337:	learn: 8.4931548	test: 11.5564229	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.1s
    3338:	learn: 8.4925571	test: 11.5573843	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.1s
    3339:	learn: 8.4911119	test: 11.5555543	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20.1s
    3340:	learn: 8.4908835	test: 11.5561239	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20s
    3341:	learn: 8.4895374	test: 11.5559954	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20s
    3342:	learn: 8.4890027	test: 11.5545814	best: 11.5434600 (3247)	total: 1m 41s	remaining: 20s
    3343:	learn: 8.4886918	test: 11.5543742	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.9s
    3344:	learn: 8.4880260	test: 11.5531900	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.9s
    3345:	learn: 8.4876763	test: 11.5524168	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.9s
    3346:	learn: 8.4866692	test: 11.5539605	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.9s
    3347:	learn: 8.4850800	test: 11.5549065	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.8s
    3348:	learn: 8.4843486	test: 11.5551224	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.8s
    3349:	learn: 8.4840240	test: 11.5527721	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.8s
    3350:	learn: 8.4831981	test: 11.5531835	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.7s
    3351:	learn: 8.4821042	test: 11.5542405	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.7s
    3352:	learn: 8.4809122	test: 11.5551955	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.7s
    3353:	learn: 8.4792056	test: 11.5552107	best: 11.5434600 (3247)	total: 1m 41s	remaining: 19.6s
    3354:	learn: 8.4788236	test: 11.5550067	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.6s
    3355:	learn: 8.4782873	test: 11.5551370	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.6s
    3356:	learn: 8.4779369	test: 11.5535177	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.5s
    3357:	learn: 8.4781015	test: 11.5548668	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.5s
    3358:	learn: 8.4773471	test: 11.5555052	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.5s
    3359:	learn: 8.4768106	test: 11.5553609	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.5s
    3360:	learn: 8.4761282	test: 11.5564185	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.4s
    3361:	learn: 8.4758530	test: 11.5563461	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.4s
    3362:	learn: 8.4741458	test: 11.5539914	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.4s
    3363:	learn: 8.4731595	test: 11.5536023	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.3s
    3364:	learn: 8.4724652	test: 11.5542639	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.3s
    3365:	learn: 8.4723130	test: 11.5527437	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.3s
    3366:	learn: 8.4723595	test: 11.5510576	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.2s
    3367:	learn: 8.4717415	test: 11.5500913	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.2s
    3368:	learn: 8.4708232	test: 11.5504653	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.2s
    3369:	learn: 8.4702993	test: 11.5509860	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.2s
    3370:	learn: 8.4697572	test: 11.5502359	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.1s
    3371:	learn: 8.4685597	test: 11.5504534	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.1s
    3372:	learn: 8.4679002	test: 11.5499600	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19.1s
    3373:	learn: 8.4674363	test: 11.5499578	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19s
    3374:	learn: 8.4669698	test: 11.5473007	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19s
    3375:	learn: 8.4664301	test: 11.5484571	best: 11.5434600 (3247)	total: 1m 42s	remaining: 19s
    3376:	learn: 8.4655985	test: 11.5461899	best: 11.5434600 (3247)	total: 1m 42s	remaining: 18.9s
    3377:	learn: 8.4650031	test: 11.5468075	best: 11.5434600 (3247)	total: 1m 42s	remaining: 18.9s
    3378:	learn: 8.4639874	test: 11.5473097	best: 11.5434600 (3247)	total: 1m 42s	remaining: 18.9s
    3379:	learn: 8.4628958	test: 11.5470630	best: 11.5434600 (3247)	total: 1m 42s	remaining: 18.9s
    3380:	learn: 8.4626591	test: 11.5460144	best: 11.5434600 (3247)	total: 1m 42s	remaining: 18.8s
    3381:	learn: 8.4607893	test: 11.5480780	best: 11.5434600 (3247)	total: 1m 42s	remaining: 18.8s
    3382:	learn: 8.4609246	test: 11.5482114	best: 11.5434600 (3247)	total: 1m 42s	remaining: 18.8s
    3383:	learn: 8.4600689	test: 11.5496216	best: 11.5434600 (3247)	total: 1m 42s	remaining: 18.7s
    3384:	learn: 8.4599987	test: 11.5505993	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.7s
    3385:	learn: 8.4593740	test: 11.5498504	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.7s
    3386:	learn: 8.4578683	test: 11.5515721	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.7s
    3387:	learn: 8.4569930	test: 11.5515953	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.6s
    3388:	learn: 8.4568267	test: 11.5512618	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.6s
    3389:	learn: 8.4557286	test: 11.5525202	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.6s
    3390:	learn: 8.4552109	test: 11.5508250	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.5s
    3391:	learn: 8.4546700	test: 11.5509055	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.5s
    3392:	learn: 8.4543163	test: 11.5499499	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.5s
    3393:	learn: 8.4535642	test: 11.5486479	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.4s
    3394:	learn: 8.4533371	test: 11.5496650	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.4s
    3395:	learn: 8.4525608	test: 11.5491489	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.4s
    3396:	learn: 8.4525015	test: 11.5494656	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.3s
    3397:	learn: 8.4523280	test: 11.5504715	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.3s
    3398:	learn: 8.4519907	test: 11.5517278	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.3s
    3399:	learn: 8.4510499	test: 11.5525106	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.3s
    3400:	learn: 8.4510282	test: 11.5530813	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.2s
    3401:	learn: 8.4499601	test: 11.5533217	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.2s
    3402:	learn: 8.4489202	test: 11.5530983	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.2s
    3403:	learn: 8.4480817	test: 11.5534857	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.1s
    3404:	learn: 8.4470672	test: 11.5539017	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.1s
    3405:	learn: 8.4460411	test: 11.5523428	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18.1s
    3406:	learn: 8.4460126	test: 11.5504215	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18s
    3407:	learn: 8.4451678	test: 11.5507567	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18s
    3408:	learn: 8.4446868	test: 11.5523199	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18s
    3409:	learn: 8.4439878	test: 11.5514646	best: 11.5434600 (3247)	total: 1m 43s	remaining: 18s
    3410:	learn: 8.4428506	test: 11.5529339	best: 11.5434600 (3247)	total: 1m 43s	remaining: 17.9s
    3411:	learn: 8.4424062	test: 11.5533141	best: 11.5434600 (3247)	total: 1m 43s	remaining: 17.9s
    3412:	learn: 8.4424179	test: 11.5535353	best: 11.5434600 (3247)	total: 1m 43s	remaining: 17.9s
    3413:	learn: 8.4417032	test: 11.5519858	best: 11.5434600 (3247)	total: 1m 43s	remaining: 17.8s
    3414:	learn: 8.4415551	test: 11.5544151	best: 11.5434600 (3247)	total: 1m 43s	remaining: 17.8s
    3415:	learn: 8.4410779	test: 11.5542581	best: 11.5434600 (3247)	total: 1m 43s	remaining: 17.8s
    3416:	learn: 8.4409756	test: 11.5539603	best: 11.5434600 (3247)	total: 1m 43s	remaining: 17.7s
    3417:	learn: 8.4402889	test: 11.5541473	best: 11.5434600 (3247)	total: 1m 43s	remaining: 17.7s
    3418:	learn: 8.4395896	test: 11.5538398	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.7s
    3419:	learn: 8.4390480	test: 11.5530158	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.6s
    3420:	learn: 8.4383513	test: 11.5533312	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.6s
    3421:	learn: 8.4372311	test: 11.5558831	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.6s
    3422:	learn: 8.4366108	test: 11.5551459	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.6s
    3423:	learn: 8.4355289	test: 11.5547398	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.5s
    3424:	learn: 8.4346995	test: 11.5548451	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.5s
    3425:	learn: 8.4343746	test: 11.5533467	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.5s
    3426:	learn: 8.4341488	test: 11.5533165	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.4s
    3427:	learn: 8.4342184	test: 11.5553002	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.4s
    3428:	learn: 8.4327767	test: 11.5536509	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.4s
    3429:	learn: 8.4321434	test: 11.5531726	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.3s
    3430:	learn: 8.4311237	test: 11.5536267	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.3s
    3431:	learn: 8.4311547	test: 11.5542877	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.3s
    3432:	learn: 8.4306352	test: 11.5540660	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.3s
    3433:	learn: 8.4303818	test: 11.5533312	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.2s
    3434:	learn: 8.4289806	test: 11.5506605	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.2s
    3435:	learn: 8.4288755	test: 11.5518676	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.2s
    3436:	learn: 8.4287964	test: 11.5526143	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.1s
    3437:	learn: 8.4282081	test: 11.5536916	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.1s
    3438:	learn: 8.4276786	test: 11.5535165	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17.1s
    3439:	learn: 8.4277976	test: 11.5524632	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17s
    3440:	learn: 8.4271436	test: 11.5510942	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17s
    3441:	learn: 8.4269531	test: 11.5513031	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17s
    3442:	learn: 8.4263500	test: 11.5522100	best: 11.5434600 (3247)	total: 1m 44s	remaining: 17s
    3443:	learn: 8.4258985	test: 11.5529296	best: 11.5434600 (3247)	total: 1m 44s	remaining: 16.9s
    3444:	learn: 8.4254691	test: 11.5534563	best: 11.5434600 (3247)	total: 1m 44s	remaining: 16.9s
    3445:	learn: 8.4249114	test: 11.5539099	best: 11.5434600 (3247)	total: 1m 44s	remaining: 16.9s
    3446:	learn: 8.4244603	test: 11.5545290	best: 11.5434600 (3247)	total: 1m 44s	remaining: 16.8s
    3447:	learn: 8.4243940	test: 11.5546473	best: 11.5434600 (3247)	total: 1m 44s	remaining: 16.8s
    3448:	learn: 8.4227450	test: 11.5556570	best: 11.5434600 (3247)	total: 1m 44s	remaining: 16.8s
    3449:	learn: 8.4221793	test: 11.5530907	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.7s
    3450:	learn: 8.4220327	test: 11.5524640	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.7s
    3451:	learn: 8.4215471	test: 11.5532828	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.7s
    3452:	learn: 8.4213992	test: 11.5525443	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.6s
    3453:	learn: 8.4205403	test: 11.5501443	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.6s
    3454:	learn: 8.4202030	test: 11.5500401	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.6s
    3455:	learn: 8.4197549	test: 11.5498319	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.6s
    3456:	learn: 8.4188056	test: 11.5485807	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.5s
    3457:	learn: 8.4183646	test: 11.5493357	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.5s
    3458:	learn: 8.4181445	test: 11.5494190	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.5s
    3459:	learn: 8.4172487	test: 11.5503252	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.4s
    3460:	learn: 8.4167810	test: 11.5498526	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.4s
    3461:	learn: 8.4156598	test: 11.5495137	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.4s
    3462:	learn: 8.4157974	test: 11.5498783	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.3s
    3463:	learn: 8.4153811	test: 11.5509698	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.3s
    3464:	learn: 8.4145325	test: 11.5516036	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.3s
    3465:	learn: 8.4140479	test: 11.5506541	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.3s
    3466:	learn: 8.4130096	test: 11.5508440	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.2s
    3467:	learn: 8.4116575	test: 11.5511576	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.2s
    3468:	learn: 8.4111384	test: 11.5517468	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.2s
    3469:	learn: 8.4109204	test: 11.5510518	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.1s
    3470:	learn: 8.4104735	test: 11.5522282	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.1s
    3471:	learn: 8.4092684	test: 11.5531254	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16.1s
    3472:	learn: 8.4096024	test: 11.5533590	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16s
    3473:	learn: 8.4089682	test: 11.5532203	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16s
    3474:	learn: 8.4087982	test: 11.5525158	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16s
    3475:	learn: 8.4081610	test: 11.5526404	best: 11.5434600 (3247)	total: 1m 45s	remaining: 16s
    3476:	learn: 8.4077366	test: 11.5527314	best: 11.5434600 (3247)	total: 1m 45s	remaining: 15.9s
    3477:	learn: 8.4069798	test: 11.5519216	best: 11.5434600 (3247)	total: 1m 45s	remaining: 15.9s
    3478:	learn: 8.4069268	test: 11.5512833	best: 11.5434600 (3247)	total: 1m 45s	remaining: 15.9s
    3479:	learn: 8.4066223	test: 11.5513195	best: 11.5434600 (3247)	total: 1m 45s	remaining: 15.8s
    3480:	learn: 8.4063156	test: 11.5524586	best: 11.5434600 (3247)	total: 1m 45s	remaining: 15.8s
    3481:	learn: 8.4051047	test: 11.5515714	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.8s
    3482:	learn: 8.4042210	test: 11.5525603	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.7s
    3483:	learn: 8.4034737	test: 11.5534163	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.7s
    3484:	learn: 8.4032782	test: 11.5528243	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.7s
    3485:	learn: 8.4028296	test: 11.5532796	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.6s
    3486:	learn: 8.4018750	test: 11.5534765	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.6s
    3487:	learn: 8.4001932	test: 11.5513384	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.6s
    3488:	learn: 8.3990330	test: 11.5518991	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.6s
    3489:	learn: 8.3981337	test: 11.5529282	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.5s
    3490:	learn: 8.3973725	test: 11.5525672	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.5s
    3491:	learn: 8.3966161	test: 11.5526648	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.5s
    3492:	learn: 8.3960688	test: 11.5523265	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.4s
    3493:	learn: 8.3958114	test: 11.5518411	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.4s
    3494:	learn: 8.3955230	test: 11.5512913	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.4s
    3495:	learn: 8.3949219	test: 11.5516502	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.3s
    3496:	learn: 8.3944512	test: 11.5523938	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.3s
    3497:	learn: 8.3939782	test: 11.5518269	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.3s
    3498:	learn: 8.3931009	test: 11.5529054	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.2s
    3499:	learn: 8.3921769	test: 11.5512511	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.2s
    3500:	learn: 8.3917876	test: 11.5511800	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.2s
    3501:	learn: 8.3911995	test: 11.5506775	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.2s
    3502:	learn: 8.3905081	test: 11.5496687	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.1s
    3503:	learn: 8.3899399	test: 11.5493657	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.1s
    3504:	learn: 8.3889156	test: 11.5482244	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15.1s
    3505:	learn: 8.3877092	test: 11.5493074	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15s
    3506:	learn: 8.3874511	test: 11.5504890	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15s
    3507:	learn: 8.3863391	test: 11.5523557	best: 11.5434600 (3247)	total: 1m 46s	remaining: 15s
    3508:	learn: 8.3864743	test: 11.5530503	best: 11.5434600 (3247)	total: 1m 46s	remaining: 14.9s
    3509:	learn: 8.3859165	test: 11.5538402	best: 11.5434600 (3247)	total: 1m 46s	remaining: 14.9s
    3510:	learn: 8.3842129	test: 11.5561080	best: 11.5434600 (3247)	total: 1m 46s	remaining: 14.9s
    3511:	learn: 8.3836919	test: 11.5548858	best: 11.5434600 (3247)	total: 1m 46s	remaining: 14.9s
    3512:	learn: 8.3828791	test: 11.5546098	best: 11.5434600 (3247)	total: 1m 46s	remaining: 14.8s
    3513:	learn: 8.3822548	test: 11.5551281	best: 11.5434600 (3247)	total: 1m 46s	remaining: 14.8s
    3514:	learn: 8.3818729	test: 11.5560098	best: 11.5434600 (3247)	total: 1m 46s	remaining: 14.8s
    3515:	learn: 8.3810097	test: 11.5557486	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.7s
    3516:	learn: 8.3806405	test: 11.5568267	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.7s
    3517:	learn: 8.3797210	test: 11.5572670	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.7s
    3518:	learn: 8.3785198	test: 11.5596402	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.6s
    3519:	learn: 8.3774471	test: 11.5593775	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.6s
    3520:	learn: 8.3767845	test: 11.5584445	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.6s
    3521:	learn: 8.3759236	test: 11.5598287	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.6s
    3522:	learn: 8.3757708	test: 11.5597023	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.5s
    3523:	learn: 8.3750123	test: 11.5600759	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.5s
    3524:	learn: 8.3740433	test: 11.5595087	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.5s
    3525:	learn: 8.3724414	test: 11.5583715	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.4s
    3526:	learn: 8.3713636	test: 11.5583021	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.4s
    3527:	learn: 8.3711263	test: 11.5579469	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.4s
    3528:	learn: 8.3706028	test: 11.5600339	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.3s
    3529:	learn: 8.3688656	test: 11.5598282	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.3s
    3530:	learn: 8.3685777	test: 11.5595878	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.3s
    3531:	learn: 8.3679643	test: 11.5569771	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.3s
    3532:	learn: 8.3671204	test: 11.5564819	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.2s
    3533:	learn: 8.3661522	test: 11.5568003	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.2s
    3534:	learn: 8.3650110	test: 11.5592820	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.2s
    3535:	learn: 8.3637826	test: 11.5584132	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.1s
    3536:	learn: 8.3633311	test: 11.5576753	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.1s
    3537:	learn: 8.3631529	test: 11.5578824	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14.1s
    3538:	learn: 8.3621546	test: 11.5566107	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14s
    3539:	learn: 8.3612059	test: 11.5574424	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14s
    3540:	learn: 8.3602044	test: 11.5578656	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14s
    3541:	learn: 8.3598703	test: 11.5586266	best: 11.5434600 (3247)	total: 1m 47s	remaining: 14s
    3542:	learn: 8.3589547	test: 11.5589217	best: 11.5434600 (3247)	total: 1m 47s	remaining: 13.9s
    3543:	learn: 8.3585877	test: 11.5600757	best: 11.5434600 (3247)	total: 1m 47s	remaining: 13.9s
    3544:	learn: 8.3581106	test: 11.5605128	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.9s
    3545:	learn: 8.3572821	test: 11.5608335	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.8s
    3546:	learn: 8.3567631	test: 11.5612963	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.8s
    3547:	learn: 8.3559639	test: 11.5604588	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.8s
    3548:	learn: 8.3559172	test: 11.5601949	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.7s
    3549:	learn: 8.3546142	test: 11.5590859	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.7s
    3550:	learn: 8.3538668	test: 11.5606261	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.7s
    3551:	learn: 8.3537701	test: 11.5598639	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.7s
    3552:	learn: 8.3529592	test: 11.5598511	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.6s
    3553:	learn: 8.3533222	test: 11.5595082	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.6s
    3554:	learn: 8.3525180	test: 11.5597795	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.6s
    3555:	learn: 8.3515175	test: 11.5590236	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.5s
    3556:	learn: 8.3514046	test: 11.5591366	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.5s
    3557:	learn: 8.3509829	test: 11.5588674	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.5s
    3558:	learn: 8.3499558	test: 11.5576588	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.4s
    3559:	learn: 8.3486101	test: 11.5574679	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.4s
    3560:	learn: 8.3483256	test: 11.5575538	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.4s
    3561:	learn: 8.3477146	test: 11.5584349	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.4s
    3562:	learn: 8.3475168	test: 11.5577393	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.3s
    3563:	learn: 8.3464986	test: 11.5577883	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.3s
    3564:	learn: 8.3460109	test: 11.5557842	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.3s
    3565:	learn: 8.3456852	test: 11.5554553	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.2s
    3566:	learn: 8.3448967	test: 11.5541031	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.2s
    3567:	learn: 8.3439959	test: 11.5553142	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.2s
    3568:	learn: 8.3439277	test: 11.5565936	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.1s
    3569:	learn: 8.3433084	test: 11.5557483	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.1s
    3570:	learn: 8.3429507	test: 11.5566790	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13.1s
    3571:	learn: 8.3427466	test: 11.5572136	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13s
    3572:	learn: 8.3413033	test: 11.5560103	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13s
    3573:	learn: 8.3396044	test: 11.5549846	best: 11.5434600 (3247)	total: 1m 48s	remaining: 13s
    3574:	learn: 8.3384293	test: 11.5539744	best: 11.5434600 (3247)	total: 1m 49s	remaining: 13s
    3575:	learn: 8.3383333	test: 11.5546068	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.9s
    3576:	learn: 8.3372541	test: 11.5524663	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.9s
    3577:	learn: 8.3367986	test: 11.5525531	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.9s
    3578:	learn: 8.3364703	test: 11.5535709	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.9s
    3579:	learn: 8.3363550	test: 11.5528254	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.8s
    3580:	learn: 8.3362496	test: 11.5528640	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.8s
    3581:	learn: 8.3357747	test: 11.5530199	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.8s
    3582:	learn: 8.3348678	test: 11.5532722	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.7s
    3583:	learn: 8.3344465	test: 11.5526555	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.7s
    3584:	learn: 8.3340346	test: 11.5527031	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.7s
    3585:	learn: 8.3325654	test: 11.5508304	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.7s
    3586:	learn: 8.3312550	test: 11.5508234	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.6s
    3587:	learn: 8.3302064	test: 11.5513646	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.6s
    3588:	learn: 8.3298407	test: 11.5533884	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.6s
    3589:	learn: 8.3292600	test: 11.5533293	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.5s
    3590:	learn: 8.3288094	test: 11.5535490	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.5s
    3591:	learn: 8.3286668	test: 11.5536542	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.5s
    3592:	learn: 8.3283741	test: 11.5529487	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.4s
    3593:	learn: 8.3277830	test: 11.5535138	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.4s
    3594:	learn: 8.3275060	test: 11.5526033	best: 11.5434600 (3247)	total: 1m 49s	remaining: 12.4s
    3595:	learn: 8.3268457	test: 11.5521698	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.4s
    3596:	learn: 8.3264428	test: 11.5502717	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.3s
    3597:	learn: 8.3260530	test: 11.5495925	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.3s
    3598:	learn: 8.3261045	test: 11.5512193	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.3s
    3599:	learn: 8.3259591	test: 11.5506171	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.2s
    3600:	learn: 8.3258525	test: 11.5512530	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.2s
    3601:	learn: 8.3233905	test: 11.5467316	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.2s
    3602:	learn: 8.3225000	test: 11.5464913	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.1s
    3603:	learn: 8.3215606	test: 11.5463319	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.1s
    3604:	learn: 8.3213341	test: 11.5460224	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.1s
    3605:	learn: 8.3208156	test: 11.5462293	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12.1s
    3606:	learn: 8.3200688	test: 11.5463305	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12s
    3607:	learn: 8.3182899	test: 11.5439427	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12s
    3608:	learn: 8.3181077	test: 11.5457012	best: 11.5434600 (3247)	total: 1m 50s	remaining: 12s
    3609:	learn: 8.3180124	test: 11.5463436	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.9s
    3610:	learn: 8.3172469	test: 11.5475720	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.9s
    3611:	learn: 8.3170360	test: 11.5491120	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.9s
    3612:	learn: 8.3167276	test: 11.5483255	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.9s
    3613:	learn: 8.3160337	test: 11.5474223	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.8s
    3614:	learn: 8.3158385	test: 11.5494668	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.8s
    3615:	learn: 8.3145193	test: 11.5482232	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.8s
    3616:	learn: 8.3144909	test: 11.5477521	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.7s
    3617:	learn: 8.3138599	test: 11.5460556	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.7s
    3618:	learn: 8.3135740	test: 11.5446860	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.7s
    3619:	learn: 8.3130822	test: 11.5452824	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.6s
    3620:	learn: 8.3122373	test: 11.5447044	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.6s
    3621:	learn: 8.3118228	test: 11.5448787	best: 11.5434600 (3247)	total: 1m 50s	remaining: 11.6s
    3622:	learn: 8.3111904	test: 11.5447679	best: 11.5434600 (3247)	total: 1m 51s	remaining: 11.6s
    3623:	learn: 8.3107277	test: 11.5452399	best: 11.5434600 (3247)	total: 1m 51s	remaining: 11.5s
    3624:	learn: 8.3106047	test: 11.5432925	best: 11.5432925 (3624)	total: 1m 51s	remaining: 11.5s
    3625:	learn: 8.3099533	test: 11.5428881	best: 11.5428881 (3625)	total: 1m 51s	remaining: 11.5s
    3626:	learn: 8.3091700	test: 11.5405031	best: 11.5405031 (3626)	total: 1m 51s	remaining: 11.4s
    3627:	learn: 8.3083766	test: 11.5400127	best: 11.5400127 (3627)	total: 1m 51s	remaining: 11.4s
    3628:	learn: 8.3078772	test: 11.5392800	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.4s
    3629:	learn: 8.3066810	test: 11.5406200	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.3s
    3630:	learn: 8.3059761	test: 11.5417637	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.3s
    3631:	learn: 8.3048323	test: 11.5424672	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.3s
    3632:	learn: 8.3039793	test: 11.5428863	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.2s
    3633:	learn: 8.3027382	test: 11.5422922	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.2s
    3634:	learn: 8.3017794	test: 11.5426960	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.2s
    3635:	learn: 8.3012761	test: 11.5434358	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.2s
    3636:	learn: 8.3008323	test: 11.5432946	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.1s
    3637:	learn: 8.2998720	test: 11.5430736	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.1s
    3638:	learn: 8.2991157	test: 11.5412575	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11.1s
    3639:	learn: 8.2982188	test: 11.5397682	best: 11.5392800 (3628)	total: 1m 51s	remaining: 11s
    3640:	learn: 8.2956260	test: 11.5366708	best: 11.5366708 (3640)	total: 1m 51s	remaining: 11s
    3641:	learn: 8.2950071	test: 11.5369093	best: 11.5366708 (3640)	total: 1m 51s	remaining: 11s
    3642:	learn: 8.2933063	test: 11.5361274	best: 11.5361274 (3642)	total: 1m 51s	remaining: 10.9s
    3643:	learn: 8.2924522	test: 11.5371006	best: 11.5361274 (3642)	total: 1m 51s	remaining: 10.9s
    3644:	learn: 8.2924723	test: 11.5367731	best: 11.5361274 (3642)	total: 1m 51s	remaining: 10.9s
    3645:	learn: 8.2910616	test: 11.5373800	best: 11.5361274 (3642)	total: 1m 51s	remaining: 10.9s
    3646:	learn: 8.2904800	test: 11.5369385	best: 11.5361274 (3642)	total: 1m 51s	remaining: 10.8s
    3647:	learn: 8.2900055	test: 11.5363072	best: 11.5361274 (3642)	total: 1m 51s	remaining: 10.8s
    3648:	learn: 8.2899032	test: 11.5355656	best: 11.5355656 (3648)	total: 1m 51s	remaining: 10.8s
    3649:	learn: 8.2882360	test: 11.5363655	best: 11.5355656 (3648)	total: 1m 51s	remaining: 10.7s
    3650:	learn: 8.2880388	test: 11.5370774	best: 11.5355656 (3648)	total: 1m 51s	remaining: 10.7s
    3651:	learn: 8.2875800	test: 11.5363826	best: 11.5355656 (3648)	total: 1m 52s	remaining: 10.7s
    3652:	learn: 8.2870008	test: 11.5356267	best: 11.5355656 (3648)	total: 1m 52s	remaining: 10.6s
    3653:	learn: 8.2865388	test: 11.5359869	best: 11.5355656 (3648)	total: 1m 52s	remaining: 10.6s
    3654:	learn: 8.2856384	test: 11.5344361	best: 11.5344361 (3654)	total: 1m 52s	remaining: 10.6s
    3655:	learn: 8.2855305	test: 11.5354320	best: 11.5344361 (3654)	total: 1m 52s	remaining: 10.6s
    3656:	learn: 8.2849520	test: 11.5341181	best: 11.5341181 (3656)	total: 1m 52s	remaining: 10.5s
    3657:	learn: 8.2850230	test: 11.5337964	best: 11.5337964 (3657)	total: 1m 52s	remaining: 10.5s
    3658:	learn: 8.2842346	test: 11.5334729	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.5s
    3659:	learn: 8.2830935	test: 11.5340064	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.4s
    3660:	learn: 8.2816638	test: 11.5349128	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.4s
    3661:	learn: 8.2808793	test: 11.5362595	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.4s
    3662:	learn: 8.2798626	test: 11.5372718	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.3s
    3663:	learn: 8.2794979	test: 11.5358369	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.3s
    3664:	learn: 8.2788862	test: 11.5351276	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.3s
    3665:	learn: 8.2785592	test: 11.5348724	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.3s
    3666:	learn: 8.2779044	test: 11.5351299	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.2s
    3667:	learn: 8.2777775	test: 11.5341812	best: 11.5334729 (3658)	total: 1m 52s	remaining: 10.2s
    3668:	learn: 8.2772897	test: 11.5329885	best: 11.5329885 (3668)	total: 1m 52s	remaining: 10.2s
    3669:	learn: 8.2774224	test: 11.5323062	best: 11.5323062 (3669)	total: 1m 52s	remaining: 10.1s
    3670:	learn: 8.2771425	test: 11.5339157	best: 11.5323062 (3669)	total: 1m 52s	remaining: 10.1s
    3671:	learn: 8.2766962	test: 11.5350798	best: 11.5323062 (3669)	total: 1m 52s	remaining: 10.1s
    3672:	learn: 8.2758351	test: 11.5350521	best: 11.5323062 (3669)	total: 1m 52s	remaining: 10s
    3673:	learn: 8.2757712	test: 11.5353394	best: 11.5323062 (3669)	total: 1m 52s	remaining: 10s
    3674:	learn: 8.2752002	test: 11.5348315	best: 11.5323062 (3669)	total: 1m 52s	remaining: 9.98s
    3675:	learn: 8.2744786	test: 11.5355751	best: 11.5323062 (3669)	total: 1m 52s	remaining: 9.95s
    3676:	learn: 8.2744160	test: 11.5346338	best: 11.5323062 (3669)	total: 1m 52s	remaining: 9.92s
    3677:	learn: 8.2724978	test: 11.5346938	best: 11.5323062 (3669)	total: 1m 52s	remaining: 9.89s
    3678:	learn: 8.2718981	test: 11.5340926	best: 11.5323062 (3669)	total: 1m 52s	remaining: 9.86s
    3679:	learn: 8.2712783	test: 11.5340227	best: 11.5323062 (3669)	total: 1m 52s	remaining: 9.82s
    3680:	learn: 8.2713620	test: 11.5327535	best: 11.5323062 (3669)	total: 1m 53s	remaining: 9.79s
    3681:	learn: 8.2709566	test: 11.5307959	best: 11.5307959 (3681)	total: 1m 53s	remaining: 9.76s
    3682:	learn: 8.2706688	test: 11.5298993	best: 11.5298993 (3682)	total: 1m 53s	remaining: 9.73s
    3683:	learn: 8.2692902	test: 11.5290631	best: 11.5290631 (3683)	total: 1m 53s	remaining: 9.7s
    3684:	learn: 8.2682109	test: 11.5281228	best: 11.5281228 (3684)	total: 1m 53s	remaining: 9.67s
    3685:	learn: 8.2683994	test: 11.5278654	best: 11.5278654 (3685)	total: 1m 53s	remaining: 9.64s
    3686:	learn: 8.2683648	test: 11.5286280	best: 11.5278654 (3685)	total: 1m 53s	remaining: 9.61s
    3687:	learn: 8.2675777	test: 11.5276949	best: 11.5276949 (3687)	total: 1m 53s	remaining: 9.58s
    3688:	learn: 8.2671288	test: 11.5283387	best: 11.5276949 (3687)	total: 1m 53s	remaining: 9.55s
    3689:	learn: 8.2666333	test: 11.5277718	best: 11.5276949 (3687)	total: 1m 53s	remaining: 9.52s
    3690:	learn: 8.2666946	test: 11.5279699	best: 11.5276949 (3687)	total: 1m 53s	remaining: 9.49s
    3691:	learn: 8.2664420	test: 11.5278135	best: 11.5276949 (3687)	total: 1m 53s	remaining: 9.46s
    3692:	learn: 8.2659624	test: 11.5274667	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.44s
    3693:	learn: 8.2645942	test: 11.5294274	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.41s
    3694:	learn: 8.2644907	test: 11.5298113	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.38s
    3695:	learn: 8.2640737	test: 11.5299612	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.35s
    3696:	learn: 8.2632559	test: 11.5321525	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.32s
    3697:	learn: 8.2627675	test: 11.5316594	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.29s
    3698:	learn: 8.2629765	test: 11.5313066	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.26s
    3699:	learn: 8.2618904	test: 11.5307023	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.22s
    3700:	learn: 8.2613726	test: 11.5286931	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.2s
    3701:	learn: 8.2612209	test: 11.5292188	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.16s
    3702:	learn: 8.2605168	test: 11.5288236	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.13s
    3703:	learn: 8.2603393	test: 11.5317842	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.1s
    3704:	learn: 8.2600618	test: 11.5314792	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.07s
    3705:	learn: 8.2592314	test: 11.5305678	best: 11.5274667 (3692)	total: 1m 53s	remaining: 9.04s
    3706:	learn: 8.2584594	test: 11.5303393	best: 11.5274667 (3692)	total: 1m 54s	remaining: 9.01s
    3707:	learn: 8.2571492	test: 11.5294738	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.98s
    3708:	learn: 8.2564446	test: 11.5297771	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.95s
    3709:	learn: 8.2562810	test: 11.5299973	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.92s
    3710:	learn: 8.2560131	test: 11.5306608	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.89s
    3711:	learn: 8.2548194	test: 11.5318981	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.86s
    3712:	learn: 8.2546505	test: 11.5322387	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.83s
    3713:	learn: 8.2539157	test: 11.5306913	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.8s
    3714:	learn: 8.2535570	test: 11.5309254	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.77s
    3715:	learn: 8.2532463	test: 11.5303992	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.73s
    3716:	learn: 8.2521261	test: 11.5318731	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.7s
    3717:	learn: 8.2511952	test: 11.5320534	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.67s
    3718:	learn: 8.2490947	test: 11.5320580	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.64s
    3719:	learn: 8.2487351	test: 11.5320603	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.61s
    3720:	learn: 8.2480964	test: 11.5314347	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.58s
    3721:	learn: 8.2476282	test: 11.5304711	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.55s
    3722:	learn: 8.2471040	test: 11.5310985	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.52s
    3723:	learn: 8.2465165	test: 11.5307427	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.49s
    3724:	learn: 8.2457971	test: 11.5304060	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.46s
    3725:	learn: 8.2456023	test: 11.5281230	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.43s
    3726:	learn: 8.2454777	test: 11.5280777	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.4s
    3727:	learn: 8.2438548	test: 11.5310494	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.37s
    3728:	learn: 8.2435353	test: 11.5295607	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.34s
    3729:	learn: 8.2423793	test: 11.5313057	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.3s
    3730:	learn: 8.2415172	test: 11.5306060	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.28s
    3731:	learn: 8.2401902	test: 11.5296454	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.24s
    3732:	learn: 8.2390170	test: 11.5294344	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.22s
    3733:	learn: 8.2384325	test: 11.5305751	best: 11.5274667 (3692)	total: 1m 54s	remaining: 8.19s
    3734:	learn: 8.2372045	test: 11.5287537	best: 11.5274667 (3692)	total: 1m 55s	remaining: 8.16s
    3735:	learn: 8.2366183	test: 11.5294333	best: 11.5274667 (3692)	total: 1m 55s	remaining: 8.13s
    3736:	learn: 8.2361760	test: 11.5306373	best: 11.5274667 (3692)	total: 1m 55s	remaining: 8.1s
    3737:	learn: 8.2360212	test: 11.5299051	best: 11.5274667 (3692)	total: 1m 55s	remaining: 8.07s
    3738:	learn: 8.2351938	test: 11.5302918	best: 11.5274667 (3692)	total: 1m 55s	remaining: 8.04s
    3739:	learn: 8.2344725	test: 11.5283236	best: 11.5274667 (3692)	total: 1m 55s	remaining: 8.01s
    3740:	learn: 8.2338403	test: 11.5274892	best: 11.5274667 (3692)	total: 1m 55s	remaining: 7.98s
    3741:	learn: 8.2335356	test: 11.5280573	best: 11.5274667 (3692)	total: 1m 55s	remaining: 7.95s
    3742:	learn: 8.2327189	test: 11.5272238	best: 11.5272238 (3742)	total: 1m 55s	remaining: 7.92s
    3743:	learn: 8.2326283	test: 11.5281652	best: 11.5272238 (3742)	total: 1m 55s	remaining: 7.89s
    3744:	learn: 8.2321116	test: 11.5276790	best: 11.5272238 (3742)	total: 1m 55s	remaining: 7.86s
    3745:	learn: 8.2310003	test: 11.5289840	best: 11.5272238 (3742)	total: 1m 55s	remaining: 7.83s
    3746:	learn: 8.2301556	test: 11.5270825	best: 11.5270825 (3746)	total: 1m 55s	remaining: 7.8s
    3747:	learn: 8.2291538	test: 11.5278790	best: 11.5270825 (3746)	total: 1m 55s	remaining: 7.77s
    3748:	learn: 8.2288943	test: 11.5264797	best: 11.5264797 (3748)	total: 1m 55s	remaining: 7.74s
    3749:	learn: 8.2287049	test: 11.5269801	best: 11.5264797 (3748)	total: 1m 55s	remaining: 7.71s
    3750:	learn: 8.2277817	test: 11.5254638	best: 11.5254638 (3750)	total: 1m 55s	remaining: 7.68s
    3751:	learn: 8.2269330	test: 11.5244011	best: 11.5244011 (3751)	total: 1m 55s	remaining: 7.65s
    3752:	learn: 8.2256761	test: 11.5225927	best: 11.5225927 (3752)	total: 1m 55s	remaining: 7.62s
    3753:	learn: 8.2250145	test: 11.5216424	best: 11.5216424 (3753)	total: 1m 55s	remaining: 7.59s
    3754:	learn: 8.2244899	test: 11.5215962	best: 11.5215962 (3754)	total: 1m 55s	remaining: 7.56s
    3755:	learn: 8.2231069	test: 11.5202259	best: 11.5202259 (3755)	total: 1m 55s	remaining: 7.53s
    3756:	learn: 8.2225213	test: 11.5216477	best: 11.5202259 (3755)	total: 1m 55s	remaining: 7.5s
    3757:	learn: 8.2222733	test: 11.5215795	best: 11.5202259 (3755)	total: 1m 55s	remaining: 7.46s
    3758:	learn: 8.2222525	test: 11.5218329	best: 11.5202259 (3755)	total: 1m 55s	remaining: 7.43s
    3759:	learn: 8.2214674	test: 11.5226595	best: 11.5202259 (3755)	total: 1m 55s	remaining: 7.4s
    3760:	learn: 8.2207932	test: 11.5230730	best: 11.5202259 (3755)	total: 1m 56s	remaining: 7.37s
    3761:	learn: 8.2201944	test: 11.5214685	best: 11.5202259 (3755)	total: 1m 56s	remaining: 7.34s
    3762:	learn: 8.2194983	test: 11.5226035	best: 11.5202259 (3755)	total: 1m 56s	remaining: 7.31s
    3763:	learn: 8.2190134	test: 11.5213531	best: 11.5202259 (3755)	total: 1m 56s	remaining: 7.28s
    3764:	learn: 8.2180441	test: 11.5217223	best: 11.5202259 (3755)	total: 1m 56s	remaining: 7.25s
    3765:	learn: 8.2175967	test: 11.5205009	best: 11.5202259 (3755)	total: 1m 56s	remaining: 7.22s
    3766:	learn: 8.2171860	test: 11.5203561	best: 11.5202259 (3755)	total: 1m 56s	remaining: 7.19s
    3767:	learn: 8.2168955	test: 11.5205219	best: 11.5202259 (3755)	total: 1m 56s	remaining: 7.16s
    3768:	learn: 8.2163348	test: 11.5198476	best: 11.5198476 (3768)	total: 1m 56s	remaining: 7.13s
    3769:	learn: 8.2159515	test: 11.5192424	best: 11.5192424 (3769)	total: 1m 56s	remaining: 7.1s
    3770:	learn: 8.2158509	test: 11.5196800	best: 11.5192424 (3769)	total: 1m 56s	remaining: 7.07s
    3771:	learn: 8.2152150	test: 11.5199700	best: 11.5192424 (3769)	total: 1m 56s	remaining: 7.04s
    3772:	learn: 8.2142401	test: 11.5209035	best: 11.5192424 (3769)	total: 1m 56s	remaining: 7.01s
    3773:	learn: 8.2137944	test: 11.5196737	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.98s
    3774:	learn: 8.2137628	test: 11.5195966	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.95s
    3775:	learn: 8.2133260	test: 11.5209144	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.92s
    3776:	learn: 8.2131713	test: 11.5202721	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.88s
    3777:	learn: 8.2122937	test: 11.5202817	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.85s
    3778:	learn: 8.2114208	test: 11.5205816	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.82s
    3779:	learn: 8.2104321	test: 11.5218572	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.79s
    3780:	learn: 8.2102023	test: 11.5234929	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.76s
    3781:	learn: 8.2102463	test: 11.5232412	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.73s
    3782:	learn: 8.2102218	test: 11.5237664	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.7s
    3783:	learn: 8.2093100	test: 11.5229976	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.67s
    3784:	learn: 8.2080989	test: 11.5236048	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.64s
    3785:	learn: 8.2072107	test: 11.5213500	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.61s
    3786:	learn: 8.2063626	test: 11.5213925	best: 11.5192424 (3769)	total: 1m 56s	remaining: 6.58s
    3787:	learn: 8.2057187	test: 11.5187953	best: 11.5187953 (3787)	total: 1m 56s	remaining: 6.54s
    3788:	learn: 8.2049761	test: 11.5207919	best: 11.5187953 (3787)	total: 1m 56s	remaining: 6.51s
    3789:	learn: 8.2044655	test: 11.5184416	best: 11.5184416 (3789)	total: 1m 57s	remaining: 6.48s
    3790:	learn: 8.2034784	test: 11.5175025	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.45s
    3791:	learn: 8.2029479	test: 11.5186367	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.42s
    3792:	learn: 8.2021554	test: 11.5197055	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.39s
    3793:	learn: 8.2015897	test: 11.5202967	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.36s
    3794:	learn: 8.2013140	test: 11.5194182	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.33s
    3795:	learn: 8.2007452	test: 11.5192811	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.3s
    3796:	learn: 8.2003378	test: 11.5197423	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.27s
    3797:	learn: 8.1999692	test: 11.5196965	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.24s
    3798:	learn: 8.1998321	test: 11.5181663	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.21s
    3799:	learn: 8.1997647	test: 11.5185667	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.17s
    3800:	learn: 8.1990401	test: 11.5213767	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.14s
    3801:	learn: 8.1988023	test: 11.5221656	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.11s
    3802:	learn: 8.1979757	test: 11.5218864	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.08s
    3803:	learn: 8.1975642	test: 11.5211717	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.05s
    3804:	learn: 8.1973977	test: 11.5213715	best: 11.5175025 (3790)	total: 1m 57s	remaining: 6.02s
    3805:	learn: 8.1968630	test: 11.5223518	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.99s
    3806:	learn: 8.1964847	test: 11.5223652	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.96s
    3807:	learn: 8.1964519	test: 11.5236427	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.93s
    3808:	learn: 8.1957605	test: 11.5244405	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.9s
    3809:	learn: 8.1952339	test: 11.5256199	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.87s
    3810:	learn: 8.1932225	test: 11.5260155	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.83s
    3811:	learn: 8.1924683	test: 11.5254718	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.8s
    3812:	learn: 8.1930674	test: 11.5275667	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.77s
    3813:	learn: 8.1930036	test: 11.5274370	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.74s
    3814:	learn: 8.1921976	test: 11.5278792	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.71s
    3815:	learn: 8.1917542	test: 11.5284562	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.68s
    3816:	learn: 8.1917502	test: 11.5285521	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.65s
    3817:	learn: 8.1916828	test: 11.5265517	best: 11.5175025 (3790)	total: 1m 57s	remaining: 5.62s
    3818:	learn: 8.1913446	test: 11.5263365	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.59s
    3819:	learn: 8.1911665	test: 11.5270907	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.56s
    3820:	learn: 8.1901895	test: 11.5259320	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.53s
    3821:	learn: 8.1895714	test: 11.5259214	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.5s
    3822:	learn: 8.1888005	test: 11.5237023	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.47s
    3823:	learn: 8.1885013	test: 11.5241355	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.44s
    3824:	learn: 8.1877430	test: 11.5234005	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.41s
    3825:	learn: 8.1868939	test: 11.5233697	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.38s
    3826:	learn: 8.1864570	test: 11.5239828	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.35s
    3827:	learn: 8.1863305	test: 11.5229423	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.32s
    3828:	learn: 8.1860515	test: 11.5241885	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.29s
    3829:	learn: 8.1854251	test: 11.5238438	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.26s
    3830:	learn: 8.1838582	test: 11.5214388	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.23s
    3831:	learn: 8.1825084	test: 11.5188024	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.2s
    3832:	learn: 8.1817457	test: 11.5186304	best: 11.5175025 (3790)	total: 1m 58s	remaining: 5.17s
    3833:	learn: 8.1808479	test: 11.5174158	best: 11.5174158 (3833)	total: 1m 58s	remaining: 5.13s
    3834:	learn: 8.1808042	test: 11.5167696	best: 11.5167696 (3834)	total: 1m 58s	remaining: 5.1s
    3835:	learn: 8.1801876	test: 11.5165997	best: 11.5165997 (3835)	total: 1m 58s	remaining: 5.07s
    3836:	learn: 8.1796443	test: 11.5166621	best: 11.5165997 (3835)	total: 1m 58s	remaining: 5.04s
    3837:	learn: 8.1781454	test: 11.5158208	best: 11.5158208 (3837)	total: 1m 58s	remaining: 5.01s
    3838:	learn: 8.1780692	test: 11.5177640	best: 11.5158208 (3837)	total: 1m 58s	remaining: 4.98s
    3839:	learn: 8.1768708	test: 11.5179520	best: 11.5158208 (3837)	total: 1m 58s	remaining: 4.95s
    3840:	learn: 8.1771366	test: 11.5182990	best: 11.5158208 (3837)	total: 1m 58s	remaining: 4.92s
    3841:	learn: 8.1764429	test: 11.5182957	best: 11.5158208 (3837)	total: 1m 58s	remaining: 4.89s
    3842:	learn: 8.1758551	test: 11.5181888	best: 11.5158208 (3837)	total: 1m 58s	remaining: 4.86s
    3843:	learn: 8.1753075	test: 11.5182537	best: 11.5158208 (3837)	total: 1m 58s	remaining: 4.82s
    3844:	learn: 8.1740256	test: 11.5167632	best: 11.5158208 (3837)	total: 1m 58s	remaining: 4.79s
    3845:	learn: 8.1735671	test: 11.5172507	best: 11.5158208 (3837)	total: 1m 58s	remaining: 4.76s
    3846:	learn: 8.1727569	test: 11.5162739	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.73s
    3847:	learn: 8.1726530	test: 11.5161682	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.7s
    3848:	learn: 8.1722173	test: 11.5183795	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.67s
    3849:	learn: 8.1707026	test: 11.5165001	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.64s
    3850:	learn: 8.1706218	test: 11.5170459	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.61s
    3851:	learn: 8.1696146	test: 11.5174196	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.58s
    3852:	learn: 8.1691647	test: 11.5161194	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.55s
    3853:	learn: 8.1687224	test: 11.5175910	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.52s
    3854:	learn: 8.1696086	test: 11.5166724	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.49s
    3855:	learn: 8.1692698	test: 11.5167631	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.46s
    3856:	learn: 8.1686716	test: 11.5164294	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.42s
    3857:	learn: 8.1682897	test: 11.5165481	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.39s
    3858:	learn: 8.1668687	test: 11.5179263	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.36s
    3859:	learn: 8.1661436	test: 11.5173463	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.33s
    3860:	learn: 8.1649707	test: 11.5183781	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.3s
    3861:	learn: 8.1639804	test: 11.5171901	best: 11.5158208 (3837)	total: 1m 59s	remaining: 4.27s
    3862:	learn: 8.1644971	test: 11.5152827	best: 11.5152827 (3862)	total: 1m 59s	remaining: 4.24s
    3863:	learn: 8.1641226	test: 11.5155255	best: 11.5152827 (3862)	total: 1m 59s	remaining: 4.21s
    3864:	learn: 8.1636825	test: 11.5157200	best: 11.5152827 (3862)	total: 1m 59s	remaining: 4.18s
    3865:	learn: 8.1621210	test: 11.5153434	best: 11.5152827 (3862)	total: 1m 59s	remaining: 4.15s
    3866:	learn: 8.1616819	test: 11.5138205	best: 11.5138205 (3866)	total: 1m 59s	remaining: 4.12s
    3867:	learn: 8.1615090	test: 11.5136505	best: 11.5136505 (3867)	total: 1m 59s	remaining: 4.09s
    3868:	learn: 8.1609568	test: 11.5142794	best: 11.5136505 (3867)	total: 1m 59s	remaining: 4.06s
    3869:	learn: 8.1608911	test: 11.5153602	best: 11.5136505 (3867)	total: 1m 59s	remaining: 4.03s
    3870:	learn: 8.1605506	test: 11.5112688	best: 11.5112688 (3870)	total: 1m 59s	remaining: 4s
    3871:	learn: 8.1596339	test: 11.5131419	best: 11.5112688 (3870)	total: 2m	remaining: 3.97s
    3872:	learn: 8.1594832	test: 11.5131559	best: 11.5112688 (3870)	total: 2m	remaining: 3.94s
    3873:	learn: 8.1586337	test: 11.5147892	best: 11.5112688 (3870)	total: 2m	remaining: 3.91s
    3874:	learn: 8.1572506	test: 11.5152416	best: 11.5112688 (3870)	total: 2m	remaining: 3.88s
    3875:	learn: 8.1565965	test: 11.5158512	best: 11.5112688 (3870)	total: 2m	remaining: 3.85s
    3876:	learn: 8.1557749	test: 11.5159356	best: 11.5112688 (3870)	total: 2m	remaining: 3.82s
    3877:	learn: 8.1550070	test: 11.5170067	best: 11.5112688 (3870)	total: 2m	remaining: 3.79s
    3878:	learn: 8.1544407	test: 11.5153181	best: 11.5112688 (3870)	total: 2m	remaining: 3.76s
    3879:	learn: 8.1536382	test: 11.5143111	best: 11.5112688 (3870)	total: 2m	remaining: 3.73s
    3880:	learn: 8.1526189	test: 11.5150068	best: 11.5112688 (3870)	total: 2m	remaining: 3.7s
    3881:	learn: 8.1499316	test: 11.5141733	best: 11.5112688 (3870)	total: 2m	remaining: 3.67s
    3882:	learn: 8.1492014	test: 11.5144805	best: 11.5112688 (3870)	total: 2m	remaining: 3.63s
    3883:	learn: 8.1491745	test: 11.5169110	best: 11.5112688 (3870)	total: 2m	remaining: 3.6s
    3884:	learn: 8.1489083	test: 11.5156063	best: 11.5112688 (3870)	total: 2m	remaining: 3.57s
    3885:	learn: 8.1494768	test: 11.5161990	best: 11.5112688 (3870)	total: 2m	remaining: 3.54s
    3886:	learn: 8.1490262	test: 11.5152660	best: 11.5112688 (3870)	total: 2m	remaining: 3.51s
    3887:	learn: 8.1487933	test: 11.5160133	best: 11.5112688 (3870)	total: 2m	remaining: 3.48s
    3888:	learn: 8.1488017	test: 11.5168384	best: 11.5112688 (3870)	total: 2m	remaining: 3.45s
    3889:	learn: 8.1483584	test: 11.5162715	best: 11.5112688 (3870)	total: 2m	remaining: 3.42s
    3890:	learn: 8.1480830	test: 11.5180557	best: 11.5112688 (3870)	total: 2m	remaining: 3.39s
    3891:	learn: 8.1472417	test: 11.5164010	best: 11.5112688 (3870)	total: 2m	remaining: 3.36s
    3892:	learn: 8.1470479	test: 11.5158520	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.33s
    3893:	learn: 8.1461608	test: 11.5166241	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.3s
    3894:	learn: 8.1453425	test: 11.5158058	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.27s
    3895:	learn: 8.1445062	test: 11.5186720	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.24s
    3896:	learn: 8.1438348	test: 11.5178488	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.21s
    3897:	learn: 8.1433828	test: 11.5191924	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.18s
    3898:	learn: 8.1430574	test: 11.5191312	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.15s
    3899:	learn: 8.1425469	test: 11.5203420	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.11s
    3900:	learn: 8.1411098	test: 11.5218375	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.08s
    3901:	learn: 8.1404999	test: 11.5228616	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.05s
    3902:	learn: 8.1401416	test: 11.5235537	best: 11.5112688 (3870)	total: 2m 1s	remaining: 3.02s
    3903:	learn: 8.1397232	test: 11.5231032	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.99s
    3904:	learn: 8.1392954	test: 11.5237004	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.96s
    3905:	learn: 8.1386156	test: 11.5227652	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.93s
    3906:	learn: 8.1383631	test: 11.5211079	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.9s
    3907:	learn: 8.1377904	test: 11.5204026	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.87s
    3908:	learn: 8.1376048	test: 11.5202977	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.83s
    3909:	learn: 8.1359996	test: 11.5194612	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.8s
    3910:	learn: 8.1355857	test: 11.5221306	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.77s
    3911:	learn: 8.1354369	test: 11.5219916	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.74s
    3912:	learn: 8.1349749	test: 11.5222713	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.71s
    3913:	learn: 8.1348012	test: 11.5242224	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.68s
    3914:	learn: 8.1342067	test: 11.5244630	best: 11.5112688 (3870)	total: 2m 1s	remaining: 2.65s
    3915:	learn: 8.1339622	test: 11.5240492	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.62s
    3916:	learn: 8.1332501	test: 11.5240454	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.59s
    3917:	learn: 8.1325697	test: 11.5232518	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.56s
    3918:	learn: 8.1311860	test: 11.5245949	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.52s
    3919:	learn: 8.1313044	test: 11.5218258	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.49s
    3920:	learn: 8.1309582	test: 11.5224655	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.46s
    3921:	learn: 8.1311100	test: 11.5218474	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.43s
    3922:	learn: 8.1304102	test: 11.5219019	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.4s
    3923:	learn: 8.1307135	test: 11.5223740	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.37s
    3924:	learn: 8.1298212	test: 11.5216772	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.34s
    3925:	learn: 8.1295616	test: 11.5221853	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.31s
    3926:	learn: 8.1292231	test: 11.5238661	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.28s
    3927:	learn: 8.1286235	test: 11.5226102	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.25s
    3928:	learn: 8.1281305	test: 11.5217622	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.21s
    3929:	learn: 8.1274711	test: 11.5208967	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.18s
    3930:	learn: 8.1263521	test: 11.5229305	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.15s
    3931:	learn: 8.1250052	test: 11.5214905	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.12s
    3932:	learn: 8.1247335	test: 11.5221654	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.09s
    3933:	learn: 8.1248169	test: 11.5225175	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.06s
    3934:	learn: 8.1242930	test: 11.5207150	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2.03s
    3935:	learn: 8.1237320	test: 11.5233398	best: 11.5112688 (3870)	total: 2m 2s	remaining: 2s
    3936:	learn: 8.1237618	test: 11.5223192	best: 11.5112688 (3870)	total: 2m 2s	remaining: 1.97s
    3937:	learn: 8.1233206	test: 11.5232514	best: 11.5112688 (3870)	total: 2m 2s	remaining: 1.93s
    3938:	learn: 8.1229411	test: 11.5213576	best: 11.5112688 (3870)	total: 2m 2s	remaining: 1.9s
    3939:	learn: 8.1225261	test: 11.5225368	best: 11.5112688 (3870)	total: 2m 2s	remaining: 1.87s
    3940:	learn: 8.1221735	test: 11.5232820	best: 11.5112688 (3870)	total: 2m 2s	remaining: 1.84s
    3941:	learn: 8.1220531	test: 11.5242522	best: 11.5112688 (3870)	total: 2m 2s	remaining: 1.81s
    3942:	learn: 8.1203483	test: 11.5297506	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.78s
    3943:	learn: 8.1200701	test: 11.5303761	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.75s
    3944:	learn: 8.1186908	test: 11.5280401	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.72s
    3945:	learn: 8.1172181	test: 11.5260624	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.68s
    3946:	learn: 8.1161051	test: 11.5247929	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.65s
    3947:	learn: 8.1147559	test: 11.5234046	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.62s
    3948:	learn: 8.1145394	test: 11.5216241	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.59s
    3949:	learn: 8.1141572	test: 11.5209616	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.56s
    3950:	learn: 8.1133074	test: 11.5208841	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.53s
    3951:	learn: 8.1131292	test: 11.5209789	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.5s
    3952:	learn: 8.1128530	test: 11.5221776	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.47s
    3953:	learn: 8.1125307	test: 11.5212195	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.44s
    3954:	learn: 8.1117913	test: 11.5221211	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.4s
    3955:	learn: 8.1113319	test: 11.5220957	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.37s
    3956:	learn: 8.1106918	test: 11.5228249	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.34s
    3957:	learn: 8.1105849	test: 11.5220751	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.31s
    3958:	learn: 8.1099246	test: 11.5228692	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.28s
    3959:	learn: 8.1093353	test: 11.5209573	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.25s
    3960:	learn: 8.1078986	test: 11.5215817	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.22s
    3961:	learn: 8.1071416	test: 11.5205058	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.19s
    3962:	learn: 8.1072492	test: 11.5198781	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.15s
    3963:	learn: 8.1067403	test: 11.5203131	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.12s
    3964:	learn: 8.1061034	test: 11.5204167	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.09s
    3965:	learn: 8.1055828	test: 11.5215761	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.06s
    3966:	learn: 8.1053810	test: 11.5216175	best: 11.5112688 (3870)	total: 2m 3s	remaining: 1.03s
    3967:	learn: 8.1037782	test: 11.5210514	best: 11.5112688 (3870)	total: 2m 3s	remaining: 999ms
    3968:	learn: 8.1032712	test: 11.5216833	best: 11.5112688 (3870)	total: 2m 3s	remaining: 967ms
    3969:	learn: 8.1025782	test: 11.5206997	best: 11.5112688 (3870)	total: 2m 3s	remaining: 936ms
    3970:	learn: 8.1017622	test: 11.5202077	best: 11.5112688 (3870)	total: 2m 3s	remaining: 905ms
    3971:	learn: 8.1013684	test: 11.5211960	best: 11.5112688 (3870)	total: 2m 3s	remaining: 874ms
    3972:	learn: 8.1009040	test: 11.5203589	best: 11.5112688 (3870)	total: 2m 3s	remaining: 843ms
    3973:	learn: 8.1009620	test: 11.5209295	best: 11.5112688 (3870)	total: 2m 4s	remaining: 811ms
    3974:	learn: 8.1014057	test: 11.5213306	best: 11.5112688 (3870)	total: 2m 4s	remaining: 780ms
    3975:	learn: 8.1015861	test: 11.5218243	best: 11.5112688 (3870)	total: 2m 4s	remaining: 749ms
    3976:	learn: 8.1009425	test: 11.5219690	best: 11.5112688 (3870)	total: 2m 4s	remaining: 718ms
    3977:	learn: 8.0998046	test: 11.5228195	best: 11.5112688 (3870)	total: 2m 4s	remaining: 687ms
    3978:	learn: 8.0993137	test: 11.5225706	best: 11.5112688 (3870)	total: 2m 4s	remaining: 656ms
    3979:	learn: 8.0983502	test: 11.5213304	best: 11.5112688 (3870)	total: 2m 4s	remaining: 624ms
    3980:	learn: 8.0976323	test: 11.5225754	best: 11.5112688 (3870)	total: 2m 4s	remaining: 593ms
    3981:	learn: 8.0980843	test: 11.5224619	best: 11.5112688 (3870)	total: 2m 4s	remaining: 562ms
    3982:	learn: 8.0977891	test: 11.5227509	best: 11.5112688 (3870)	total: 2m 4s	remaining: 531ms
    3983:	learn: 8.0973981	test: 11.5224614	best: 11.5112688 (3870)	total: 2m 4s	remaining: 499ms
    3984:	learn: 8.0958411	test: 11.5230466	best: 11.5112688 (3870)	total: 2m 4s	remaining: 468ms
    3985:	learn: 8.0955985	test: 11.5209659	best: 11.5112688 (3870)	total: 2m 4s	remaining: 437ms
    3986:	learn: 8.0952687	test: 11.5228027	best: 11.5112688 (3870)	total: 2m 4s	remaining: 406ms
    3987:	learn: 8.0944473	test: 11.5216375	best: 11.5112688 (3870)	total: 2m 4s	remaining: 375ms
    3988:	learn: 8.0933831	test: 11.5225299	best: 11.5112688 (3870)	total: 2m 4s	remaining: 343ms
    3989:	learn: 8.0928824	test: 11.5210245	best: 11.5112688 (3870)	total: 2m 4s	remaining: 312ms
    3990:	learn: 8.0923356	test: 11.5217343	best: 11.5112688 (3870)	total: 2m 4s	remaining: 281ms
    3991:	learn: 8.0922034	test: 11.5203219	best: 11.5112688 (3870)	total: 2m 4s	remaining: 250ms
    3992:	learn: 8.0910522	test: 11.5202558	best: 11.5112688 (3870)	total: 2m 4s	remaining: 218ms
    3993:	learn: 8.0911943	test: 11.5211061	best: 11.5112688 (3870)	total: 2m 4s	remaining: 187ms
    3994:	learn: 8.0905014	test: 11.5194805	best: 11.5112688 (3870)	total: 2m 4s	remaining: 156ms
    3995:	learn: 8.0895106	test: 11.5182308	best: 11.5112688 (3870)	total: 2m 4s	remaining: 125ms
    3996:	learn: 8.0893076	test: 11.5209533	best: 11.5112688 (3870)	total: 2m 4s	remaining: 93.6ms
    3997:	learn: 8.0884477	test: 11.5223004	best: 11.5112688 (3870)	total: 2m 4s	remaining: 62.4ms
    3998:	learn: 8.0877300	test: 11.5211664	best: 11.5112688 (3870)	total: 2m 4s	remaining: 31.2ms
    3999:	learn: 8.0874401	test: 11.5223206	best: 11.5112688 (3870)	total: 2m 4s	remaining: 0us
    
    bestTest = 11.51126882
    bestIteration = 3870
    
    Shrink model to first 3871 iterations.
    




    <catboost.core.CatBoostRegressor at 0x24bff40b390>




```python
# observe result
y_pred = model.predict(X_test)
print('Mean Absolute Error:', MAE(y_test, y_pred))
```

    Mean Absolute Error: 15.20918354570364
    


```python
# observe features importance
pd.DataFrame(data={'features':X_train.columns, 'importance':model.feature_importances_}).plot.bar(x='features', y='importance')
plt.show()
```


![png](output_27_0.png)


### Step 5: train the model again using all training data and do prediction for the 'submit' set


```python
# train test split for train, and cv
X_train, X_cv, y_train, y_cv = train_test_split(training.drop(['total_cases'], axis=1), training.total_cases, test_size=0.3, stratify=training.city, random_state=123)
X_train.drop('city', axis=1, inplace=True)
X_cv.drop('city', axis=1, inplace=True)
```


```python
# build model
model = CatBoostRegressor(iterations = 4000, learning_rate = 0.3, loss_function='MAE', eval_metric='MAE', use_best_model=True, random_seed=123)
model.fit(X_train, y_train, eval_set=(X_cv, y_cv), verbose=1)
```

    0:	learn: 24.1974352	test: 25.3920835	best: 25.3920835 (0)	total: 30.3ms	remaining: 2m 1s
    1:	learn: 24.0744886	test: 25.2775173	best: 25.2775173 (1)	total: 62.2ms	remaining: 2m 4s
    2:	learn: 23.9612315	test: 25.1710107	best: 25.1710107 (2)	total: 96.5ms	remaining: 2m 8s
    3:	learn: 23.8469576	test: 25.0655092	best: 25.0655092 (3)	total: 129ms	remaining: 2m 9s
    4:	learn: 23.7201477	test: 24.9454754	best: 24.9454754 (4)	total: 161ms	remaining: 2m 8s
    5:	learn: 23.6047111	test: 24.8404418	best: 24.8404418 (5)	total: 197ms	remaining: 2m 11s
    6:	learn: 23.4851525	test: 24.7287017	best: 24.7287017 (6)	total: 231ms	remaining: 2m 11s
    7:	learn: 23.3732312	test: 24.6276916	best: 24.6276916 (7)	total: 273ms	remaining: 2m 16s
    8:	learn: 23.2610199	test: 24.5244938	best: 24.5244938 (8)	total: 331ms	remaining: 2m 26s
    9:	learn: 23.1545849	test: 24.4284554	best: 24.4284554 (9)	total: 373ms	remaining: 2m 28s
    10:	learn: 23.0505472	test: 24.3327020	best: 24.3327020 (10)	total: 402ms	remaining: 2m 25s
    11:	learn: 22.9446640	test: 24.2380613	best: 24.2380613 (11)	total: 431ms	remaining: 2m 23s
    12:	learn: 22.8419089	test: 24.1430098	best: 24.1430098 (12)	total: 462ms	remaining: 2m 21s
    13:	learn: 22.7221708	test: 24.0330756	best: 24.0330756 (13)	total: 496ms	remaining: 2m 21s
    14:	learn: 22.6158110	test: 23.9329844	best: 23.9329844 (14)	total: 547ms	remaining: 2m 25s
    15:	learn: 22.5186415	test: 23.8420159	best: 23.8420159 (15)	total: 574ms	remaining: 2m 22s
    16:	learn: 22.4141415	test: 23.7445928	best: 23.7445928 (16)	total: 608ms	remaining: 2m 22s
    17:	learn: 22.3201290	test: 23.6575542	best: 23.6575542 (17)	total: 641ms	remaining: 2m 21s
    18:	learn: 22.2171665	test: 23.5594839	best: 23.5594839 (18)	total: 674ms	remaining: 2m 21s
    19:	learn: 22.1195457	test: 23.4660380	best: 23.4660380 (19)	total: 708ms	remaining: 2m 20s
    20:	learn: 22.0293964	test: 23.3812739	best: 23.3812739 (20)	total: 756ms	remaining: 2m 23s
    21:	learn: 21.9406537	test: 23.3000640	best: 23.3000640 (21)	total: 804ms	remaining: 2m 25s
    22:	learn: 21.8602828	test: 23.2259891	best: 23.2259891 (22)	total: 831ms	remaining: 2m 23s
    23:	learn: 21.7736869	test: 23.1438481	best: 23.1438481 (23)	total: 853ms	remaining: 2m 21s
    24:	learn: 21.6967338	test: 23.0724525	best: 23.0724525 (24)	total: 883ms	remaining: 2m 20s
    25:	learn: 21.6083822	test: 22.9897271	best: 22.9897271 (25)	total: 903ms	remaining: 2m 17s
    26:	learn: 21.5219241	test: 22.9106644	best: 22.9106644 (26)	total: 939ms	remaining: 2m 18s
    27:	learn: 21.4362518	test: 22.8290871	best: 22.8290871 (27)	total: 973ms	remaining: 2m 18s
    28:	learn: 21.3501517	test: 22.7477627	best: 22.7477627 (28)	total: 1.02s	remaining: 2m 19s
    29:	learn: 21.2692638	test: 22.6723902	best: 22.6723902 (29)	total: 1.05s	remaining: 2m 19s
    30:	learn: 21.1883034	test: 22.5987102	best: 22.5987102 (30)	total: 1.08s	remaining: 2m 18s
    31:	learn: 21.1077496	test: 22.5237985	best: 22.5237985 (31)	total: 1.12s	remaining: 2m 18s
    32:	learn: 21.0315807	test: 22.4551048	best: 22.4551048 (32)	total: 1.15s	remaining: 2m 17s
    33:	learn: 20.9577275	test: 22.3888543	best: 22.3888543 (33)	total: 1.19s	remaining: 2m 18s
    34:	learn: 20.8833803	test: 22.3220133	best: 22.3220133 (34)	total: 1.24s	remaining: 2m 20s
    35:	learn: 20.8118308	test: 22.2551278	best: 22.2551278 (35)	total: 1.28s	remaining: 2m 20s
    36:	learn: 20.7419876	test: 22.1916369	best: 22.1916369 (36)	total: 1.31s	remaining: 2m 20s
    37:	learn: 20.6716735	test: 22.1269821	best: 22.1269821 (37)	total: 1.34s	remaining: 2m 19s
    38:	learn: 20.6043029	test: 22.0645237	best: 22.0645237 (38)	total: 1.38s	remaining: 2m 19s
    39:	learn: 20.5362380	test: 22.0018901	best: 22.0018901 (39)	total: 1.42s	remaining: 2m 20s
    40:	learn: 20.4672885	test: 21.9390492	best: 21.9390492 (40)	total: 1.46s	remaining: 2m 20s
    41:	learn: 20.3992311	test: 21.8732779	best: 21.8732779 (41)	total: 1.49s	remaining: 2m 20s
    42:	learn: 20.3296652	test: 21.8032167	best: 21.8032167 (42)	total: 1.52s	remaining: 2m 19s
    43:	learn: 20.2676228	test: 21.7437832	best: 21.7437832 (43)	total: 1.54s	remaining: 2m 18s
    44:	learn: 20.2037348	test: 21.6843494	best: 21.6843494 (44)	total: 1.57s	remaining: 2m 18s
    45:	learn: 20.1439542	test: 21.6265105	best: 21.6265105 (45)	total: 1.61s	remaining: 2m 18s
    46:	learn: 20.0844584	test: 21.5696778	best: 21.5696778 (46)	total: 1.67s	remaining: 2m 20s
    47:	learn: 20.0256425	test: 21.5154220	best: 21.5154220 (47)	total: 1.7s	remaining: 2m 20s
    48:	learn: 19.9644574	test: 21.4529853	best: 21.4529853 (48)	total: 1.73s	remaining: 2m 19s
    49:	learn: 19.9089086	test: 21.3984330	best: 21.3984330 (49)	total: 1.76s	remaining: 2m 19s
    50:	learn: 19.8537828	test: 21.3430484	best: 21.3430484 (50)	total: 1.79s	remaining: 2m 18s
    51:	learn: 19.8000751	test: 21.2933473	best: 21.2933473 (51)	total: 1.83s	remaining: 2m 19s
    52:	learn: 19.7345829	test: 21.2289547	best: 21.2289547 (52)	total: 1.88s	remaining: 2m 19s
    53:	learn: 19.6777161	test: 21.1718551	best: 21.1718551 (53)	total: 1.91s	remaining: 2m 19s
    54:	learn: 19.6200089	test: 21.1140484	best: 21.1140484 (54)	total: 1.94s	remaining: 2m 19s
    55:	learn: 19.5584751	test: 21.0518247	best: 21.0518247 (55)	total: 1.97s	remaining: 2m 18s
    56:	learn: 19.5081277	test: 21.0025211	best: 21.0025211 (56)	total: 2s	remaining: 2m 18s
    57:	learn: 19.4509899	test: 20.9438417	best: 20.9438417 (57)	total: 2.03s	remaining: 2m 18s
    58:	learn: 19.4026453	test: 20.8972208	best: 20.8972208 (58)	total: 2.08s	remaining: 2m 18s
    59:	learn: 19.3514672	test: 20.8485491	best: 20.8485491 (59)	total: 2.11s	remaining: 2m 18s
    60:	learn: 19.3002047	test: 20.8018402	best: 20.8018402 (60)	total: 2.14s	remaining: 2m 18s
    61:	learn: 19.2485737	test: 20.7506545	best: 20.7506545 (61)	total: 2.17s	remaining: 2m 17s
    62:	learn: 19.1962760	test: 20.7009920	best: 20.7009920 (62)	total: 2.2s	remaining: 2m 17s
    63:	learn: 19.1472546	test: 20.6525081	best: 20.6525081 (63)	total: 2.24s	remaining: 2m 17s
    64:	learn: 19.0917432	test: 20.5930828	best: 20.5930828 (64)	total: 2.28s	remaining: 2m 17s
    65:	learn: 19.0466864	test: 20.5465838	best: 20.5465838 (65)	total: 2.32s	remaining: 2m 18s
    66:	learn: 19.0003488	test: 20.5015554	best: 20.5015554 (66)	total: 2.36s	remaining: 2m 18s
    67:	learn: 18.9535410	test: 20.4596769	best: 20.4596769 (67)	total: 2.4s	remaining: 2m 18s
    68:	learn: 18.9091457	test: 20.4170006	best: 20.4170006 (68)	total: 2.43s	remaining: 2m 18s
    69:	learn: 18.8614658	test: 20.3698612	best: 20.3698612 (69)	total: 2.47s	remaining: 2m 18s
    70:	learn: 18.8191155	test: 20.3259499	best: 20.3259499 (70)	total: 2.51s	remaining: 2m 18s
    71:	learn: 18.7730133	test: 20.2787129	best: 20.2787129 (71)	total: 2.54s	remaining: 2m 18s
    72:	learn: 18.7266615	test: 20.2334583	best: 20.2334583 (72)	total: 2.58s	remaining: 2m 18s
    73:	learn: 18.6850006	test: 20.1930364	best: 20.1930364 (73)	total: 2.62s	remaining: 2m 18s
    74:	learn: 18.6398913	test: 20.1489799	best: 20.1489799 (74)	total: 2.65s	remaining: 2m 18s
    75:	learn: 18.6030786	test: 20.1107316	best: 20.1107316 (75)	total: 2.69s	remaining: 2m 18s
    76:	learn: 18.5674521	test: 20.0723837	best: 20.0723837 (76)	total: 2.73s	remaining: 2m 19s
    77:	learn: 18.5300475	test: 20.0338151	best: 20.0338151 (77)	total: 2.77s	remaining: 2m 19s
    78:	learn: 18.4929438	test: 19.9936323	best: 19.9936323 (78)	total: 2.8s	remaining: 2m 19s
    79:	learn: 18.4536943	test: 19.9545206	best: 19.9545206 (79)	total: 2.83s	remaining: 2m 18s
    80:	learn: 18.4100290	test: 19.9114358	best: 19.9114358 (80)	total: 2.86s	remaining: 2m 18s
    81:	learn: 18.3708429	test: 19.8702270	best: 19.8702270 (81)	total: 2.9s	remaining: 2m 18s
    82:	learn: 18.3330562	test: 19.8315590	best: 19.8315590 (82)	total: 2.93s	remaining: 2m 18s
    83:	learn: 18.2873643	test: 19.7882933	best: 19.7882933 (83)	total: 2.97s	remaining: 2m 18s
    84:	learn: 18.2449996	test: 19.7462817	best: 19.7462817 (84)	total: 3.01s	remaining: 2m 18s
    85:	learn: 18.2061418	test: 19.7086833	best: 19.7086833 (85)	total: 3.04s	remaining: 2m 18s
    86:	learn: 18.1644548	test: 19.6690363	best: 19.6690363 (86)	total: 3.07s	remaining: 2m 18s
    87:	learn: 18.1225045	test: 19.6277149	best: 19.6277149 (87)	total: 3.1s	remaining: 2m 18s
    88:	learn: 18.0789280	test: 19.5887375	best: 19.5887375 (88)	total: 3.15s	remaining: 2m 18s
    89:	learn: 18.0399832	test: 19.5484200	best: 19.5484200 (89)	total: 3.19s	remaining: 2m 18s
    90:	learn: 17.9963940	test: 19.5091850	best: 19.5091850 (90)	total: 3.23s	remaining: 2m 18s
    91:	learn: 17.9575682	test: 19.4714058	best: 19.4714058 (91)	total: 3.27s	remaining: 2m 18s
    92:	learn: 17.9143427	test: 19.4310400	best: 19.4310400 (92)	total: 3.3s	remaining: 2m 18s
    93:	learn: 17.8747958	test: 19.3960916	best: 19.3960916 (93)	total: 3.34s	remaining: 2m 18s
    94:	learn: 17.8406954	test: 19.3613216	best: 19.3613216 (94)	total: 3.39s	remaining: 2m 19s
    95:	learn: 17.8072769	test: 19.3269652	best: 19.3269652 (95)	total: 3.42s	remaining: 2m 19s
    96:	learn: 17.7683909	test: 19.2875523	best: 19.2875523 (96)	total: 3.46s	remaining: 2m 19s
    97:	learn: 17.7264961	test: 19.2490890	best: 19.2490890 (97)	total: 3.49s	remaining: 2m 18s
    98:	learn: 17.6903672	test: 19.2148055	best: 19.2148055 (98)	total: 3.52s	remaining: 2m 18s
    99:	learn: 17.6539863	test: 19.1792789	best: 19.1792789 (99)	total: 3.55s	remaining: 2m 18s
    100:	learn: 17.6202863	test: 19.1507177	best: 19.1507177 (100)	total: 3.59s	remaining: 2m 18s
    101:	learn: 17.5854204	test: 19.1169710	best: 19.1169710 (101)	total: 3.63s	remaining: 2m 18s
    102:	learn: 17.5506719	test: 19.0862155	best: 19.0862155 (102)	total: 3.66s	remaining: 2m 18s
    103:	learn: 17.5185499	test: 19.0553225	best: 19.0553225 (103)	total: 3.7s	remaining: 2m 18s
    104:	learn: 17.4888674	test: 19.0238881	best: 19.0238881 (104)	total: 3.73s	remaining: 2m 18s
    105:	learn: 17.4511397	test: 18.9862621	best: 18.9862621 (105)	total: 3.77s	remaining: 2m 18s
    106:	learn: 17.4147299	test: 18.9532638	best: 18.9532638 (106)	total: 3.81s	remaining: 2m 18s
    107:	learn: 17.3778895	test: 18.9236704	best: 18.9236704 (107)	total: 3.85s	remaining: 2m 18s
    108:	learn: 17.3410807	test: 18.8894186	best: 18.8894186 (108)	total: 3.9s	remaining: 2m 19s
    109:	learn: 17.3084205	test: 18.8629130	best: 18.8629130 (109)	total: 3.93s	remaining: 2m 18s
    110:	learn: 17.2768609	test: 18.8278159	best: 18.8278159 (110)	total: 3.96s	remaining: 2m 18s
    111:	learn: 17.2420567	test: 18.7953573	best: 18.7953573 (111)	total: 3.99s	remaining: 2m 18s
    112:	learn: 17.2066622	test: 18.7617763	best: 18.7617763 (112)	total: 4.03s	remaining: 2m 18s
    113:	learn: 17.1718539	test: 18.7292780	best: 18.7292780 (113)	total: 4.07s	remaining: 2m 18s
    114:	learn: 17.1359212	test: 18.6949909	best: 18.6949909 (114)	total: 4.11s	remaining: 2m 18s
    115:	learn: 17.1026046	test: 18.6561374	best: 18.6561374 (115)	total: 4.14s	remaining: 2m 18s
    116:	learn: 17.0682570	test: 18.6262162	best: 18.6262162 (116)	total: 4.17s	remaining: 2m 18s
    117:	learn: 17.0347143	test: 18.5979974	best: 18.5979974 (117)	total: 4.21s	remaining: 2m 18s
    118:	learn: 17.0037512	test: 18.5711224	best: 18.5711224 (118)	total: 4.25s	remaining: 2m 18s
    119:	learn: 16.9702038	test: 18.5447792	best: 18.5447792 (119)	total: 4.29s	remaining: 2m 18s
    120:	learn: 16.9363698	test: 18.5163564	best: 18.5163564 (120)	total: 4.33s	remaining: 2m 18s
    121:	learn: 16.9080729	test: 18.4913832	best: 18.4913832 (121)	total: 4.36s	remaining: 2m 18s
    122:	learn: 16.8810363	test: 18.4635202	best: 18.4635202 (122)	total: 4.4s	remaining: 2m 18s
    123:	learn: 16.8494479	test: 18.4324321	best: 18.4324321 (123)	total: 4.43s	remaining: 2m 18s
    124:	learn: 16.8146084	test: 18.4007778	best: 18.4007778 (124)	total: 4.47s	remaining: 2m 18s
    125:	learn: 16.7799137	test: 18.3698758	best: 18.3698758 (125)	total: 4.5s	remaining: 2m 18s
    126:	learn: 16.7511278	test: 18.3441815	best: 18.3441815 (126)	total: 4.54s	remaining: 2m 18s
    127:	learn: 16.7199166	test: 18.3180187	best: 18.3180187 (127)	total: 4.57s	remaining: 2m 18s
    128:	learn: 16.6949677	test: 18.2956567	best: 18.2956567 (128)	total: 4.6s	remaining: 2m 18s
    129:	learn: 16.6661811	test: 18.2678152	best: 18.2678152 (129)	total: 4.64s	remaining: 2m 18s
    130:	learn: 16.6352473	test: 18.2410473	best: 18.2410473 (130)	total: 4.67s	remaining: 2m 18s
    131:	learn: 16.6041911	test: 18.2165548	best: 18.2165548 (131)	total: 4.72s	remaining: 2m 18s
    132:	learn: 16.5763016	test: 18.1876488	best: 18.1876488 (132)	total: 4.78s	remaining: 2m 19s
    133:	learn: 16.5467551	test: 18.1616056	best: 18.1616056 (133)	total: 4.83s	remaining: 2m 19s
    134:	learn: 16.5192372	test: 18.1342658	best: 18.1342658 (134)	total: 4.89s	remaining: 2m 20s
    135:	learn: 16.4891896	test: 18.1108521	best: 18.1108521 (135)	total: 4.95s	remaining: 2m 20s
    136:	learn: 16.4616620	test: 18.0841978	best: 18.0841978 (136)	total: 5s	remaining: 2m 20s
    137:	learn: 16.4375324	test: 18.0618043	best: 18.0618043 (137)	total: 5.04s	remaining: 2m 20s
    138:	learn: 16.4126299	test: 18.0411048	best: 18.0411048 (138)	total: 5.07s	remaining: 2m 20s
    139:	learn: 16.3826663	test: 18.0146872	best: 18.0146872 (139)	total: 5.11s	remaining: 2m 21s
    140:	learn: 16.3585307	test: 17.9908658	best: 17.9908658 (140)	total: 5.16s	remaining: 2m 21s
    141:	learn: 16.3277269	test: 17.9708954	best: 17.9708954 (141)	total: 5.19s	remaining: 2m 21s
    142:	learn: 16.2988829	test: 17.9465854	best: 17.9465854 (142)	total: 5.23s	remaining: 2m 20s
    143:	learn: 16.2704657	test: 17.9247257	best: 17.9247257 (143)	total: 5.26s	remaining: 2m 20s
    144:	learn: 16.2495408	test: 17.9041840	best: 17.9041840 (144)	total: 5.3s	remaining: 2m 20s
    145:	learn: 16.2236885	test: 17.8777245	best: 17.8777245 (145)	total: 5.34s	remaining: 2m 20s
    146:	learn: 16.1982213	test: 17.8554772	best: 17.8554772 (146)	total: 5.37s	remaining: 2m 20s
    147:	learn: 16.1738792	test: 17.8334948	best: 17.8334948 (147)	total: 5.41s	remaining: 2m 20s
    148:	learn: 16.1441655	test: 17.8124058	best: 17.8124058 (148)	total: 5.44s	remaining: 2m 20s
    149:	learn: 16.1187206	test: 17.7936716	best: 17.7936716 (149)	total: 5.48s	remaining: 2m 20s
    150:	learn: 16.0886032	test: 17.7690421	best: 17.7690421 (150)	total: 5.51s	remaining: 2m 20s
    151:	learn: 16.0634972	test: 17.7443135	best: 17.7443135 (151)	total: 5.54s	remaining: 2m 20s
    152:	learn: 16.0377436	test: 17.7197187	best: 17.7197187 (152)	total: 5.58s	remaining: 2m 20s
    153:	learn: 16.0143504	test: 17.7013295	best: 17.7013295 (153)	total: 5.61s	remaining: 2m 20s
    154:	learn: 15.9863599	test: 17.6764293	best: 17.6764293 (154)	total: 5.65s	remaining: 2m 20s
    155:	learn: 15.9593284	test: 17.6514095	best: 17.6514095 (155)	total: 5.69s	remaining: 2m 20s
    156:	learn: 15.9304409	test: 17.6212940	best: 17.6212940 (156)	total: 5.72s	remaining: 2m 20s
    157:	learn: 15.9094393	test: 17.5983686	best: 17.5983686 (157)	total: 5.75s	remaining: 2m 19s
    158:	learn: 15.8855451	test: 17.5802549	best: 17.5802549 (158)	total: 5.79s	remaining: 2m 19s
    159:	learn: 15.8583678	test: 17.5599638	best: 17.5599638 (159)	total: 5.83s	remaining: 2m 19s
    160:	learn: 15.8351142	test: 17.5378747	best: 17.5378747 (160)	total: 5.86s	remaining: 2m 19s
    161:	learn: 15.8127778	test: 17.5202740	best: 17.5202740 (161)	total: 5.9s	remaining: 2m 19s
    162:	learn: 15.7910723	test: 17.5016981	best: 17.5016981 (162)	total: 5.93s	remaining: 2m 19s
    163:	learn: 15.7687252	test: 17.4802481	best: 17.4802481 (163)	total: 5.96s	remaining: 2m 19s
    164:	learn: 15.7458332	test: 17.4636237	best: 17.4636237 (164)	total: 6s	remaining: 2m 19s
    165:	learn: 15.7204983	test: 17.4461543	best: 17.4461543 (165)	total: 6.04s	remaining: 2m 19s
    166:	learn: 15.6974290	test: 17.4289284	best: 17.4289284 (166)	total: 6.08s	remaining: 2m 19s
    167:	learn: 15.6765402	test: 17.4141169	best: 17.4141169 (167)	total: 6.12s	remaining: 2m 19s
    168:	learn: 15.6526006	test: 17.3915374	best: 17.3915374 (168)	total: 6.15s	remaining: 2m 19s
    169:	learn: 15.6396371	test: 17.3789382	best: 17.3789382 (169)	total: 6.18s	remaining: 2m 19s
    170:	learn: 15.6216509	test: 17.3631421	best: 17.3631421 (170)	total: 6.22s	remaining: 2m 19s
    171:	learn: 15.5982283	test: 17.3427177	best: 17.3427177 (171)	total: 6.27s	remaining: 2m 19s
    172:	learn: 15.5724539	test: 17.3165871	best: 17.3165871 (172)	total: 6.3s	remaining: 2m 19s
    173:	learn: 15.5497921	test: 17.2980486	best: 17.2980486 (173)	total: 6.34s	remaining: 2m 19s
    174:	learn: 15.5257439	test: 17.2762286	best: 17.2762286 (174)	total: 6.37s	remaining: 2m 19s
    175:	learn: 15.5016382	test: 17.2546682	best: 17.2546682 (175)	total: 6.41s	remaining: 2m 19s
    176:	learn: 15.4764851	test: 17.2346058	best: 17.2346058 (176)	total: 6.45s	remaining: 2m 19s
    177:	learn: 15.4536956	test: 17.2154839	best: 17.2154839 (177)	total: 6.49s	remaining: 2m 19s
    178:	learn: 15.4335320	test: 17.1980206	best: 17.1980206 (178)	total: 6.53s	remaining: 2m 19s
    179:	learn: 15.4183203	test: 17.1837869	best: 17.1837869 (179)	total: 6.56s	remaining: 2m 19s
    180:	learn: 15.3881500	test: 17.1543521	best: 17.1543521 (180)	total: 6.6s	remaining: 2m 19s
    181:	learn: 15.3707284	test: 17.1413844	best: 17.1413844 (181)	total: 6.64s	remaining: 2m 19s
    182:	learn: 15.3458558	test: 17.1237944	best: 17.1237944 (182)	total: 6.67s	remaining: 2m 19s
    183:	learn: 15.3261718	test: 17.1074323	best: 17.1074323 (183)	total: 6.71s	remaining: 2m 19s
    184:	learn: 15.3089777	test: 17.0953402	best: 17.0953402 (184)	total: 6.75s	remaining: 2m 19s
    185:	learn: 15.2872848	test: 17.0774478	best: 17.0774478 (185)	total: 6.79s	remaining: 2m 19s
    186:	learn: 15.2624927	test: 17.0577768	best: 17.0577768 (186)	total: 6.85s	remaining: 2m 19s
    187:	learn: 15.2434304	test: 17.0445437	best: 17.0445437 (187)	total: 6.92s	remaining: 2m 20s
    188:	learn: 15.2333390	test: 17.0344447	best: 17.0344447 (188)	total: 6.99s	remaining: 2m 20s
    189:	learn: 15.2147664	test: 17.0173182	best: 17.0173182 (189)	total: 7.06s	remaining: 2m 21s
    190:	learn: 15.1956938	test: 17.0027418	best: 17.0027418 (190)	total: 7.12s	remaining: 2m 22s
    191:	learn: 15.1728505	test: 16.9829012	best: 16.9829012 (191)	total: 7.18s	remaining: 2m 22s
    192:	learn: 15.1571498	test: 16.9714818	best: 16.9714818 (192)	total: 7.25s	remaining: 2m 22s
    193:	learn: 15.1438892	test: 16.9588313	best: 16.9588313 (193)	total: 7.31s	remaining: 2m 23s
    194:	learn: 15.1242371	test: 16.9414273	best: 16.9414273 (194)	total: 7.38s	remaining: 2m 23s
    195:	learn: 15.1038777	test: 16.9262856	best: 16.9262856 (195)	total: 7.42s	remaining: 2m 24s
    196:	learn: 15.0874093	test: 16.9110685	best: 16.9110685 (196)	total: 7.46s	remaining: 2m 24s
    197:	learn: 15.0728713	test: 16.8970792	best: 16.8970792 (197)	total: 7.5s	remaining: 2m 23s
    198:	learn: 15.0533084	test: 16.8826067	best: 16.8826067 (198)	total: 7.53s	remaining: 2m 23s
    199:	learn: 15.0323784	test: 16.8682163	best: 16.8682163 (199)	total: 7.57s	remaining: 2m 23s
    200:	learn: 15.0136638	test: 16.8543991	best: 16.8543991 (200)	total: 7.6s	remaining: 2m 23s
    201:	learn: 14.9956552	test: 16.8370101	best: 16.8370101 (201)	total: 7.65s	remaining: 2m 23s
    202:	learn: 14.9777410	test: 16.8205129	best: 16.8205129 (202)	total: 7.71s	remaining: 2m 24s
    203:	learn: 14.9608259	test: 16.8087886	best: 16.8087886 (203)	total: 7.77s	remaining: 2m 24s
    204:	learn: 14.9415638	test: 16.7929997	best: 16.7929997 (204)	total: 7.82s	remaining: 2m 24s
    205:	learn: 14.9250556	test: 16.7795931	best: 16.7795931 (205)	total: 7.87s	remaining: 2m 24s
    206:	learn: 14.9043053	test: 16.7615751	best: 16.7615751 (206)	total: 7.92s	remaining: 2m 25s
    207:	learn: 14.8922196	test: 16.7514031	best: 16.7514031 (207)	total: 7.96s	remaining: 2m 25s
    208:	learn: 14.8748466	test: 16.7360959	best: 16.7360959 (208)	total: 8s	remaining: 2m 25s
    209:	learn: 14.8609378	test: 16.7245565	best: 16.7245565 (209)	total: 8.04s	remaining: 2m 25s
    210:	learn: 14.8474744	test: 16.7155554	best: 16.7155554 (210)	total: 8.07s	remaining: 2m 25s
    211:	learn: 14.8302325	test: 16.7002501	best: 16.7002501 (211)	total: 8.11s	remaining: 2m 24s
    212:	learn: 14.8106394	test: 16.6799345	best: 16.6799345 (212)	total: 8.15s	remaining: 2m 24s
    213:	learn: 14.7953558	test: 16.6640330	best: 16.6640330 (213)	total: 8.18s	remaining: 2m 24s
    214:	learn: 14.7823946	test: 16.6540978	best: 16.6540978 (214)	total: 8.22s	remaining: 2m 24s
    215:	learn: 14.7644861	test: 16.6367532	best: 16.6367532 (215)	total: 8.26s	remaining: 2m 24s
    216:	learn: 14.7500873	test: 16.6242064	best: 16.6242064 (216)	total: 8.3s	remaining: 2m 24s
    217:	learn: 14.7390633	test: 16.6163336	best: 16.6163336 (217)	total: 8.34s	remaining: 2m 24s
    218:	learn: 14.7257224	test: 16.6050741	best: 16.6050741 (218)	total: 8.39s	remaining: 2m 24s
    219:	learn: 14.7146150	test: 16.5968067	best: 16.5968067 (219)	total: 8.45s	remaining: 2m 25s
    220:	learn: 14.6998803	test: 16.5838612	best: 16.5838612 (220)	total: 8.51s	remaining: 2m 25s
    221:	learn: 14.6864073	test: 16.5727396	best: 16.5727396 (221)	total: 8.56s	remaining: 2m 25s
    222:	learn: 14.6743525	test: 16.5637584	best: 16.5637584 (222)	total: 8.6s	remaining: 2m 25s
    223:	learn: 14.6592536	test: 16.5525062	best: 16.5525062 (223)	total: 8.65s	remaining: 2m 25s
    224:	learn: 14.6459684	test: 16.5410528	best: 16.5410528 (224)	total: 8.68s	remaining: 2m 25s
    225:	learn: 14.6311734	test: 16.5288831	best: 16.5288831 (225)	total: 8.71s	remaining: 2m 25s
    226:	learn: 14.6160129	test: 16.5154121	best: 16.5154121 (226)	total: 8.75s	remaining: 2m 25s
    227:	learn: 14.6021226	test: 16.5010850	best: 16.5010850 (227)	total: 8.78s	remaining: 2m 25s
    228:	learn: 14.5928432	test: 16.4917125	best: 16.4917125 (228)	total: 8.8s	remaining: 2m 24s
    229:	learn: 14.5856289	test: 16.4847522	best: 16.4847522 (229)	total: 8.83s	remaining: 2m 24s
    230:	learn: 14.5755321	test: 16.4777491	best: 16.4777491 (230)	total: 8.86s	remaining: 2m 24s
    231:	learn: 14.5597399	test: 16.4627471	best: 16.4627471 (231)	total: 8.88s	remaining: 2m 24s
    232:	learn: 14.5433552	test: 16.4477134	best: 16.4477134 (232)	total: 8.91s	remaining: 2m 24s
    233:	learn: 14.5263059	test: 16.4315115	best: 16.4315115 (233)	total: 8.94s	remaining: 2m 23s
    234:	learn: 14.5143859	test: 16.4190554	best: 16.4190554 (234)	total: 8.98s	remaining: 2m 23s
    235:	learn: 14.5006165	test: 16.4076579	best: 16.4076579 (235)	total: 9.01s	remaining: 2m 23s
    236:	learn: 14.4911172	test: 16.3981497	best: 16.3981497 (236)	total: 9.04s	remaining: 2m 23s
    237:	learn: 14.4742248	test: 16.3847595	best: 16.3847595 (237)	total: 9.07s	remaining: 2m 23s
    238:	learn: 14.4603089	test: 16.3748863	best: 16.3748863 (238)	total: 9.1s	remaining: 2m 23s
    239:	learn: 14.4448852	test: 16.3636842	best: 16.3636842 (239)	total: 9.13s	remaining: 2m 23s
    240:	learn: 14.4318204	test: 16.3504019	best: 16.3504019 (240)	total: 9.18s	remaining: 2m 23s
    241:	learn: 14.4224802	test: 16.3406379	best: 16.3406379 (241)	total: 9.21s	remaining: 2m 23s
    242:	learn: 14.4100527	test: 16.3303356	best: 16.3303356 (242)	total: 9.25s	remaining: 2m 22s
    243:	learn: 14.3977045	test: 16.3173248	best: 16.3173248 (243)	total: 9.28s	remaining: 2m 22s
    244:	learn: 14.3855551	test: 16.3067116	best: 16.3067116 (244)	total: 9.31s	remaining: 2m 22s
    245:	learn: 14.3768182	test: 16.2976544	best: 16.2976544 (245)	total: 9.35s	remaining: 2m 22s
    246:	learn: 14.3644291	test: 16.2874322	best: 16.2874322 (246)	total: 9.39s	remaining: 2m 22s
    247:	learn: 14.3563903	test: 16.2799971	best: 16.2799971 (247)	total: 9.42s	remaining: 2m 22s
    248:	learn: 14.3438447	test: 16.2691822	best: 16.2691822 (248)	total: 9.45s	remaining: 2m 22s
    249:	learn: 14.3333335	test: 16.2581412	best: 16.2581412 (249)	total: 9.47s	remaining: 2m 22s
    250:	learn: 14.3206114	test: 16.2465625	best: 16.2465625 (250)	total: 9.5s	remaining: 2m 21s
    251:	learn: 14.3056211	test: 16.2364387	best: 16.2364387 (251)	total: 9.53s	remaining: 2m 21s
    252:	learn: 14.2930569	test: 16.2227445	best: 16.2227445 (252)	total: 9.55s	remaining: 2m 21s
    253:	learn: 14.2841330	test: 16.2161365	best: 16.2161365 (253)	total: 9.59s	remaining: 2m 21s
    254:	learn: 14.2756263	test: 16.2078384	best: 16.2078384 (254)	total: 9.62s	remaining: 2m 21s
    255:	learn: 14.2631196	test: 16.1959364	best: 16.1959364 (255)	total: 9.64s	remaining: 2m 21s
    256:	learn: 14.2509701	test: 16.1857053	best: 16.1857053 (256)	total: 9.67s	remaining: 2m 20s
    257:	learn: 14.2449039	test: 16.1798503	best: 16.1798503 (257)	total: 9.7s	remaining: 2m 20s
    258:	learn: 14.2332004	test: 16.1733125	best: 16.1733125 (258)	total: 9.73s	remaining: 2m 20s
    259:	learn: 14.2235361	test: 16.1652646	best: 16.1652646 (259)	total: 9.76s	remaining: 2m 20s
    260:	learn: 14.2140379	test: 16.1570163	best: 16.1570163 (260)	total: 9.81s	remaining: 2m 20s
    261:	learn: 14.2037841	test: 16.1485105	best: 16.1485105 (261)	total: 9.85s	remaining: 2m 20s
    262:	learn: 14.1939839	test: 16.1430542	best: 16.1430542 (262)	total: 9.89s	remaining: 2m 20s
    263:	learn: 14.1810257	test: 16.1344690	best: 16.1344690 (263)	total: 9.93s	remaining: 2m 20s
    264:	learn: 14.1724235	test: 16.1280356	best: 16.1280356 (264)	total: 9.97s	remaining: 2m 20s
    265:	learn: 14.1627673	test: 16.1222139	best: 16.1222139 (265)	total: 10s	remaining: 2m 20s
    266:	learn: 14.1531020	test: 16.1167833	best: 16.1167833 (266)	total: 10s	remaining: 2m 20s
    267:	learn: 14.1457002	test: 16.1097812	best: 16.1097812 (267)	total: 10.1s	remaining: 2m 20s
    268:	learn: 14.1342511	test: 16.1025595	best: 16.1025595 (268)	total: 10.1s	remaining: 2m 19s
    269:	learn: 14.1300316	test: 16.0971004	best: 16.0971004 (269)	total: 10.1s	remaining: 2m 19s
    270:	learn: 14.1189873	test: 16.0877122	best: 16.0877122 (270)	total: 10.2s	remaining: 2m 19s
    271:	learn: 14.1094901	test: 16.0805673	best: 16.0805673 (271)	total: 10.2s	remaining: 2m 19s
    272:	learn: 14.1008230	test: 16.0740232	best: 16.0740232 (272)	total: 10.2s	remaining: 2m 19s
    273:	learn: 14.0960713	test: 16.0685506	best: 16.0685506 (273)	total: 10.3s	remaining: 2m 19s
    274:	learn: 14.0896625	test: 16.0632638	best: 16.0632638 (274)	total: 10.3s	remaining: 2m 19s
    275:	learn: 14.0808139	test: 16.0590277	best: 16.0590277 (275)	total: 10.3s	remaining: 2m 19s
    276:	learn: 14.0724736	test: 16.0527767	best: 16.0527767 (276)	total: 10.4s	remaining: 2m 19s
    277:	learn: 14.0628844	test: 16.0446603	best: 16.0446603 (277)	total: 10.4s	remaining: 2m 19s
    278:	learn: 14.0582790	test: 16.0395079	best: 16.0395079 (278)	total: 10.4s	remaining: 2m 19s
    279:	learn: 14.0469560	test: 16.0306884	best: 16.0306884 (279)	total: 10.5s	remaining: 2m 18s
    280:	learn: 14.0387250	test: 16.0214788	best: 16.0214788 (280)	total: 10.5s	remaining: 2m 18s
    281:	learn: 14.0335432	test: 16.0160223	best: 16.0160223 (281)	total: 10.5s	remaining: 2m 18s
    282:	learn: 14.0279298	test: 16.0141582	best: 16.0141582 (282)	total: 10.5s	remaining: 2m 18s
    283:	learn: 14.0175893	test: 16.0088580	best: 16.0088580 (283)	total: 10.6s	remaining: 2m 18s
    284:	learn: 14.0151256	test: 16.0048114	best: 16.0048114 (284)	total: 10.6s	remaining: 2m 18s
    285:	learn: 14.0062244	test: 15.9975527	best: 15.9975527 (285)	total: 10.6s	remaining: 2m 17s
    286:	learn: 13.9976795	test: 15.9895127	best: 15.9895127 (286)	total: 10.7s	remaining: 2m 17s
    287:	learn: 13.9905057	test: 15.9837564	best: 15.9837564 (287)	total: 10.7s	remaining: 2m 17s
    288:	learn: 13.9874061	test: 15.9801169	best: 15.9801169 (288)	total: 10.7s	remaining: 2m 17s
    289:	learn: 13.9753513	test: 15.9717146	best: 15.9717146 (289)	total: 10.7s	remaining: 2m 17s
    290:	learn: 13.9646802	test: 15.9660505	best: 15.9660505 (290)	total: 10.8s	remaining: 2m 17s
    291:	learn: 13.9508315	test: 15.9526234	best: 15.9526234 (291)	total: 10.8s	remaining: 2m 16s
    292:	learn: 13.9417485	test: 15.9420738	best: 15.9420738 (292)	total: 10.8s	remaining: 2m 16s
    293:	learn: 13.9311117	test: 15.9339339	best: 15.9339339 (293)	total: 10.8s	remaining: 2m 16s
    294:	learn: 13.9221674	test: 15.9245361	best: 15.9245361 (294)	total: 10.9s	remaining: 2m 16s
    295:	learn: 13.9184017	test: 15.9200973	best: 15.9200973 (295)	total: 10.9s	remaining: 2m 16s
    296:	learn: 13.9113358	test: 15.9143301	best: 15.9143301 (296)	total: 10.9s	remaining: 2m 16s
    297:	learn: 13.9006856	test: 15.9073221	best: 15.9073221 (297)	total: 11s	remaining: 2m 16s
    298:	learn: 13.8903507	test: 15.8984015	best: 15.8984015 (298)	total: 11s	remaining: 2m 16s
    299:	learn: 13.8826444	test: 15.8940695	best: 15.8940695 (299)	total: 11s	remaining: 2m 15s
    300:	learn: 13.8777708	test: 15.8922562	best: 15.8922562 (300)	total: 11.1s	remaining: 2m 15s
    301:	learn: 13.8646357	test: 15.8809898	best: 15.8809898 (301)	total: 11.1s	remaining: 2m 15s
    302:	learn: 13.8569737	test: 15.8727345	best: 15.8727345 (302)	total: 11.1s	remaining: 2m 15s
    303:	learn: 13.8523183	test: 15.8715868	best: 15.8715868 (303)	total: 11.2s	remaining: 2m 15s
    304:	learn: 13.8438308	test: 15.8661899	best: 15.8661899 (304)	total: 11.2s	remaining: 2m 15s
    305:	learn: 13.8393764	test: 15.8657167	best: 15.8657167 (305)	total: 11.2s	remaining: 2m 15s
    306:	learn: 13.8325062	test: 15.8605063	best: 15.8605063 (306)	total: 11.3s	remaining: 2m 15s
    307:	learn: 13.8281821	test: 15.8562997	best: 15.8562997 (307)	total: 11.3s	remaining: 2m 15s
    308:	learn: 13.8182851	test: 15.8476499	best: 15.8476499 (308)	total: 11.3s	remaining: 2m 15s
    309:	learn: 13.8060127	test: 15.8358475	best: 15.8358475 (309)	total: 11.4s	remaining: 2m 15s
    310:	learn: 13.7989984	test: 15.8299206	best: 15.8299206 (310)	total: 11.4s	remaining: 2m 15s
    311:	learn: 13.7907030	test: 15.8226731	best: 15.8226731 (311)	total: 11.4s	remaining: 2m 15s
    312:	learn: 13.7871630	test: 15.8207307	best: 15.8207307 (312)	total: 11.4s	remaining: 2m 14s
    313:	learn: 13.7738511	test: 15.8088639	best: 15.8088639 (313)	total: 11.5s	remaining: 2m 14s
    314:	learn: 13.7653099	test: 15.8064368	best: 15.8064368 (314)	total: 11.5s	remaining: 2m 14s
    315:	learn: 13.7614538	test: 15.8048559	best: 15.8048559 (315)	total: 11.5s	remaining: 2m 14s
    316:	learn: 13.7547200	test: 15.8014046	best: 15.8014046 (316)	total: 11.6s	remaining: 2m 14s
    317:	learn: 13.7463615	test: 15.7959139	best: 15.7959139 (317)	total: 11.6s	remaining: 2m 14s
    318:	learn: 13.7358219	test: 15.7879212	best: 15.7879212 (318)	total: 11.6s	remaining: 2m 14s
    319:	learn: 13.7296807	test: 15.7831654	best: 15.7831654 (319)	total: 11.6s	remaining: 2m 13s
    320:	learn: 13.7204407	test: 15.7759623	best: 15.7759623 (320)	total: 11.7s	remaining: 2m 13s
    321:	learn: 13.7164438	test: 15.7742829	best: 15.7742829 (321)	total: 11.7s	remaining: 2m 13s
    322:	learn: 13.7067672	test: 15.7650713	best: 15.7650713 (322)	total: 11.7s	remaining: 2m 13s
    323:	learn: 13.7040627	test: 15.7616701	best: 15.7616701 (323)	total: 11.8s	remaining: 2m 13s
    324:	learn: 13.6942917	test: 15.7535984	best: 15.7535984 (324)	total: 11.8s	remaining: 2m 13s
    325:	learn: 13.6805704	test: 15.7412978	best: 15.7412978 (325)	total: 11.8s	remaining: 2m 13s
    326:	learn: 13.6752805	test: 15.7395673	best: 15.7395673 (326)	total: 11.9s	remaining: 2m 13s
    327:	learn: 13.6735686	test: 15.7378163	best: 15.7378163 (327)	total: 11.9s	remaining: 2m 12s
    328:	learn: 13.6690518	test: 15.7337957	best: 15.7337957 (328)	total: 11.9s	remaining: 2m 12s
    329:	learn: 13.6650309	test: 15.7314748	best: 15.7314748 (329)	total: 11.9s	remaining: 2m 12s
    330:	learn: 13.6582867	test: 15.7286889	best: 15.7286889 (330)	total: 12s	remaining: 2m 12s
    331:	learn: 13.6527983	test: 15.7239679	best: 15.7239679 (331)	total: 12s	remaining: 2m 12s
    332:	learn: 13.6488853	test: 15.7213239	best: 15.7213239 (332)	total: 12s	remaining: 2m 12s
    333:	learn: 13.6390924	test: 15.7141410	best: 15.7141410 (333)	total: 12.1s	remaining: 2m 12s
    334:	learn: 13.6314980	test: 15.7087456	best: 15.7087456 (334)	total: 12.1s	remaining: 2m 12s
    335:	learn: 13.6284032	test: 15.7057345	best: 15.7057345 (335)	total: 12.2s	remaining: 2m 13s
    336:	learn: 13.6208091	test: 15.6988399	best: 15.6988399 (336)	total: 12.3s	remaining: 2m 13s
    337:	learn: 13.6176499	test: 15.6978951	best: 15.6978951 (337)	total: 12.3s	remaining: 2m 13s
    338:	learn: 13.6106847	test: 15.6918931	best: 15.6918931 (338)	total: 12.4s	remaining: 2m 13s
    339:	learn: 13.6050050	test: 15.6878519	best: 15.6878519 (339)	total: 12.5s	remaining: 2m 14s
    340:	learn: 13.5988248	test: 15.6843553	best: 15.6843553 (340)	total: 12.5s	remaining: 2m 14s
    341:	learn: 13.5958306	test: 15.6813620	best: 15.6813620 (341)	total: 12.6s	remaining: 2m 14s
    342:	learn: 13.5901392	test: 15.6814592	best: 15.6813620 (341)	total: 12.6s	remaining: 2m 14s
    343:	learn: 13.5819795	test: 15.6730254	best: 15.6730254 (343)	total: 12.7s	remaining: 2m 14s
    344:	learn: 13.5783303	test: 15.6702167	best: 15.6702167 (344)	total: 12.7s	remaining: 2m 14s
    345:	learn: 13.5723635	test: 15.6651333	best: 15.6651333 (345)	total: 12.8s	remaining: 2m 15s
    346:	learn: 13.5681058	test: 15.6633841	best: 15.6633841 (346)	total: 12.8s	remaining: 2m 15s
    347:	learn: 13.5622686	test: 15.6611628	best: 15.6611628 (347)	total: 12.9s	remaining: 2m 15s
    348:	learn: 13.5593663	test: 15.6600666	best: 15.6600666 (348)	total: 12.9s	remaining: 2m 15s
    349:	learn: 13.5482736	test: 15.6504541	best: 15.6504541 (349)	total: 13s	remaining: 2m 15s
    350:	learn: 13.5405587	test: 15.6424666	best: 15.6424666 (350)	total: 13s	remaining: 2m 15s
    351:	learn: 13.5358832	test: 15.6375920	best: 15.6375920 (351)	total: 13s	remaining: 2m 15s
    352:	learn: 13.5325578	test: 15.6339912	best: 15.6339912 (352)	total: 13.1s	remaining: 2m 14s
    353:	learn: 13.5290750	test: 15.6352238	best: 15.6339912 (352)	total: 13.1s	remaining: 2m 14s
    354:	learn: 13.5246124	test: 15.6335697	best: 15.6335697 (354)	total: 13.1s	remaining: 2m 14s
    355:	learn: 13.5216588	test: 15.6312476	best: 15.6312476 (355)	total: 13.2s	remaining: 2m 14s
    356:	learn: 13.5185323	test: 15.6298331	best: 15.6298331 (356)	total: 13.2s	remaining: 2m 14s
    357:	learn: 13.5134853	test: 15.6267946	best: 15.6267946 (357)	total: 13.2s	remaining: 2m 14s
    358:	learn: 13.5047735	test: 15.6198535	best: 15.6198535 (358)	total: 13.3s	remaining: 2m 14s
    359:	learn: 13.4987447	test: 15.6178750	best: 15.6178750 (359)	total: 13.3s	remaining: 2m 14s
    360:	learn: 13.4941092	test: 15.6125428	best: 15.6125428 (360)	total: 13.4s	remaining: 2m 14s
    361:	learn: 13.4836076	test: 15.6050896	best: 15.6050896 (361)	total: 13.4s	remaining: 2m 14s
    362:	learn: 13.4779710	test: 15.6031165	best: 15.6031165 (362)	total: 13.4s	remaining: 2m 14s
    363:	learn: 13.4703509	test: 15.5972136	best: 15.5972136 (363)	total: 13.5s	remaining: 2m 14s
    364:	learn: 13.4636128	test: 15.5950681	best: 15.5950681 (364)	total: 13.5s	remaining: 2m 14s
    365:	learn: 13.4584644	test: 15.5923118	best: 15.5923118 (365)	total: 13.5s	remaining: 2m 14s
    366:	learn: 13.4533658	test: 15.5903360	best: 15.5903360 (366)	total: 13.5s	remaining: 2m 14s
    367:	learn: 13.4418737	test: 15.5782366	best: 15.5782366 (367)	total: 13.6s	remaining: 2m 14s
    368:	learn: 13.4354385	test: 15.5755150	best: 15.5755150 (368)	total: 13.6s	remaining: 2m 13s
    369:	learn: 13.4259317	test: 15.5670985	best: 15.5670985 (369)	total: 13.6s	remaining: 2m 13s
    370:	learn: 13.4195157	test: 15.5634507	best: 15.5634507 (370)	total: 13.7s	remaining: 2m 13s
    371:	learn: 13.4131781	test: 15.5573933	best: 15.5573933 (371)	total: 13.7s	remaining: 2m 13s
    372:	learn: 13.4059458	test: 15.5521014	best: 15.5521014 (372)	total: 13.7s	remaining: 2m 13s
    373:	learn: 13.4032257	test: 15.5496276	best: 15.5496276 (373)	total: 13.8s	remaining: 2m 13s
    374:	learn: 13.3950155	test: 15.5458834	best: 15.5458834 (374)	total: 13.8s	remaining: 2m 13s
    375:	learn: 13.3900500	test: 15.5430108	best: 15.5430108 (375)	total: 13.8s	remaining: 2m 13s
    376:	learn: 13.3862926	test: 15.5419644	best: 15.5419644 (376)	total: 13.9s	remaining: 2m 13s
    377:	learn: 13.3827630	test: 15.5400806	best: 15.5400806 (377)	total: 13.9s	remaining: 2m 13s
    378:	learn: 13.3793279	test: 15.5374519	best: 15.5374519 (378)	total: 13.9s	remaining: 2m 13s
    379:	learn: 13.3774439	test: 15.5355831	best: 15.5355831 (379)	total: 14s	remaining: 2m 13s
    380:	learn: 13.3670237	test: 15.5272502	best: 15.5272502 (380)	total: 14s	remaining: 2m 13s
    381:	learn: 13.3641906	test: 15.5241157	best: 15.5241157 (381)	total: 14.1s	remaining: 2m 13s
    382:	learn: 13.3603841	test: 15.5231142	best: 15.5231142 (382)	total: 14.1s	remaining: 2m 13s
    383:	learn: 13.3561962	test: 15.5220282	best: 15.5220282 (383)	total: 14.2s	remaining: 2m 13s
    384:	learn: 13.3520005	test: 15.5187662	best: 15.5187662 (384)	total: 14.2s	remaining: 2m 13s
    385:	learn: 13.3492376	test: 15.5168094	best: 15.5168094 (385)	total: 14.3s	remaining: 2m 13s
    386:	learn: 13.3417487	test: 15.5111058	best: 15.5111058 (386)	total: 14.3s	remaining: 2m 13s
    387:	learn: 13.3387446	test: 15.5108907	best: 15.5108907 (387)	total: 14.3s	remaining: 2m 13s
    388:	learn: 13.3351597	test: 15.5085947	best: 15.5085947 (388)	total: 14.4s	remaining: 2m 13s
    389:	learn: 13.3338646	test: 15.5068955	best: 15.5068955 (389)	total: 14.4s	remaining: 2m 13s
    390:	learn: 13.3323966	test: 15.5052449	best: 15.5052449 (390)	total: 14.4s	remaining: 2m 13s
    391:	learn: 13.3278076	test: 15.5038004	best: 15.5038004 (391)	total: 14.4s	remaining: 2m 12s
    392:	learn: 13.3245261	test: 15.5029621	best: 15.5029621 (392)	total: 14.5s	remaining: 2m 12s
    393:	learn: 13.3216075	test: 15.5021046	best: 15.5021046 (393)	total: 14.5s	remaining: 2m 12s
    394:	learn: 13.3191643	test: 15.5009477	best: 15.5009477 (394)	total: 14.5s	remaining: 2m 12s
    395:	learn: 13.3137263	test: 15.4961486	best: 15.4961486 (395)	total: 14.6s	remaining: 2m 12s
    396:	learn: 13.3109108	test: 15.4936107	best: 15.4936107 (396)	total: 14.6s	remaining: 2m 12s
    397:	learn: 13.3023320	test: 15.4875893	best: 15.4875893 (397)	total: 14.6s	remaining: 2m 12s
    398:	learn: 13.2984380	test: 15.4857202	best: 15.4857202 (398)	total: 14.6s	remaining: 2m 12s
    399:	learn: 13.2944386	test: 15.4830334	best: 15.4830334 (399)	total: 14.7s	remaining: 2m 11s
    400:	learn: 13.2911987	test: 15.4815594	best: 15.4815594 (400)	total: 14.7s	remaining: 2m 11s
    401:	learn: 13.2843829	test: 15.4778822	best: 15.4778822 (401)	total: 14.7s	remaining: 2m 11s
    402:	learn: 13.2800551	test: 15.4775868	best: 15.4775868 (402)	total: 14.8s	remaining: 2m 11s
    403:	learn: 13.2709436	test: 15.4710196	best: 15.4710196 (403)	total: 14.8s	remaining: 2m 11s
    404:	learn: 13.2652883	test: 15.4684269	best: 15.4684269 (404)	total: 14.8s	remaining: 2m 11s
    405:	learn: 13.2634395	test: 15.4667615	best: 15.4667615 (405)	total: 14.9s	remaining: 2m 11s
    406:	learn: 13.2562665	test: 15.4621309	best: 15.4621309 (406)	total: 14.9s	remaining: 2m 11s
    407:	learn: 13.2524625	test: 15.4618351	best: 15.4618351 (407)	total: 14.9s	remaining: 2m 11s
    408:	learn: 13.2480002	test: 15.4590309	best: 15.4590309 (408)	total: 14.9s	remaining: 2m 11s
    409:	learn: 13.2439118	test: 15.4577075	best: 15.4577075 (409)	total: 15s	remaining: 2m 11s
    410:	learn: 13.2361395	test: 15.4501473	best: 15.4501473 (410)	total: 15s	remaining: 2m 11s
    411:	learn: 13.2321161	test: 15.4501182	best: 15.4501182 (411)	total: 15s	remaining: 2m 11s
    412:	learn: 13.2293045	test: 15.4477634	best: 15.4477634 (412)	total: 15.1s	remaining: 2m 10s
    413:	learn: 13.2254945	test: 15.4456867	best: 15.4456867 (413)	total: 15.1s	remaining: 2m 10s
    414:	learn: 13.2208422	test: 15.4443742	best: 15.4443742 (414)	total: 15.1s	remaining: 2m 10s
    415:	learn: 13.2126799	test: 15.4422150	best: 15.4422150 (415)	total: 15.2s	remaining: 2m 10s
    416:	learn: 13.2103221	test: 15.4410036	best: 15.4410036 (416)	total: 15.2s	remaining: 2m 10s
    417:	learn: 13.2061488	test: 15.4383943	best: 15.4383943 (417)	total: 15.2s	remaining: 2m 10s
    418:	learn: 13.2020200	test: 15.4352693	best: 15.4352693 (418)	total: 15.3s	remaining: 2m 10s
    419:	learn: 13.1980976	test: 15.4350799	best: 15.4350799 (419)	total: 15.3s	remaining: 2m 10s
    420:	learn: 13.1926353	test: 15.4309529	best: 15.4309529 (420)	total: 15.3s	remaining: 2m 10s
    421:	learn: 13.1892071	test: 15.4304720	best: 15.4304720 (421)	total: 15.4s	remaining: 2m 10s
    422:	learn: 13.1847693	test: 15.4279977	best: 15.4279977 (422)	total: 15.4s	remaining: 2m 10s
    423:	learn: 13.1806708	test: 15.4267666	best: 15.4267666 (423)	total: 15.4s	remaining: 2m 10s
    424:	learn: 13.1781571	test: 15.4237375	best: 15.4237375 (424)	total: 15.5s	remaining: 2m 10s
    425:	learn: 13.1711069	test: 15.4188042	best: 15.4188042 (425)	total: 15.5s	remaining: 2m 10s
    426:	learn: 13.1638627	test: 15.4134825	best: 15.4134825 (426)	total: 15.5s	remaining: 2m 9s
    427:	learn: 13.1611404	test: 15.4108411	best: 15.4108411 (427)	total: 15.6s	remaining: 2m 9s
    428:	learn: 13.1519433	test: 15.4001692	best: 15.4001692 (428)	total: 15.6s	remaining: 2m 9s
    429:	learn: 13.1444183	test: 15.3921027	best: 15.3921027 (429)	total: 15.6s	remaining: 2m 9s
    430:	learn: 13.1380974	test: 15.3904238	best: 15.3904238 (430)	total: 15.7s	remaining: 2m 9s
    431:	learn: 13.1317599	test: 15.3876038	best: 15.3876038 (431)	total: 15.7s	remaining: 2m 9s
    432:	learn: 13.1295604	test: 15.3853900	best: 15.3853900 (432)	total: 15.7s	remaining: 2m 9s
    433:	learn: 13.1238200	test: 15.3817820	best: 15.3817820 (433)	total: 15.7s	remaining: 2m 9s
    434:	learn: 13.1192699	test: 15.3833725	best: 15.3817820 (433)	total: 15.8s	remaining: 2m 9s
    435:	learn: 13.1113405	test: 15.3771940	best: 15.3771940 (435)	total: 15.8s	remaining: 2m 9s
    436:	learn: 13.1019921	test: 15.3702739	best: 15.3702739 (436)	total: 15.8s	remaining: 2m 9s
    437:	learn: 13.1001343	test: 15.3692388	best: 15.3692388 (437)	total: 15.9s	remaining: 2m 9s
    438:	learn: 13.0970040	test: 15.3691409	best: 15.3691409 (438)	total: 15.9s	remaining: 2m 8s
    439:	learn: 13.0944879	test: 15.3694288	best: 15.3691409 (438)	total: 15.9s	remaining: 2m 8s
    440:	learn: 13.0889682	test: 15.3666992	best: 15.3666992 (440)	total: 16s	remaining: 2m 8s
    441:	learn: 13.0822224	test: 15.3607969	best: 15.3607969 (441)	total: 16s	remaining: 2m 8s
    442:	learn: 13.0736518	test: 15.3536617	best: 15.3536617 (442)	total: 16s	remaining: 2m 8s
    443:	learn: 13.0670148	test: 15.3505245	best: 15.3505245 (443)	total: 16.1s	remaining: 2m 8s
    444:	learn: 13.0583598	test: 15.3433789	best: 15.3433789 (444)	total: 16.1s	remaining: 2m 8s
    445:	learn: 13.0545613	test: 15.3417224	best: 15.3417224 (445)	total: 16.1s	remaining: 2m 8s
    446:	learn: 13.0500728	test: 15.3411882	best: 15.3411882 (446)	total: 16.1s	remaining: 2m 8s
    447:	learn: 13.0465803	test: 15.3409376	best: 15.3409376 (447)	total: 16.2s	remaining: 2m 8s
    448:	learn: 13.0424895	test: 15.3402591	best: 15.3402591 (448)	total: 16.2s	remaining: 2m 8s
    449:	learn: 13.0384799	test: 15.3373331	best: 15.3373331 (449)	total: 16.2s	remaining: 2m 8s
    450:	learn: 13.0334884	test: 15.3349681	best: 15.3349681 (450)	total: 16.3s	remaining: 2m 8s
    451:	learn: 13.0298543	test: 15.3329686	best: 15.3329686 (451)	total: 16.3s	remaining: 2m 8s
    452:	learn: 13.0266019	test: 15.3325776	best: 15.3325776 (452)	total: 16.3s	remaining: 2m 7s
    453:	learn: 13.0159504	test: 15.3290377	best: 15.3290377 (453)	total: 16.4s	remaining: 2m 7s
    454:	learn: 13.0128286	test: 15.3274748	best: 15.3274748 (454)	total: 16.4s	remaining: 2m 7s
    455:	learn: 13.0102377	test: 15.3253930	best: 15.3253930 (455)	total: 16.4s	remaining: 2m 7s
    456:	learn: 13.0084598	test: 15.3240907	best: 15.3240907 (456)	total: 16.5s	remaining: 2m 7s
    457:	learn: 13.0055385	test: 15.3225171	best: 15.3225171 (457)	total: 16.5s	remaining: 2m 7s
    458:	learn: 13.0029302	test: 15.3211662	best: 15.3211662 (458)	total: 16.5s	remaining: 2m 7s
    459:	learn: 13.0018137	test: 15.3195917	best: 15.3195917 (459)	total: 16.5s	remaining: 2m 7s
    460:	learn: 12.9969948	test: 15.3173019	best: 15.3173019 (460)	total: 16.6s	remaining: 2m 7s
    461:	learn: 12.9941514	test: 15.3143051	best: 15.3143051 (461)	total: 16.6s	remaining: 2m 7s
    462:	learn: 12.9918944	test: 15.3126689	best: 15.3126689 (462)	total: 16.6s	remaining: 2m 6s
    463:	learn: 12.9878328	test: 15.3128280	best: 15.3126689 (462)	total: 16.7s	remaining: 2m 6s
    464:	learn: 12.9790652	test: 15.3067616	best: 15.3067616 (464)	total: 16.7s	remaining: 2m 6s
    465:	learn: 12.9721007	test: 15.3000668	best: 15.3000668 (465)	total: 16.7s	remaining: 2m 6s
    466:	learn: 12.9677200	test: 15.2955773	best: 15.2955773 (466)	total: 16.7s	remaining: 2m 6s
    467:	learn: 12.9605022	test: 15.2915685	best: 15.2915685 (467)	total: 16.8s	remaining: 2m 6s
    468:	learn: 12.9578038	test: 15.2928523	best: 15.2915685 (467)	total: 16.8s	remaining: 2m 6s
    469:	learn: 12.9555814	test: 15.2913990	best: 15.2913990 (469)	total: 16.8s	remaining: 2m 6s
    470:	learn: 12.9529531	test: 15.2897851	best: 15.2897851 (470)	total: 16.9s	remaining: 2m 6s
    471:	learn: 12.9495725	test: 15.2894920	best: 15.2894920 (471)	total: 16.9s	remaining: 2m 6s
    472:	learn: 12.9469404	test: 15.2888573	best: 15.2888573 (472)	total: 16.9s	remaining: 2m 6s
    473:	learn: 12.9433817	test: 15.2873761	best: 15.2873761 (473)	total: 16.9s	remaining: 2m 6s
    474:	learn: 12.9350802	test: 15.2813333	best: 15.2813333 (474)	total: 17s	remaining: 2m 5s
    475:	learn: 12.9303365	test: 15.2799895	best: 15.2799895 (475)	total: 17s	remaining: 2m 5s
    476:	learn: 12.9217701	test: 15.2710351	best: 15.2710351 (476)	total: 17s	remaining: 2m 5s
    477:	learn: 12.9189216	test: 15.2696612	best: 15.2696612 (477)	total: 17.1s	remaining: 2m 5s
    478:	learn: 12.9142782	test: 15.2664681	best: 15.2664681 (478)	total: 17.1s	remaining: 2m 5s
    479:	learn: 12.9128500	test: 15.2655570	best: 15.2655570 (479)	total: 17.1s	remaining: 2m 5s
    480:	learn: 12.9105296	test: 15.2658229	best: 15.2655570 (479)	total: 17.1s	remaining: 2m 5s
    481:	learn: 12.9005155	test: 15.2609592	best: 15.2609592 (481)	total: 17.2s	remaining: 2m 5s
    482:	learn: 12.8906921	test: 15.2549780	best: 15.2549780 (482)	total: 17.2s	remaining: 2m 5s
    483:	learn: 12.8875838	test: 15.2539037	best: 15.2539037 (483)	total: 17.2s	remaining: 2m 5s
    484:	learn: 12.8797294	test: 15.2459248	best: 15.2459248 (484)	total: 17.3s	remaining: 2m 5s
    485:	learn: 12.8779991	test: 15.2465476	best: 15.2459248 (484)	total: 17.3s	remaining: 2m 5s
    486:	learn: 12.8768382	test: 15.2461682	best: 15.2459248 (484)	total: 17.3s	remaining: 2m 4s
    487:	learn: 12.8742523	test: 15.2462402	best: 15.2459248 (484)	total: 17.4s	remaining: 2m 4s
    488:	learn: 12.8715819	test: 15.2457544	best: 15.2457544 (488)	total: 17.4s	remaining: 2m 4s
    489:	learn: 12.8656339	test: 15.2397276	best: 15.2397276 (489)	total: 17.4s	remaining: 2m 4s
    490:	learn: 12.8640474	test: 15.2390280	best: 15.2390280 (490)	total: 17.4s	remaining: 2m 4s
    491:	learn: 12.8560080	test: 15.2344767	best: 15.2344767 (491)	total: 17.5s	remaining: 2m 4s
    492:	learn: 12.8522288	test: 15.2317633	best: 15.2317633 (492)	total: 17.5s	remaining: 2m 4s
    493:	learn: 12.8484421	test: 15.2316082	best: 15.2316082 (493)	total: 17.5s	remaining: 2m 4s
    494:	learn: 12.8412094	test: 15.2229861	best: 15.2229861 (494)	total: 17.6s	remaining: 2m 4s
    495:	learn: 12.8373446	test: 15.2225505	best: 15.2225505 (495)	total: 17.6s	remaining: 2m 4s
    496:	learn: 12.8311979	test: 15.2175659	best: 15.2175659 (496)	total: 17.6s	remaining: 2m 4s
    497:	learn: 12.8260661	test: 15.2153048	best: 15.2153048 (497)	total: 17.6s	remaining: 2m 4s
    498:	learn: 12.8140033	test: 15.2111439	best: 15.2111439 (498)	total: 17.7s	remaining: 2m 3s
    499:	learn: 12.8046489	test: 15.2037008	best: 15.2037008 (499)	total: 17.7s	remaining: 2m 3s
    500:	learn: 12.7996096	test: 15.1995149	best: 15.1995149 (500)	total: 17.7s	remaining: 2m 3s
    501:	learn: 12.7927078	test: 15.1956347	best: 15.1956347 (501)	total: 17.8s	remaining: 2m 3s
    502:	learn: 12.7805692	test: 15.1873631	best: 15.1873631 (502)	total: 17.8s	remaining: 2m 3s
    503:	learn: 12.7773882	test: 15.1871014	best: 15.1871014 (503)	total: 17.8s	remaining: 2m 3s
    504:	learn: 12.7717564	test: 15.1822215	best: 15.1822215 (504)	total: 17.8s	remaining: 2m 3s
    505:	learn: 12.7665630	test: 15.1825320	best: 15.1822215 (504)	total: 17.9s	remaining: 2m 3s
    506:	learn: 12.7639692	test: 15.1800422	best: 15.1800422 (506)	total: 17.9s	remaining: 2m 3s
    507:	learn: 12.7621801	test: 15.1789980	best: 15.1789980 (507)	total: 17.9s	remaining: 2m 3s
    508:	learn: 12.7575068	test: 15.1776886	best: 15.1776886 (508)	total: 18s	remaining: 2m 3s
    509:	learn: 12.7497308	test: 15.1714370	best: 15.1714370 (509)	total: 18s	remaining: 2m 3s
    510:	learn: 12.7478680	test: 15.1707015	best: 15.1707015 (510)	total: 18.1s	remaining: 2m 3s
    511:	learn: 12.7443456	test: 15.1677572	best: 15.1677572 (511)	total: 18.1s	remaining: 2m 3s
    512:	learn: 12.7363429	test: 15.1634074	best: 15.1634074 (512)	total: 18.2s	remaining: 2m 3s
    513:	learn: 12.7327647	test: 15.1615161	best: 15.1615161 (513)	total: 18.2s	remaining: 2m 3s
    514:	learn: 12.7308551	test: 15.1612821	best: 15.1612821 (514)	total: 18.3s	remaining: 2m 3s
    515:	learn: 12.7271680	test: 15.1586914	best: 15.1586914 (515)	total: 18.3s	remaining: 2m 3s
    516:	learn: 12.7195763	test: 15.1529854	best: 15.1529854 (516)	total: 18.4s	remaining: 2m 3s
    517:	learn: 12.7143888	test: 15.1481868	best: 15.1481868 (517)	total: 18.4s	remaining: 2m 3s
    518:	learn: 12.7122430	test: 15.1475418	best: 15.1475418 (518)	total: 18.5s	remaining: 2m 3s
    519:	learn: 12.7083216	test: 15.1470751	best: 15.1470751 (519)	total: 18.5s	remaining: 2m 3s
    520:	learn: 12.7045529	test: 15.1451107	best: 15.1451107 (520)	total: 18.6s	remaining: 2m 3s
    521:	learn: 12.7017223	test: 15.1454976	best: 15.1451107 (520)	total: 18.6s	remaining: 2m 4s
    522:	learn: 12.6990009	test: 15.1436296	best: 15.1436296 (522)	total: 18.7s	remaining: 2m 4s
    523:	learn: 12.6961691	test: 15.1427973	best: 15.1427973 (523)	total: 18.7s	remaining: 2m 4s
    524:	learn: 12.6947453	test: 15.1417320	best: 15.1417320 (524)	total: 18.8s	remaining: 2m 4s
    525:	learn: 12.6927611	test: 15.1409061	best: 15.1409061 (525)	total: 18.8s	remaining: 2m 4s
    526:	learn: 12.6899612	test: 15.1381838	best: 15.1381838 (526)	total: 18.9s	remaining: 2m 4s
    527:	learn: 12.6856561	test: 15.1358616	best: 15.1358616 (527)	total: 18.9s	remaining: 2m 4s
    528:	learn: 12.6788727	test: 15.1303821	best: 15.1303821 (528)	total: 19s	remaining: 2m 4s
    529:	learn: 12.6696482	test: 15.1234498	best: 15.1234498 (529)	total: 19s	remaining: 2m 4s
    530:	learn: 12.6669467	test: 15.1234107	best: 15.1234107 (530)	total: 19.1s	remaining: 2m 4s
    531:	learn: 12.6599409	test: 15.1198330	best: 15.1198330 (531)	total: 19.1s	remaining: 2m 4s
    532:	learn: 12.6573478	test: 15.1201481	best: 15.1198330 (531)	total: 19.1s	remaining: 2m 4s
    533:	learn: 12.6541816	test: 15.1162778	best: 15.1162778 (533)	total: 19.2s	remaining: 2m 4s
    534:	learn: 12.6529828	test: 15.1153126	best: 15.1153126 (534)	total: 19.2s	remaining: 2m 4s
    535:	learn: 12.6486484	test: 15.1149881	best: 15.1149881 (535)	total: 19.2s	remaining: 2m 4s
    536:	learn: 12.6452110	test: 15.1150251	best: 15.1149881 (535)	total: 19.3s	remaining: 2m 4s
    537:	learn: 12.6441867	test: 15.1135635	best: 15.1135635 (537)	total: 19.3s	remaining: 2m 4s
    538:	learn: 12.6432884	test: 15.1134727	best: 15.1134727 (538)	total: 19.3s	remaining: 2m 4s
    539:	learn: 12.6338544	test: 15.1088576	best: 15.1088576 (539)	total: 19.4s	remaining: 2m 4s
    540:	learn: 12.6321858	test: 15.1076240	best: 15.1076240 (540)	total: 19.4s	remaining: 2m 3s
    541:	learn: 12.6309832	test: 15.1086137	best: 15.1076240 (540)	total: 19.4s	remaining: 2m 3s
    542:	learn: 12.6294935	test: 15.1087788	best: 15.1076240 (540)	total: 19.4s	remaining: 2m 3s
    543:	learn: 12.6263137	test: 15.1066331	best: 15.1066331 (543)	total: 19.5s	remaining: 2m 3s
    544:	learn: 12.6203289	test: 15.1001714	best: 15.1001714 (544)	total: 19.5s	remaining: 2m 3s
    545:	learn: 12.6180469	test: 15.0994114	best: 15.0994114 (545)	total: 19.5s	remaining: 2m 3s
    546:	learn: 12.6147072	test: 15.0979444	best: 15.0979444 (546)	total: 19.6s	remaining: 2m 3s
    547:	learn: 12.6082497	test: 15.0927490	best: 15.0927490 (547)	total: 19.6s	remaining: 2m 3s
    548:	learn: 12.6067229	test: 15.0915644	best: 15.0915644 (548)	total: 19.6s	remaining: 2m 3s
    549:	learn: 12.6033282	test: 15.0899046	best: 15.0899046 (549)	total: 19.6s	remaining: 2m 3s
    550:	learn: 12.6000472	test: 15.0883674	best: 15.0883674 (550)	total: 19.7s	remaining: 2m 3s
    551:	learn: 12.5984457	test: 15.0866322	best: 15.0866322 (551)	total: 19.7s	remaining: 2m 3s
    552:	learn: 12.5916120	test: 15.0818170	best: 15.0818170 (552)	total: 19.7s	remaining: 2m 2s
    553:	learn: 12.5884108	test: 15.0804683	best: 15.0804683 (553)	total: 19.8s	remaining: 2m 2s
    554:	learn: 12.5839084	test: 15.0773129	best: 15.0773129 (554)	total: 19.8s	remaining: 2m 2s
    555:	learn: 12.5801478	test: 15.0764150	best: 15.0764150 (555)	total: 19.8s	remaining: 2m 2s
    556:	learn: 12.5724611	test: 15.0693049	best: 15.0693049 (556)	total: 19.8s	remaining: 2m 2s
    557:	learn: 12.5701905	test: 15.0687966	best: 15.0687966 (557)	total: 19.9s	remaining: 2m 2s
    558:	learn: 12.5683374	test: 15.0692479	best: 15.0687966 (557)	total: 19.9s	remaining: 2m 2s
    559:	learn: 12.5657716	test: 15.0677323	best: 15.0677323 (559)	total: 19.9s	remaining: 2m 2s
    560:	learn: 12.5627772	test: 15.0656133	best: 15.0656133 (560)	total: 20s	remaining: 2m 2s
    561:	learn: 12.5600920	test: 15.0637916	best: 15.0637916 (561)	total: 20s	remaining: 2m 2s
    562:	learn: 12.5516692	test: 15.0610909	best: 15.0610909 (562)	total: 20.1s	remaining: 2m 2s
    563:	learn: 12.5503838	test: 15.0605146	best: 15.0605146 (563)	total: 20.1s	remaining: 2m 2s
    564:	learn: 12.5464345	test: 15.0616388	best: 15.0605146 (563)	total: 20.2s	remaining: 2m 2s
    565:	learn: 12.5381140	test: 15.0564833	best: 15.0564833 (565)	total: 20.3s	remaining: 2m 2s
    566:	learn: 12.5344430	test: 15.0570285	best: 15.0564833 (565)	total: 20.3s	remaining: 2m 2s
    567:	learn: 12.5299841	test: 15.0537950	best: 15.0537950 (567)	total: 20.3s	remaining: 2m 2s
    568:	learn: 12.5244806	test: 15.0468838	best: 15.0468838 (568)	total: 20.4s	remaining: 2m 2s
    569:	learn: 12.5219325	test: 15.0444956	best: 15.0444956 (569)	total: 20.4s	remaining: 2m 2s
    570:	learn: 12.5182479	test: 15.0434024	best: 15.0434024 (570)	total: 20.4s	remaining: 2m 2s
    571:	learn: 12.5145352	test: 15.0399492	best: 15.0399492 (571)	total: 20.4s	remaining: 2m 2s
    572:	learn: 12.5115636	test: 15.0378397	best: 15.0378397 (572)	total: 20.5s	remaining: 2m 2s
    573:	learn: 12.5080054	test: 15.0382665	best: 15.0378397 (572)	total: 20.5s	remaining: 2m 2s
    574:	learn: 12.5057907	test: 15.0383976	best: 15.0378397 (572)	total: 20.5s	remaining: 2m 2s
    575:	learn: 12.4985514	test: 15.0323383	best: 15.0323383 (575)	total: 20.6s	remaining: 2m 2s
    576:	learn: 12.4948009	test: 15.0303112	best: 15.0303112 (576)	total: 20.6s	remaining: 2m 2s
    577:	learn: 12.4885998	test: 15.0255046	best: 15.0255046 (577)	total: 20.6s	remaining: 2m 2s
    578:	learn: 12.4866902	test: 15.0252344	best: 15.0252344 (578)	total: 20.6s	remaining: 2m 1s
    579:	learn: 12.4757842	test: 15.0180691	best: 15.0180691 (579)	total: 20.7s	remaining: 2m 1s
    580:	learn: 12.4730916	test: 15.0155726	best: 15.0155726 (580)	total: 20.7s	remaining: 2m 1s
    581:	learn: 12.4650328	test: 15.0077821	best: 15.0077821 (581)	total: 20.7s	remaining: 2m 1s
    582:	learn: 12.4618438	test: 15.0059115	best: 15.0059115 (582)	total: 20.8s	remaining: 2m 1s
    583:	learn: 12.4598621	test: 15.0045258	best: 15.0045258 (583)	total: 20.8s	remaining: 2m 1s
    584:	learn: 12.4575472	test: 15.0036710	best: 15.0036710 (584)	total: 20.8s	remaining: 2m 1s
    585:	learn: 12.4559667	test: 15.0026531	best: 15.0026531 (585)	total: 20.9s	remaining: 2m 1s
    586:	learn: 12.4533961	test: 14.9996828	best: 14.9996828 (586)	total: 20.9s	remaining: 2m 1s
    587:	learn: 12.4489143	test: 14.9984865	best: 14.9984865 (587)	total: 20.9s	remaining: 2m 1s
    588:	learn: 12.4368047	test: 14.9899360	best: 14.9899360 (588)	total: 21s	remaining: 2m 1s
    589:	learn: 12.4295606	test: 14.9842258	best: 14.9842258 (589)	total: 21s	remaining: 2m 1s
    590:	learn: 12.4263624	test: 14.9826942	best: 14.9826942 (590)	total: 21s	remaining: 2m 1s
    591:	learn: 12.4241978	test: 14.9801750	best: 14.9801750 (591)	total: 21.1s	remaining: 2m 1s
    592:	learn: 12.4222244	test: 14.9786386	best: 14.9786386 (592)	total: 21.1s	remaining: 2m 1s
    593:	learn: 12.4175925	test: 14.9763436	best: 14.9763436 (593)	total: 21.1s	remaining: 2m 1s
    594:	learn: 12.4138858	test: 14.9737350	best: 14.9737350 (594)	total: 21.2s	remaining: 2m 1s
    595:	learn: 12.4123934	test: 14.9721763	best: 14.9721763 (595)	total: 21.2s	remaining: 2m 1s
    596:	learn: 12.4110011	test: 14.9695838	best: 14.9695838 (596)	total: 21.2s	remaining: 2m
    597:	learn: 12.4070414	test: 14.9672510	best: 14.9672510 (597)	total: 21.3s	remaining: 2m
    598:	learn: 12.4036068	test: 14.9646348	best: 14.9646348 (598)	total: 21.3s	remaining: 2m
    599:	learn: 12.4005274	test: 14.9629819	best: 14.9629819 (599)	total: 21.3s	remaining: 2m
    600:	learn: 12.3943308	test: 14.9602097	best: 14.9602097 (600)	total: 21.3s	remaining: 2m
    601:	learn: 12.3913723	test: 14.9573985	best: 14.9573985 (601)	total: 21.4s	remaining: 2m
    602:	learn: 12.3889560	test: 14.9558808	best: 14.9558808 (602)	total: 21.4s	remaining: 2m
    603:	learn: 12.3860352	test: 14.9533887	best: 14.9533887 (603)	total: 21.4s	remaining: 2m
    604:	learn: 12.3842712	test: 14.9531464	best: 14.9531464 (604)	total: 21.5s	remaining: 2m
    605:	learn: 12.3804390	test: 14.9523873	best: 14.9523873 (605)	total: 21.5s	remaining: 2m
    606:	learn: 12.3780914	test: 14.9521339	best: 14.9521339 (606)	total: 21.5s	remaining: 2m
    607:	learn: 12.3758832	test: 14.9514745	best: 14.9514745 (607)	total: 21.6s	remaining: 2m
    608:	learn: 12.3734046	test: 14.9491151	best: 14.9491151 (608)	total: 21.6s	remaining: 2m
    609:	learn: 12.3710576	test: 14.9472609	best: 14.9472609 (609)	total: 21.7s	remaining: 2m
    610:	learn: 12.3683417	test: 14.9457065	best: 14.9457065 (610)	total: 21.7s	remaining: 2m
    611:	learn: 12.3604799	test: 14.9382797	best: 14.9382797 (611)	total: 21.8s	remaining: 2m
    612:	learn: 12.3590481	test: 14.9388608	best: 14.9382797 (611)	total: 21.8s	remaining: 2m
    613:	learn: 12.3570662	test: 14.9379250	best: 14.9379250 (613)	total: 21.9s	remaining: 2m
    614:	learn: 12.3514155	test: 14.9316653	best: 14.9316653 (614)	total: 21.9s	remaining: 2m
    615:	learn: 12.3499161	test: 14.9312264	best: 14.9312264 (615)	total: 22s	remaining: 2m
    616:	learn: 12.3481772	test: 14.9317071	best: 14.9312264 (615)	total: 22s	remaining: 2m
    617:	learn: 12.3400473	test: 14.9259254	best: 14.9259254 (617)	total: 22s	remaining: 2m
    618:	learn: 12.3334703	test: 14.9216006	best: 14.9216006 (618)	total: 22.1s	remaining: 2m
    619:	learn: 12.3315832	test: 14.9212754	best: 14.9212754 (619)	total: 22.1s	remaining: 2m
    620:	learn: 12.3279418	test: 14.9181406	best: 14.9181406 (620)	total: 22.1s	remaining: 2m
    621:	learn: 12.3261852	test: 14.9160097	best: 14.9160097 (621)	total: 22.1s	remaining: 2m
    622:	learn: 12.3207572	test: 14.9145381	best: 14.9145381 (622)	total: 22.2s	remaining: 2m
    623:	learn: 12.3164298	test: 14.9128260	best: 14.9128260 (623)	total: 22.2s	remaining: 2m
    624:	learn: 12.3151998	test: 14.9127446	best: 14.9127446 (624)	total: 22.2s	remaining: 2m
    625:	learn: 12.3086063	test: 14.9048496	best: 14.9048496 (625)	total: 22.3s	remaining: 2m
    626:	learn: 12.3048717	test: 14.9037095	best: 14.9037095 (626)	total: 22.3s	remaining: 1m 59s
    627:	learn: 12.2996755	test: 14.8983147	best: 14.8983147 (627)	total: 22.3s	remaining: 1m 59s
    628:	learn: 12.2931603	test: 14.8943708	best: 14.8943708 (628)	total: 22.4s	remaining: 1m 59s
    629:	learn: 12.2868712	test: 14.8883226	best: 14.8883226 (629)	total: 22.4s	remaining: 1m 59s
    630:	learn: 12.2761291	test: 14.8805695	best: 14.8805695 (630)	total: 22.4s	remaining: 1m 59s
    631:	learn: 12.2722889	test: 14.8786064	best: 14.8786064 (631)	total: 22.5s	remaining: 1m 59s
    632:	learn: 12.2664215	test: 14.8727679	best: 14.8727679 (632)	total: 22.5s	remaining: 1m 59s
    633:	learn: 12.2641618	test: 14.8706446	best: 14.8706446 (633)	total: 22.5s	remaining: 1m 59s
    634:	learn: 12.2566748	test: 14.8655064	best: 14.8655064 (634)	total: 22.5s	remaining: 1m 59s
    635:	learn: 12.2553419	test: 14.8652802	best: 14.8652802 (635)	total: 22.6s	remaining: 1m 59s
    636:	learn: 12.2485555	test: 14.8623320	best: 14.8623320 (636)	total: 22.6s	remaining: 1m 59s
    637:	learn: 12.2469737	test: 14.8620465	best: 14.8620465 (637)	total: 22.6s	remaining: 1m 59s
    638:	learn: 12.2443821	test: 14.8610833	best: 14.8610833 (638)	total: 22.6s	remaining: 1m 59s
    639:	learn: 12.2420583	test: 14.8598640	best: 14.8598640 (639)	total: 22.7s	remaining: 1m 59s
    640:	learn: 12.2396157	test: 14.8588529	best: 14.8588529 (640)	total: 22.7s	remaining: 1m 58s
    641:	learn: 12.2381757	test: 14.8587387	best: 14.8587387 (641)	total: 22.7s	remaining: 1m 58s
    642:	learn: 12.2356374	test: 14.8613059	best: 14.8587387 (641)	total: 22.8s	remaining: 1m 58s
    643:	learn: 12.2332049	test: 14.8628061	best: 14.8587387 (641)	total: 22.8s	remaining: 1m 58s
    644:	learn: 12.2288071	test: 14.8587694	best: 14.8587387 (641)	total: 22.8s	remaining: 1m 58s
    645:	learn: 12.2264995	test: 14.8585418	best: 14.8585418 (645)	total: 22.8s	remaining: 1m 58s
    646:	learn: 12.2223972	test: 14.8584116	best: 14.8584116 (646)	total: 22.9s	remaining: 1m 58s
    647:	learn: 12.2208130	test: 14.8573004	best: 14.8573004 (647)	total: 22.9s	remaining: 1m 58s
    648:	learn: 12.2171207	test: 14.8554229	best: 14.8554229 (648)	total: 22.9s	remaining: 1m 58s
    649:	learn: 12.2163962	test: 14.8547031	best: 14.8547031 (649)	total: 23s	remaining: 1m 58s
    650:	learn: 12.2148424	test: 14.8532654	best: 14.8532654 (650)	total: 23s	remaining: 1m 58s
    651:	learn: 12.2118485	test: 14.8509306	best: 14.8509306 (651)	total: 23s	remaining: 1m 58s
    652:	learn: 12.2041475	test: 14.8458972	best: 14.8458972 (652)	total: 23.1s	remaining: 1m 58s
    653:	learn: 12.1974322	test: 14.8436271	best: 14.8436271 (653)	total: 23.1s	remaining: 1m 58s
    654:	learn: 12.1953404	test: 14.8414809	best: 14.8414809 (654)	total: 23.1s	remaining: 1m 58s
    655:	learn: 12.1932538	test: 14.8392639	best: 14.8392639 (655)	total: 23.2s	remaining: 1m 58s
    656:	learn: 12.1928696	test: 14.8401499	best: 14.8392639 (655)	total: 23.2s	remaining: 1m 57s
    657:	learn: 12.1844465	test: 14.8371645	best: 14.8371645 (657)	total: 23.2s	remaining: 1m 57s
    658:	learn: 12.1772062	test: 14.8323956	best: 14.8323956 (658)	total: 23.2s	remaining: 1m 57s
    659:	learn: 12.1730137	test: 14.8289694	best: 14.8289694 (659)	total: 23.3s	remaining: 1m 57s
    660:	learn: 12.1716316	test: 14.8293998	best: 14.8289694 (659)	total: 23.3s	remaining: 1m 57s
    661:	learn: 12.1643569	test: 14.8235189	best: 14.8235189 (661)	total: 23.3s	remaining: 1m 57s
    662:	learn: 12.1590352	test: 14.8189990	best: 14.8189990 (662)	total: 23.4s	remaining: 1m 57s
    663:	learn: 12.1569722	test: 14.8183369	best: 14.8183369 (663)	total: 23.4s	remaining: 1m 57s
    664:	learn: 12.1548673	test: 14.8174042	best: 14.8174042 (664)	total: 23.4s	remaining: 1m 57s
    665:	learn: 12.1516465	test: 14.8125138	best: 14.8125138 (665)	total: 23.5s	remaining: 1m 57s
    666:	learn: 12.1482347	test: 14.8123674	best: 14.8123674 (666)	total: 23.5s	remaining: 1m 57s
    667:	learn: 12.1472232	test: 14.8124494	best: 14.8123674 (666)	total: 23.5s	remaining: 1m 57s
    668:	learn: 12.1450178	test: 14.8110589	best: 14.8110589 (668)	total: 23.5s	remaining: 1m 57s
    669:	learn: 12.1431981	test: 14.8096076	best: 14.8096076 (669)	total: 23.6s	remaining: 1m 57s
    670:	learn: 12.1414703	test: 14.8074679	best: 14.8074679 (670)	total: 23.6s	remaining: 1m 57s
    671:	learn: 12.1337962	test: 14.8010637	best: 14.8010637 (671)	total: 23.6s	remaining: 1m 57s
    672:	learn: 12.1282832	test: 14.7981880	best: 14.7981880 (672)	total: 23.7s	remaining: 1m 57s
    673:	learn: 12.1267768	test: 14.7986290	best: 14.7981880 (672)	total: 23.7s	remaining: 1m 57s
    674:	learn: 12.1204415	test: 14.7969119	best: 14.7969119 (674)	total: 23.8s	remaining: 1m 57s
    675:	learn: 12.1184398	test: 14.7946042	best: 14.7946042 (675)	total: 23.8s	remaining: 1m 57s
    676:	learn: 12.1116866	test: 14.7897314	best: 14.7897314 (676)	total: 23.9s	remaining: 1m 57s
    677:	learn: 12.1085329	test: 14.7884041	best: 14.7884041 (677)	total: 23.9s	remaining: 1m 57s
    678:	learn: 12.1004434	test: 14.7836466	best: 14.7836466 (678)	total: 23.9s	remaining: 1m 57s
    679:	learn: 12.0962609	test: 14.7790652	best: 14.7790652 (679)	total: 24s	remaining: 1m 57s
    680:	learn: 12.0939597	test: 14.7774759	best: 14.7774759 (680)	total: 24s	remaining: 1m 56s
    681:	learn: 12.0854662	test: 14.7713896	best: 14.7713896 (681)	total: 24s	remaining: 1m 56s
    682:	learn: 12.0811666	test: 14.7691540	best: 14.7691540 (682)	total: 24.1s	remaining: 1m 56s
    683:	learn: 12.0772719	test: 14.7658195	best: 14.7658195 (683)	total: 24.1s	remaining: 1m 56s
    684:	learn: 12.0739218	test: 14.7648607	best: 14.7648607 (684)	total: 24.1s	remaining: 1m 56s
    685:	learn: 12.0699781	test: 14.7623822	best: 14.7623822 (685)	total: 24.1s	remaining: 1m 56s
    686:	learn: 12.0674758	test: 14.7599426	best: 14.7599426 (686)	total: 24.2s	remaining: 1m 56s
    687:	learn: 12.0644541	test: 14.7582981	best: 14.7582981 (687)	total: 24.2s	remaining: 1m 56s
    688:	learn: 12.0555899	test: 14.7531828	best: 14.7531828 (688)	total: 24.2s	remaining: 1m 56s
    689:	learn: 12.0529175	test: 14.7528057	best: 14.7528057 (689)	total: 24.3s	remaining: 1m 56s
    690:	learn: 12.0508993	test: 14.7530990	best: 14.7528057 (689)	total: 24.3s	remaining: 1m 56s
    691:	learn: 12.0473560	test: 14.7526636	best: 14.7526636 (691)	total: 24.3s	remaining: 1m 56s
    692:	learn: 12.0454939	test: 14.7513753	best: 14.7513753 (692)	total: 24.4s	remaining: 1m 56s
    693:	learn: 12.0408017	test: 14.7484934	best: 14.7484934 (693)	total: 24.4s	remaining: 1m 56s
    694:	learn: 12.0313777	test: 14.7428478	best: 14.7428478 (694)	total: 24.4s	remaining: 1m 56s
    695:	learn: 12.0297900	test: 14.7434214	best: 14.7428478 (694)	total: 24.5s	remaining: 1m 56s
    696:	learn: 12.0286133	test: 14.7427223	best: 14.7427223 (696)	total: 24.5s	remaining: 1m 56s
    697:	learn: 12.0209616	test: 14.7359115	best: 14.7359115 (697)	total: 24.5s	remaining: 1m 56s
    698:	learn: 12.0153501	test: 14.7316288	best: 14.7316288 (698)	total: 24.6s	remaining: 1m 56s
    699:	learn: 12.0085537	test: 14.7261298	best: 14.7261298 (699)	total: 24.6s	remaining: 1m 56s
    700:	learn: 12.0055814	test: 14.7239527	best: 14.7239527 (700)	total: 24.6s	remaining: 1m 55s
    701:	learn: 11.9984557	test: 14.7193399	best: 14.7193399 (701)	total: 24.7s	remaining: 1m 55s
    702:	learn: 11.9964467	test: 14.7173478	best: 14.7173478 (702)	total: 24.7s	remaining: 1m 55s
    703:	learn: 11.9930880	test: 14.7169890	best: 14.7169890 (703)	total: 24.7s	remaining: 1m 55s
    704:	learn: 11.9898171	test: 14.7162940	best: 14.7162940 (704)	total: 24.8s	remaining: 1m 55s
    705:	learn: 11.9875102	test: 14.7145331	best: 14.7145331 (705)	total: 24.8s	remaining: 1m 55s
    706:	learn: 11.9835652	test: 14.7112013	best: 14.7112013 (706)	total: 24.8s	remaining: 1m 55s
    707:	learn: 11.9817721	test: 14.7112587	best: 14.7112013 (706)	total: 24.8s	remaining: 1m 55s
    708:	learn: 11.9788943	test: 14.7082177	best: 14.7082177 (708)	total: 24.9s	remaining: 1m 55s
    709:	learn: 11.9765147	test: 14.7073833	best: 14.7073833 (709)	total: 24.9s	remaining: 1m 55s
    710:	learn: 11.9749905	test: 14.7059348	best: 14.7059348 (710)	total: 24.9s	remaining: 1m 55s
    711:	learn: 11.9733657	test: 14.7063107	best: 14.7059348 (710)	total: 25s	remaining: 1m 55s
    712:	learn: 11.9709689	test: 14.7055677	best: 14.7055677 (712)	total: 25s	remaining: 1m 55s
    713:	learn: 11.9673936	test: 14.7054307	best: 14.7054307 (713)	total: 25s	remaining: 1m 55s
    714:	learn: 11.9646698	test: 14.7060635	best: 14.7054307 (713)	total: 25.1s	remaining: 1m 55s
    715:	learn: 11.9628907	test: 14.7059947	best: 14.7054307 (713)	total: 25.1s	remaining: 1m 55s
    716:	learn: 11.9609702	test: 14.7042912	best: 14.7042912 (716)	total: 25.1s	remaining: 1m 55s
    717:	learn: 11.9595043	test: 14.7035841	best: 14.7035841 (717)	total: 25.2s	remaining: 1m 54s
    718:	learn: 11.9553445	test: 14.7017596	best: 14.7017596 (718)	total: 25.2s	remaining: 1m 54s
    719:	learn: 11.9541347	test: 14.7016975	best: 14.7016975 (719)	total: 25.2s	remaining: 1m 54s
    720:	learn: 11.9483385	test: 14.6968004	best: 14.6968004 (720)	total: 25.2s	remaining: 1m 54s
    721:	learn: 11.9470256	test: 14.6962591	best: 14.6962591 (721)	total: 25.3s	remaining: 1m 54s
    722:	learn: 11.9440868	test: 14.6940898	best: 14.6940898 (722)	total: 25.3s	remaining: 1m 54s
    723:	learn: 11.9392584	test: 14.6925707	best: 14.6925707 (723)	total: 25.3s	remaining: 1m 54s
    724:	learn: 11.9374058	test: 14.6917657	best: 14.6917657 (724)	total: 25.4s	remaining: 1m 54s
    725:	learn: 11.9343181	test: 14.6924095	best: 14.6917657 (724)	total: 25.4s	remaining: 1m 54s
    726:	learn: 11.9330484	test: 14.6918375	best: 14.6917657 (724)	total: 25.5s	remaining: 1m 54s
    727:	learn: 11.9311472	test: 14.6900187	best: 14.6900187 (727)	total: 25.5s	remaining: 1m 54s
    728:	learn: 11.9287157	test: 14.6887757	best: 14.6887757 (728)	total: 25.6s	remaining: 1m 54s
    729:	learn: 11.9269987	test: 14.6884980	best: 14.6884980 (729)	total: 25.6s	remaining: 1m 54s
    730:	learn: 11.9214406	test: 14.6848030	best: 14.6848030 (730)	total: 25.6s	remaining: 1m 54s
    731:	learn: 11.9137135	test: 14.6807599	best: 14.6807599 (731)	total: 25.6s	remaining: 1m 54s
    732:	learn: 11.9086751	test: 14.6781496	best: 14.6781496 (732)	total: 25.7s	remaining: 1m 54s
    733:	learn: 11.9064141	test: 14.6778192	best: 14.6778192 (733)	total: 25.7s	remaining: 1m 54s
    734:	learn: 11.9037116	test: 14.6791085	best: 14.6778192 (733)	total: 25.7s	remaining: 1m 54s
    735:	learn: 11.9023532	test: 14.6798621	best: 14.6778192 (733)	total: 25.8s	remaining: 1m 54s
    736:	learn: 11.9002512	test: 14.6780916	best: 14.6778192 (733)	total: 25.8s	remaining: 1m 54s
    737:	learn: 11.8965885	test: 14.6785665	best: 14.6778192 (733)	total: 25.8s	remaining: 1m 54s
    738:	learn: 11.8947072	test: 14.6784855	best: 14.6778192 (733)	total: 25.8s	remaining: 1m 53s
    739:	learn: 11.8917040	test: 14.6772383	best: 14.6772383 (739)	total: 25.9s	remaining: 1m 53s
    740:	learn: 11.8870049	test: 14.6731114	best: 14.6731114 (740)	total: 25.9s	remaining: 1m 53s
    741:	learn: 11.8846580	test: 14.6727827	best: 14.6727827 (741)	total: 25.9s	remaining: 1m 53s
    742:	learn: 11.8813483	test: 14.6708494	best: 14.6708494 (742)	total: 26s	remaining: 1m 53s
    743:	learn: 11.8788617	test: 14.6704815	best: 14.6704815 (743)	total: 26s	remaining: 1m 53s
    744:	learn: 11.8766528	test: 14.6695661	best: 14.6695661 (744)	total: 26s	remaining: 1m 53s
    745:	learn: 11.8746610	test: 14.6669854	best: 14.6669854 (745)	total: 26s	remaining: 1m 53s
    746:	learn: 11.8721150	test: 14.6675319	best: 14.6669854 (745)	total: 26.1s	remaining: 1m 53s
    747:	learn: 11.8703636	test: 14.6656006	best: 14.6656006 (747)	total: 26.1s	remaining: 1m 53s
    748:	learn: 11.8696938	test: 14.6649696	best: 14.6649696 (748)	total: 26.2s	remaining: 1m 53s
    749:	learn: 11.8672909	test: 14.6614835	best: 14.6614835 (749)	total: 26.2s	remaining: 1m 53s
    750:	learn: 11.8660998	test: 14.6630523	best: 14.6614835 (749)	total: 26.3s	remaining: 1m 53s
    751:	learn: 11.8628747	test: 14.6584322	best: 14.6584322 (751)	total: 26.3s	remaining: 1m 53s
    752:	learn: 11.8604648	test: 14.6571890	best: 14.6571890 (752)	total: 26.4s	remaining: 1m 53s
    753:	learn: 11.8590554	test: 14.6577987	best: 14.6571890 (752)	total: 26.4s	remaining: 1m 53s
    754:	learn: 11.8527305	test: 14.6534869	best: 14.6534869 (754)	total: 26.5s	remaining: 1m 53s
    755:	learn: 11.8497160	test: 14.6514690	best: 14.6514690 (755)	total: 26.5s	remaining: 1m 53s
    756:	learn: 11.8468403	test: 14.6505110	best: 14.6505110 (756)	total: 26.6s	remaining: 1m 53s
    757:	learn: 11.8451998	test: 14.6495406	best: 14.6495406 (757)	total: 26.6s	remaining: 1m 53s
    758:	learn: 11.8429803	test: 14.6495851	best: 14.6495406 (757)	total: 26.7s	remaining: 1m 53s
    759:	learn: 11.8387689	test: 14.6469859	best: 14.6469859 (759)	total: 26.7s	remaining: 1m 53s
    760:	learn: 11.8371484	test: 14.6469756	best: 14.6469756 (760)	total: 26.7s	remaining: 1m 53s
    761:	learn: 11.8300210	test: 14.6422640	best: 14.6422640 (761)	total: 26.8s	remaining: 1m 53s
    762:	learn: 11.8277916	test: 14.6417034	best: 14.6417034 (762)	total: 26.8s	remaining: 1m 53s
    763:	learn: 11.8223488	test: 14.6384202	best: 14.6384202 (763)	total: 26.9s	remaining: 1m 53s
    764:	learn: 11.8132644	test: 14.6325327	best: 14.6325327 (764)	total: 26.9s	remaining: 1m 53s
    765:	learn: 11.8111749	test: 14.6321551	best: 14.6321551 (765)	total: 26.9s	remaining: 1m 53s
    766:	learn: 11.8008593	test: 14.6255766	best: 14.6255766 (766)	total: 27s	remaining: 1m 53s
    767:	learn: 11.7990413	test: 14.6269111	best: 14.6255766 (766)	total: 27s	remaining: 1m 53s
    768:	learn: 11.7912771	test: 14.6215512	best: 14.6215512 (768)	total: 27s	remaining: 1m 53s
    769:	learn: 11.7829271	test: 14.6155274	best: 14.6155274 (769)	total: 27.1s	remaining: 1m 53s
    770:	learn: 11.7808520	test: 14.6155807	best: 14.6155274 (769)	total: 27.1s	remaining: 1m 53s
    771:	learn: 11.7768955	test: 14.6120152	best: 14.6120152 (771)	total: 27.2s	remaining: 1m 53s
    772:	learn: 11.7740518	test: 14.6090131	best: 14.6090131 (772)	total: 27.2s	remaining: 1m 53s
    773:	learn: 11.7701803	test: 14.6074359	best: 14.6074359 (773)	total: 27.3s	remaining: 1m 53s
    774:	learn: 11.7652930	test: 14.6035181	best: 14.6035181 (774)	total: 27.3s	remaining: 1m 53s
    775:	learn: 11.7640157	test: 14.6038992	best: 14.6035181 (774)	total: 27.3s	remaining: 1m 53s
    776:	learn: 11.7599419	test: 14.6011141	best: 14.6011141 (776)	total: 27.3s	remaining: 1m 53s
    777:	learn: 11.7530734	test: 14.5969275	best: 14.5969275 (777)	total: 27.4s	remaining: 1m 53s
    778:	learn: 11.7506216	test: 14.5942221	best: 14.5942221 (778)	total: 27.4s	remaining: 1m 53s
    779:	learn: 11.7481782	test: 14.5920505	best: 14.5920505 (779)	total: 27.5s	remaining: 1m 53s
    780:	learn: 11.7458098	test: 14.5911249	best: 14.5911249 (780)	total: 27.5s	remaining: 1m 53s
    781:	learn: 11.7447987	test: 14.5916068	best: 14.5911249 (780)	total: 27.5s	remaining: 1m 53s
    782:	learn: 11.7423888	test: 14.5923967	best: 14.5911249 (780)	total: 27.6s	remaining: 1m 53s
    783:	learn: 11.7387844	test: 14.5932883	best: 14.5911249 (780)	total: 27.6s	remaining: 1m 53s
    784:	learn: 11.7351519	test: 14.5946165	best: 14.5911249 (780)	total: 27.6s	remaining: 1m 53s
    785:	learn: 11.7337719	test: 14.5954882	best: 14.5911249 (780)	total: 27.7s	remaining: 1m 53s
    786:	learn: 11.7278343	test: 14.5893412	best: 14.5893412 (786)	total: 27.7s	remaining: 1m 53s
    787:	learn: 11.7250632	test: 14.5865829	best: 14.5865829 (787)	total: 27.7s	remaining: 1m 53s
    788:	learn: 11.7225543	test: 14.5851331	best: 14.5851331 (788)	total: 27.8s	remaining: 1m 53s
    789:	learn: 11.7174337	test: 14.5823278	best: 14.5823278 (789)	total: 27.8s	remaining: 1m 52s
    790:	learn: 11.7102111	test: 14.5769266	best: 14.5769266 (790)	total: 27.8s	remaining: 1m 52s
    791:	learn: 11.7067989	test: 14.5766379	best: 14.5766379 (791)	total: 27.9s	remaining: 1m 52s
    792:	learn: 11.7024022	test: 14.5755088	best: 14.5755088 (792)	total: 27.9s	remaining: 1m 52s
    793:	learn: 11.7006616	test: 14.5754260	best: 14.5754260 (793)	total: 27.9s	remaining: 1m 52s
    794:	learn: 11.7001170	test: 14.5749140	best: 14.5749140 (794)	total: 28s	remaining: 1m 52s
    795:	learn: 11.6986660	test: 14.5752826	best: 14.5749140 (794)	total: 28s	remaining: 1m 52s
    796:	learn: 11.6960846	test: 14.5736390	best: 14.5736390 (796)	total: 28.1s	remaining: 1m 52s
    797:	learn: 11.6941528	test: 14.5733602	best: 14.5733602 (797)	total: 28.1s	remaining: 1m 52s
    798:	learn: 11.6920271	test: 14.5715518	best: 14.5715518 (798)	total: 28.1s	remaining: 1m 52s
    799:	learn: 11.6884854	test: 14.5696051	best: 14.5696051 (799)	total: 28.2s	remaining: 1m 52s
    800:	learn: 11.6851262	test: 14.5669396	best: 14.5669396 (800)	total: 28.2s	remaining: 1m 52s
    801:	learn: 11.6819524	test: 14.5680211	best: 14.5669396 (800)	total: 28.2s	remaining: 1m 52s
    802:	learn: 11.6770777	test: 14.5664369	best: 14.5664369 (802)	total: 28.3s	remaining: 1m 52s
    803:	learn: 11.6703575	test: 14.5602106	best: 14.5602106 (803)	total: 28.3s	remaining: 1m 52s
    804:	learn: 11.6660257	test: 14.5599698	best: 14.5599698 (804)	total: 28.3s	remaining: 1m 52s
    805:	learn: 11.6600571	test: 14.5558883	best: 14.5558883 (805)	total: 28.4s	remaining: 1m 52s
    806:	learn: 11.6539157	test: 14.5492226	best: 14.5492226 (806)	total: 28.4s	remaining: 1m 52s
    807:	learn: 11.6528438	test: 14.5490869	best: 14.5490869 (807)	total: 28.4s	remaining: 1m 52s
    808:	learn: 11.6505835	test: 14.5495178	best: 14.5490869 (807)	total: 28.5s	remaining: 1m 52s
    809:	learn: 11.6465705	test: 14.5475245	best: 14.5475245 (809)	total: 28.5s	remaining: 1m 52s
    810:	learn: 11.6442582	test: 14.5478266	best: 14.5475245 (809)	total: 28.5s	remaining: 1m 52s
    811:	learn: 11.6424959	test: 14.5476515	best: 14.5475245 (809)	total: 28.6s	remaining: 1m 52s
    812:	learn: 11.6408903	test: 14.5465548	best: 14.5465548 (812)	total: 28.6s	remaining: 1m 52s
    813:	learn: 11.6359873	test: 14.5443780	best: 14.5443780 (813)	total: 28.6s	remaining: 1m 52s
    814:	learn: 11.6304762	test: 14.5416357	best: 14.5416357 (814)	total: 28.7s	remaining: 1m 52s
    815:	learn: 11.6271046	test: 14.5401521	best: 14.5401521 (815)	total: 28.7s	remaining: 1m 52s
    816:	learn: 11.6235959	test: 14.5398127	best: 14.5398127 (816)	total: 28.7s	remaining: 1m 51s
    817:	learn: 11.6220019	test: 14.5396379	best: 14.5396379 (817)	total: 28.8s	remaining: 1m 51s
    818:	learn: 11.6189069	test: 14.5394239	best: 14.5394239 (818)	total: 28.8s	remaining: 1m 51s
    819:	learn: 11.6171832	test: 14.5394499	best: 14.5394239 (818)	total: 28.8s	remaining: 1m 51s
    820:	learn: 11.6096716	test: 14.5341621	best: 14.5341621 (820)	total: 28.9s	remaining: 1m 51s
    821:	learn: 11.6073678	test: 14.5336727	best: 14.5336727 (821)	total: 28.9s	remaining: 1m 51s
    822:	learn: 11.6010742	test: 14.5283664	best: 14.5283664 (822)	total: 28.9s	remaining: 1m 51s
    823:	learn: 11.5994695	test: 14.5282579	best: 14.5282579 (823)	total: 29s	remaining: 1m 51s
    824:	learn: 11.5962101	test: 14.5273919	best: 14.5273919 (824)	total: 29s	remaining: 1m 51s
    825:	learn: 11.5905096	test: 14.5245146	best: 14.5245146 (825)	total: 29s	remaining: 1m 51s
    826:	learn: 11.5870237	test: 14.5216292	best: 14.5216292 (826)	total: 29.1s	remaining: 1m 51s
    827:	learn: 11.5850080	test: 14.5192763	best: 14.5192763 (827)	total: 29.1s	remaining: 1m 51s
    828:	learn: 11.5823303	test: 14.5168911	best: 14.5168911 (828)	total: 29.1s	remaining: 1m 51s
    829:	learn: 11.5796780	test: 14.5150507	best: 14.5150507 (829)	total: 29.2s	remaining: 1m 51s
    830:	learn: 11.5728364	test: 14.5105033	best: 14.5105033 (830)	total: 29.2s	remaining: 1m 51s
    831:	learn: 11.5704648	test: 14.5096060	best: 14.5096060 (831)	total: 29.2s	remaining: 1m 51s
    832:	learn: 11.5685641	test: 14.5069162	best: 14.5069162 (832)	total: 29.3s	remaining: 1m 51s
    833:	learn: 11.5658930	test: 14.5072082	best: 14.5069162 (832)	total: 29.3s	remaining: 1m 51s
    834:	learn: 11.5638209	test: 14.5065031	best: 14.5065031 (834)	total: 29.3s	remaining: 1m 51s
    835:	learn: 11.5613871	test: 14.5058062	best: 14.5058062 (835)	total: 29.4s	remaining: 1m 51s
    836:	learn: 11.5585450	test: 14.5050533	best: 14.5050533 (836)	total: 29.4s	remaining: 1m 51s
    837:	learn: 11.5558628	test: 14.5046382	best: 14.5046382 (837)	total: 29.4s	remaining: 1m 51s
    838:	learn: 11.5541419	test: 14.5030283	best: 14.5030283 (838)	total: 29.5s	remaining: 1m 50s
    839:	learn: 11.5504832	test: 14.5023340	best: 14.5023340 (839)	total: 29.5s	remaining: 1m 50s
    840:	learn: 11.5480297	test: 14.5030010	best: 14.5023340 (839)	total: 29.5s	remaining: 1m 50s
    841:	learn: 11.5433124	test: 14.5015193	best: 14.5015193 (841)	total: 29.6s	remaining: 1m 50s
    842:	learn: 11.5407079	test: 14.4998781	best: 14.4998781 (842)	total: 29.6s	remaining: 1m 50s
    843:	learn: 11.5396343	test: 14.5010410	best: 14.4998781 (842)	total: 29.6s	remaining: 1m 50s
    844:	learn: 11.5356215	test: 14.4996705	best: 14.4996705 (844)	total: 29.7s	remaining: 1m 50s
    845:	learn: 11.5338971	test: 14.4993845	best: 14.4993845 (845)	total: 29.7s	remaining: 1m 50s
    846:	learn: 11.5300378	test: 14.4984570	best: 14.4984570 (846)	total: 29.7s	remaining: 1m 50s
    847:	learn: 11.5292397	test: 14.4993749	best: 14.4984570 (846)	total: 29.8s	remaining: 1m 50s
    848:	learn: 11.5280739	test: 14.4990593	best: 14.4984570 (846)	total: 29.8s	remaining: 1m 50s
    849:	learn: 11.5267617	test: 14.4999639	best: 14.4984570 (846)	total: 29.8s	remaining: 1m 50s
    850:	learn: 11.5250589	test: 14.4993039	best: 14.4984570 (846)	total: 29.9s	remaining: 1m 50s
    851:	learn: 11.5217719	test: 14.4989312	best: 14.4984570 (846)	total: 29.9s	remaining: 1m 50s
    852:	learn: 11.5199447	test: 14.4994275	best: 14.4984570 (846)	total: 29.9s	remaining: 1m 50s
    853:	learn: 11.5167618	test: 14.4962966	best: 14.4962966 (853)	total: 30s	remaining: 1m 50s
    854:	learn: 11.5162596	test: 14.4961228	best: 14.4961228 (854)	total: 30s	remaining: 1m 50s
    855:	learn: 11.5141089	test: 14.4963965	best: 14.4961228 (854)	total: 30s	remaining: 1m 50s
    856:	learn: 11.5076757	test: 14.4907042	best: 14.4907042 (856)	total: 30s	remaining: 1m 50s
    857:	learn: 11.5065459	test: 14.4884824	best: 14.4884824 (857)	total: 30.1s	remaining: 1m 50s
    858:	learn: 11.5040605	test: 14.4878574	best: 14.4878574 (858)	total: 30.1s	remaining: 1m 50s
    859:	learn: 11.5016571	test: 14.4860194	best: 14.4860194 (859)	total: 30.1s	remaining: 1m 50s
    860:	learn: 11.5003195	test: 14.4872481	best: 14.4860194 (859)	total: 30.2s	remaining: 1m 49s
    861:	learn: 11.4975738	test: 14.4851745	best: 14.4851745 (861)	total: 30.2s	remaining: 1m 49s
    862:	learn: 11.4951851	test: 14.4846274	best: 14.4846274 (862)	total: 30.2s	remaining: 1m 49s
    863:	learn: 11.4936512	test: 14.4837615	best: 14.4837615 (863)	total: 30.3s	remaining: 1m 49s
    864:	learn: 11.4897294	test: 14.4816144	best: 14.4816144 (864)	total: 30.3s	remaining: 1m 49s
    865:	learn: 11.4859343	test: 14.4812489	best: 14.4812489 (865)	total: 30.3s	remaining: 1m 49s
    866:	learn: 11.4849185	test: 14.4804680	best: 14.4804680 (866)	total: 30.4s	remaining: 1m 49s
    867:	learn: 11.4816386	test: 14.4799899	best: 14.4799899 (867)	total: 30.4s	remaining: 1m 49s
    868:	learn: 11.4788123	test: 14.4788655	best: 14.4788655 (868)	total: 30.4s	remaining: 1m 49s
    869:	learn: 11.4781783	test: 14.4782576	best: 14.4782576 (869)	total: 30.5s	remaining: 1m 49s
    870:	learn: 11.4760832	test: 14.4789837	best: 14.4782576 (869)	total: 30.5s	remaining: 1m 49s
    871:	learn: 11.4753542	test: 14.4773775	best: 14.4773775 (871)	total: 30.6s	remaining: 1m 49s
    872:	learn: 11.4733963	test: 14.4794287	best: 14.4773775 (871)	total: 30.7s	remaining: 1m 49s
    873:	learn: 11.4675623	test: 14.4745922	best: 14.4745922 (873)	total: 30.7s	remaining: 1m 49s
    874:	learn: 11.4609235	test: 14.4687668	best: 14.4687668 (874)	total: 30.7s	remaining: 1m 49s
    875:	learn: 11.4591130	test: 14.4687319	best: 14.4687319 (875)	total: 30.8s	remaining: 1m 49s
    876:	learn: 11.4560315	test: 14.4675554	best: 14.4675554 (876)	total: 30.8s	remaining: 1m 49s
    877:	learn: 11.4543761	test: 14.4659003	best: 14.4659003 (877)	total: 30.9s	remaining: 1m 49s
    878:	learn: 11.4497220	test: 14.4643975	best: 14.4643975 (878)	total: 30.9s	remaining: 1m 49s
    879:	learn: 11.4474256	test: 14.4635810	best: 14.4635810 (879)	total: 30.9s	remaining: 1m 49s
    880:	learn: 11.4435556	test: 14.4619557	best: 14.4619557 (880)	total: 31s	remaining: 1m 49s
    881:	learn: 11.4419938	test: 14.4631019	best: 14.4619557 (880)	total: 31.1s	remaining: 1m 49s
    882:	learn: 11.4382111	test: 14.4626776	best: 14.4619557 (880)	total: 31.2s	remaining: 1m 50s
    883:	learn: 11.4366597	test: 14.4621823	best: 14.4619557 (880)	total: 31.2s	remaining: 1m 50s
    884:	learn: 11.4293127	test: 14.4580628	best: 14.4580628 (884)	total: 31.3s	remaining: 1m 50s
    885:	learn: 11.4214900	test: 14.4536150	best: 14.4536150 (885)	total: 31.3s	remaining: 1m 49s
    886:	learn: 11.4195128	test: 14.4540383	best: 14.4536150 (885)	total: 31.3s	remaining: 1m 49s
    887:	learn: 11.4169899	test: 14.4521859	best: 14.4521859 (887)	total: 31.4s	remaining: 1m 49s
    888:	learn: 11.4148818	test: 14.4501844	best: 14.4501844 (888)	total: 31.4s	remaining: 1m 49s
    889:	learn: 11.4131315	test: 14.4494638	best: 14.4494638 (889)	total: 31.4s	remaining: 1m 49s
    890:	learn: 11.4098145	test: 14.4492323	best: 14.4492323 (890)	total: 31.5s	remaining: 1m 49s
    891:	learn: 11.4083555	test: 14.4481538	best: 14.4481538 (891)	total: 31.5s	remaining: 1m 49s
    892:	learn: 11.4063655	test: 14.4476274	best: 14.4476274 (892)	total: 31.5s	remaining: 1m 49s
    893:	learn: 11.4047988	test: 14.4474268	best: 14.4474268 (893)	total: 31.6s	remaining: 1m 49s
    894:	learn: 11.4027476	test: 14.4465714	best: 14.4465714 (894)	total: 31.6s	remaining: 1m 49s
    895:	learn: 11.4010674	test: 14.4460272	best: 14.4460272 (895)	total: 31.6s	remaining: 1m 49s
    896:	learn: 11.3991545	test: 14.4466183	best: 14.4460272 (895)	total: 31.7s	remaining: 1m 49s
    897:	learn: 11.3955427	test: 14.4450561	best: 14.4450561 (897)	total: 31.7s	remaining: 1m 49s
    898:	learn: 11.3897861	test: 14.4419368	best: 14.4419368 (898)	total: 31.7s	remaining: 1m 49s
    899:	learn: 11.3862653	test: 14.4387315	best: 14.4387315 (899)	total: 31.8s	remaining: 1m 49s
    900:	learn: 11.3791049	test: 14.4317508	best: 14.4317508 (900)	total: 31.8s	remaining: 1m 49s
    901:	learn: 11.3770259	test: 14.4317400	best: 14.4317400 (901)	total: 31.8s	remaining: 1m 49s
    902:	learn: 11.3732729	test: 14.4315474	best: 14.4315474 (902)	total: 31.9s	remaining: 1m 49s
    903:	learn: 11.3721731	test: 14.4310079	best: 14.4310079 (903)	total: 31.9s	remaining: 1m 49s
    904:	learn: 11.3676809	test: 14.4293083	best: 14.4293083 (904)	total: 31.9s	remaining: 1m 49s
    905:	learn: 11.3668734	test: 14.4286964	best: 14.4286964 (905)	total: 32s	remaining: 1m 49s
    906:	learn: 11.3653245	test: 14.4286618	best: 14.4286618 (906)	total: 32s	remaining: 1m 49s
    907:	learn: 11.3637937	test: 14.4291299	best: 14.4286618 (906)	total: 32s	remaining: 1m 49s
    908:	learn: 11.3588464	test: 14.4281776	best: 14.4281776 (908)	total: 32.1s	remaining: 1m 49s
    909:	learn: 11.3550865	test: 14.4260648	best: 14.4260648 (909)	total: 32.1s	remaining: 1m 48s
    910:	learn: 11.3541595	test: 14.4247542	best: 14.4247542 (910)	total: 32.1s	remaining: 1m 48s
    911:	learn: 11.3509172	test: 14.4238003	best: 14.4238003 (911)	total: 32.2s	remaining: 1m 48s
    912:	learn: 11.3498702	test: 14.4242303	best: 14.4238003 (911)	total: 32.2s	remaining: 1m 48s
    913:	learn: 11.3461624	test: 14.4198087	best: 14.4198087 (913)	total: 32.2s	remaining: 1m 48s
    914:	learn: 11.3413577	test: 14.4169522	best: 14.4169522 (914)	total: 32.3s	remaining: 1m 48s
    915:	learn: 11.3389501	test: 14.4157773	best: 14.4157773 (915)	total: 32.3s	remaining: 1m 48s
    916:	learn: 11.3364668	test: 14.4146576	best: 14.4146576 (916)	total: 32.3s	remaining: 1m 48s
    917:	learn: 11.3347307	test: 14.4135294	best: 14.4135294 (917)	total: 32.4s	remaining: 1m 48s
    918:	learn: 11.3337635	test: 14.4129106	best: 14.4129106 (918)	total: 32.4s	remaining: 1m 48s
    919:	learn: 11.3311170	test: 14.4121541	best: 14.4121541 (919)	total: 32.4s	remaining: 1m 48s
    920:	learn: 11.3298589	test: 14.4119165	best: 14.4119165 (920)	total: 32.5s	remaining: 1m 48s
    921:	learn: 11.3292589	test: 14.4103735	best: 14.4103735 (921)	total: 32.5s	remaining: 1m 48s
    922:	learn: 11.3268260	test: 14.4093081	best: 14.4093081 (922)	total: 32.5s	remaining: 1m 48s
    923:	learn: 11.3258526	test: 14.4093031	best: 14.4093031 (923)	total: 32.6s	remaining: 1m 48s
    924:	learn: 11.3236302	test: 14.4095111	best: 14.4093031 (923)	total: 32.6s	remaining: 1m 48s
    925:	learn: 11.3211314	test: 14.4074442	best: 14.4074442 (925)	total: 32.6s	remaining: 1m 48s
    926:	learn: 11.3204785	test: 14.4069103	best: 14.4069103 (926)	total: 32.7s	remaining: 1m 48s
    927:	learn: 11.3154647	test: 14.4057679	best: 14.4057679 (927)	total: 32.7s	remaining: 1m 48s
    928:	learn: 11.3132755	test: 14.4049662	best: 14.4049662 (928)	total: 32.7s	remaining: 1m 48s
    929:	learn: 11.3092599	test: 14.4036387	best: 14.4036387 (929)	total: 32.8s	remaining: 1m 48s
    930:	learn: 11.3075072	test: 14.4012151	best: 14.4012151 (930)	total: 32.8s	remaining: 1m 48s
    931:	learn: 11.3059552	test: 14.4007397	best: 14.4007397 (931)	total: 32.8s	remaining: 1m 48s
    932:	learn: 11.3028286	test: 14.3984893	best: 14.3984893 (932)	total: 32.9s	remaining: 1m 47s
    933:	learn: 11.2991372	test: 14.3970522	best: 14.3970522 (933)	total: 32.9s	remaining: 1m 47s
    934:	learn: 11.2926950	test: 14.3925113	best: 14.3925113 (934)	total: 32.9s	remaining: 1m 47s
    935:	learn: 11.2914208	test: 14.3924900	best: 14.3924900 (935)	total: 33s	remaining: 1m 47s
    936:	learn: 11.2859890	test: 14.3885764	best: 14.3885764 (936)	total: 33s	remaining: 1m 47s
    937:	learn: 11.2842141	test: 14.3871494	best: 14.3871494 (937)	total: 33s	remaining: 1m 47s
    938:	learn: 11.2793799	test: 14.3850098	best: 14.3850098 (938)	total: 33.1s	remaining: 1m 47s
    939:	learn: 11.2779885	test: 14.3846785	best: 14.3846785 (939)	total: 33.1s	remaining: 1m 47s
    940:	learn: 11.2758792	test: 14.3816555	best: 14.3816555 (940)	total: 33.1s	remaining: 1m 47s
    941:	learn: 11.2749875	test: 14.3807317	best: 14.3807317 (941)	total: 33.2s	remaining: 1m 47s
    942:	learn: 11.2727572	test: 14.3806995	best: 14.3806995 (942)	total: 33.2s	remaining: 1m 47s
    943:	learn: 11.2709630	test: 14.3799630	best: 14.3799630 (943)	total: 33.2s	remaining: 1m 47s
    944:	learn: 11.2690205	test: 14.3798648	best: 14.3798648 (944)	total: 33.3s	remaining: 1m 47s
    945:	learn: 11.2661594	test: 14.3785348	best: 14.3785348 (945)	total: 33.3s	remaining: 1m 47s
    946:	learn: 11.2648110	test: 14.3787891	best: 14.3785348 (945)	total: 33.3s	remaining: 1m 47s
    947:	learn: 11.2604028	test: 14.3794645	best: 14.3785348 (945)	total: 33.4s	remaining: 1m 47s
    948:	learn: 11.2574837	test: 14.3777770	best: 14.3777770 (948)	total: 33.4s	remaining: 1m 47s
    949:	learn: 11.2566347	test: 14.3778069	best: 14.3777770 (948)	total: 33.4s	remaining: 1m 47s
    950:	learn: 11.2548401	test: 14.3776249	best: 14.3776249 (950)	total: 33.5s	remaining: 1m 47s
    951:	learn: 11.2528163	test: 14.3773056	best: 14.3773056 (951)	total: 33.5s	remaining: 1m 47s
    952:	learn: 11.2488247	test: 14.3755792	best: 14.3755792 (952)	total: 33.5s	remaining: 1m 47s
    953:	learn: 11.2478594	test: 14.3749834	best: 14.3749834 (953)	total: 33.6s	remaining: 1m 47s
    954:	learn: 11.2442930	test: 14.3741590	best: 14.3741590 (954)	total: 33.6s	remaining: 1m 47s
    955:	learn: 11.2406161	test: 14.3741281	best: 14.3741281 (955)	total: 33.6s	remaining: 1m 47s
    956:	learn: 11.2382032	test: 14.3735260	best: 14.3735260 (956)	total: 33.7s	remaining: 1m 47s
    957:	learn: 11.2363952	test: 14.3734504	best: 14.3734504 (957)	total: 33.8s	remaining: 1m 47s
    958:	learn: 11.2345590	test: 14.3716406	best: 14.3716406 (958)	total: 33.8s	remaining: 1m 47s
    959:	learn: 11.2304306	test: 14.3711666	best: 14.3711666 (959)	total: 33.9s	remaining: 1m 47s
    960:	learn: 11.2288264	test: 14.3697073	best: 14.3697073 (960)	total: 33.9s	remaining: 1m 47s
    961:	learn: 11.2269845	test: 14.3691786	best: 14.3691786 (961)	total: 33.9s	remaining: 1m 47s
    962:	learn: 11.2250238	test: 14.3679240	best: 14.3679240 (962)	total: 34s	remaining: 1m 47s
    963:	learn: 11.2240701	test: 14.3673565	best: 14.3673565 (963)	total: 34s	remaining: 1m 47s
    964:	learn: 11.2218555	test: 14.3687904	best: 14.3673565 (963)	total: 34s	remaining: 1m 47s
    965:	learn: 11.2202054	test: 14.3678813	best: 14.3673565 (963)	total: 34.1s	remaining: 1m 46s
    966:	learn: 11.2189365	test: 14.3679064	best: 14.3673565 (963)	total: 34.1s	remaining: 1m 46s
    967:	learn: 11.2147920	test: 14.3658804	best: 14.3658804 (967)	total: 34.1s	remaining: 1m 46s
    968:	learn: 11.2107708	test: 14.3637071	best: 14.3637071 (968)	total: 34.2s	remaining: 1m 47s
    969:	learn: 11.2076446	test: 14.3634114	best: 14.3634114 (969)	total: 34.3s	remaining: 1m 47s
    970:	learn: 11.2068284	test: 14.3635455	best: 14.3634114 (969)	total: 34.3s	remaining: 1m 47s
    971:	learn: 11.2055308	test: 14.3629591	best: 14.3629591 (971)	total: 34.4s	remaining: 1m 47s
    972:	learn: 11.2019848	test: 14.3607834	best: 14.3607834 (972)	total: 34.4s	remaining: 1m 46s
    973:	learn: 11.2000820	test: 14.3595569	best: 14.3595569 (973)	total: 34.4s	remaining: 1m 46s
    974:	learn: 11.1983735	test: 14.3592949	best: 14.3592949 (974)	total: 34.5s	remaining: 1m 46s
    975:	learn: 11.1962096	test: 14.3583340	best: 14.3583340 (975)	total: 34.5s	remaining: 1m 46s
    976:	learn: 11.1904957	test: 14.3549274	best: 14.3549274 (976)	total: 34.5s	remaining: 1m 46s
    977:	learn: 11.1882379	test: 14.3552016	best: 14.3549274 (976)	total: 34.5s	remaining: 1m 46s
    978:	learn: 11.1861756	test: 14.3543809	best: 14.3543809 (978)	total: 34.6s	remaining: 1m 46s
    979:	learn: 11.1853738	test: 14.3551940	best: 14.3543809 (978)	total: 34.6s	remaining: 1m 46s
    980:	learn: 11.1839681	test: 14.3546603	best: 14.3543809 (978)	total: 34.6s	remaining: 1m 46s
    981:	learn: 11.1833724	test: 14.3546074	best: 14.3543809 (978)	total: 34.7s	remaining: 1m 46s
    982:	learn: 11.1821106	test: 14.3531335	best: 14.3531335 (982)	total: 34.7s	remaining: 1m 46s
    983:	learn: 11.1809665	test: 14.3532848	best: 14.3531335 (982)	total: 34.7s	remaining: 1m 46s
    984:	learn: 11.1766018	test: 14.3512703	best: 14.3512703 (984)	total: 34.8s	remaining: 1m 46s
    985:	learn: 11.1756824	test: 14.3516701	best: 14.3512703 (984)	total: 34.8s	remaining: 1m 46s
    986:	learn: 11.1745151	test: 14.3516144	best: 14.3512703 (984)	total: 34.9s	remaining: 1m 46s
    987:	learn: 11.1714063	test: 14.3545213	best: 14.3512703 (984)	total: 35s	remaining: 1m 46s
    988:	learn: 11.1684079	test: 14.3523083	best: 14.3512703 (984)	total: 35s	remaining: 1m 46s
    989:	learn: 11.1662197	test: 14.3534846	best: 14.3512703 (984)	total: 35s	remaining: 1m 46s
    990:	learn: 11.1645671	test: 14.3529870	best: 14.3512703 (984)	total: 35.1s	remaining: 1m 46s
    991:	learn: 11.1611347	test: 14.3509504	best: 14.3509504 (991)	total: 35.1s	remaining: 1m 46s
    992:	learn: 11.1595188	test: 14.3501492	best: 14.3501492 (992)	total: 35.1s	remaining: 1m 46s
    993:	learn: 11.1579814	test: 14.3502374	best: 14.3501492 (992)	total: 35.2s	remaining: 1m 46s
    994:	learn: 11.1559225	test: 14.3494286	best: 14.3494286 (994)	total: 35.2s	remaining: 1m 46s
    995:	learn: 11.1543307	test: 14.3494366	best: 14.3494286 (994)	total: 35.2s	remaining: 1m 46s
    996:	learn: 11.1521232	test: 14.3474436	best: 14.3474436 (996)	total: 35.3s	remaining: 1m 46s
    997:	learn: 11.1505497	test: 14.3471488	best: 14.3471488 (997)	total: 35.3s	remaining: 1m 46s
    998:	learn: 11.1478876	test: 14.3446284	best: 14.3446284 (998)	total: 35.3s	remaining: 1m 46s
    999:	learn: 11.1427962	test: 14.3423671	best: 14.3423671 (999)	total: 35.4s	remaining: 1m 46s
    1000:	learn: 11.1401767	test: 14.3424097	best: 14.3423671 (999)	total: 35.5s	remaining: 1m 46s
    1001:	learn: 11.1384433	test: 14.3421177	best: 14.3421177 (1001)	total: 35.5s	remaining: 1m 46s
    1002:	learn: 11.1307079	test: 14.3381760	best: 14.3381760 (1002)	total: 35.6s	remaining: 1m 46s
    1003:	learn: 11.1233446	test: 14.3328372	best: 14.3328372 (1003)	total: 35.6s	remaining: 1m 46s
    1004:	learn: 11.1168374	test: 14.3277600	best: 14.3277600 (1004)	total: 35.6s	remaining: 1m 46s
    1005:	learn: 11.1156361	test: 14.3283054	best: 14.3277600 (1004)	total: 35.7s	remaining: 1m 46s
    1006:	learn: 11.1134670	test: 14.3295546	best: 14.3277600 (1004)	total: 35.7s	remaining: 1m 46s
    1007:	learn: 11.1119331	test: 14.3279274	best: 14.3277600 (1004)	total: 35.8s	remaining: 1m 46s
    1008:	learn: 11.1096269	test: 14.3290037	best: 14.3277600 (1004)	total: 35.8s	remaining: 1m 46s
    1009:	learn: 11.1084837	test: 14.3287233	best: 14.3277600 (1004)	total: 35.8s	remaining: 1m 46s
    1010:	learn: 11.1073473	test: 14.3275120	best: 14.3275120 (1010)	total: 35.8s	remaining: 1m 45s
    1011:	learn: 11.1067400	test: 14.3277231	best: 14.3275120 (1010)	total: 35.9s	remaining: 1m 45s
    1012:	learn: 11.1025956	test: 14.3253270	best: 14.3253270 (1012)	total: 35.9s	remaining: 1m 45s
    1013:	learn: 11.1012255	test: 14.3240331	best: 14.3240331 (1013)	total: 36s	remaining: 1m 45s
    1014:	learn: 11.0989041	test: 14.3235743	best: 14.3235743 (1014)	total: 36s	remaining: 1m 45s
    1015:	learn: 11.0981898	test: 14.3242471	best: 14.3235743 (1014)	total: 36s	remaining: 1m 45s
    1016:	learn: 11.0949012	test: 14.3237366	best: 14.3235743 (1014)	total: 36.1s	remaining: 1m 45s
    1017:	learn: 11.0939197	test: 14.3245923	best: 14.3235743 (1014)	total: 36.1s	remaining: 1m 45s
    1018:	learn: 11.0921165	test: 14.3238806	best: 14.3235743 (1014)	total: 36.1s	remaining: 1m 45s
    1019:	learn: 11.0900916	test: 14.3218703	best: 14.3218703 (1019)	total: 36.2s	remaining: 1m 45s
    1020:	learn: 11.0877956	test: 14.3210602	best: 14.3210602 (1020)	total: 36.2s	remaining: 1m 45s
    1021:	learn: 11.0863864	test: 14.3218824	best: 14.3210602 (1020)	total: 36.2s	remaining: 1m 45s
    1022:	learn: 11.0833612	test: 14.3190876	best: 14.3190876 (1022)	total: 36.3s	remaining: 1m 45s
    1023:	learn: 11.0814245	test: 14.3188126	best: 14.3188126 (1023)	total: 36.3s	remaining: 1m 45s
    1024:	learn: 11.0783557	test: 14.3165864	best: 14.3165864 (1024)	total: 36.3s	remaining: 1m 45s
    1025:	learn: 11.0756393	test: 14.3159360	best: 14.3159360 (1025)	total: 36.4s	remaining: 1m 45s
    1026:	learn: 11.0712688	test: 14.3131959	best: 14.3131959 (1026)	total: 36.4s	remaining: 1m 45s
    1027:	learn: 11.0698399	test: 14.3129207	best: 14.3129207 (1027)	total: 36.4s	remaining: 1m 45s
    1028:	learn: 11.0674998	test: 14.3120757	best: 14.3120757 (1028)	total: 36.5s	remaining: 1m 45s
    1029:	learn: 11.0668179	test: 14.3121970	best: 14.3120757 (1028)	total: 36.5s	remaining: 1m 45s
    1030:	learn: 11.0644458	test: 14.3097719	best: 14.3097719 (1030)	total: 36.5s	remaining: 1m 45s
    1031:	learn: 11.0625109	test: 14.3103819	best: 14.3097719 (1030)	total: 36.6s	remaining: 1m 45s
    1032:	learn: 11.0617389	test: 14.3099013	best: 14.3097719 (1030)	total: 36.6s	remaining: 1m 45s
    1033:	learn: 11.0602560	test: 14.3082009	best: 14.3082009 (1033)	total: 36.6s	remaining: 1m 45s
    1034:	learn: 11.0591802	test: 14.3085446	best: 14.3082009 (1033)	total: 36.7s	remaining: 1m 45s
    1035:	learn: 11.0582668	test: 14.3076830	best: 14.3076830 (1035)	total: 36.8s	remaining: 1m 45s
    1036:	learn: 11.0573531	test: 14.3082563	best: 14.3076830 (1035)	total: 36.8s	remaining: 1m 45s
    1037:	learn: 11.0565261	test: 14.3071169	best: 14.3071169 (1037)	total: 36.9s	remaining: 1m 45s
    1038:	learn: 11.0506692	test: 14.3036169	best: 14.3036169 (1038)	total: 37s	remaining: 1m 45s
    1039:	learn: 11.0489727	test: 14.3037585	best: 14.3036169 (1038)	total: 37s	remaining: 1m 45s
    1040:	learn: 11.0477876	test: 14.3030861	best: 14.3030861 (1040)	total: 37s	remaining: 1m 45s
    1041:	learn: 11.0455857	test: 14.3036468	best: 14.3030861 (1040)	total: 37s	remaining: 1m 45s
    1042:	learn: 11.0449209	test: 14.3038196	best: 14.3030861 (1040)	total: 37.1s	remaining: 1m 45s
    1043:	learn: 11.0382481	test: 14.2988493	best: 14.2988493 (1043)	total: 37.1s	remaining: 1m 45s
    1044:	learn: 11.0368472	test: 14.2982242	best: 14.2982242 (1044)	total: 37.1s	remaining: 1m 44s
    1045:	learn: 11.0351371	test: 14.2968559	best: 14.2968559 (1045)	total: 37.2s	remaining: 1m 44s
    1046:	learn: 11.0336025	test: 14.2956467	best: 14.2956467 (1046)	total: 37.2s	remaining: 1m 44s
    1047:	learn: 11.0323795	test: 14.2956806	best: 14.2956467 (1046)	total: 37.2s	remaining: 1m 44s
    1048:	learn: 11.0283735	test: 14.2921991	best: 14.2921991 (1048)	total: 37.3s	remaining: 1m 44s
    1049:	learn: 11.0269826	test: 14.2915525	best: 14.2915525 (1049)	total: 37.3s	remaining: 1m 44s
    1050:	learn: 11.0240215	test: 14.2922015	best: 14.2915525 (1049)	total: 37.3s	remaining: 1m 44s
    1051:	learn: 11.0231968	test: 14.2922772	best: 14.2915525 (1049)	total: 37.3s	remaining: 1m 44s
    1052:	learn: 11.0221655	test: 14.2936072	best: 14.2915525 (1049)	total: 37.4s	remaining: 1m 44s
    1053:	learn: 11.0213939	test: 14.2935878	best: 14.2915525 (1049)	total: 37.4s	remaining: 1m 44s
    1054:	learn: 11.0208871	test: 14.2939850	best: 14.2915525 (1049)	total: 37.4s	remaining: 1m 44s
    1055:	learn: 11.0151465	test: 14.2909872	best: 14.2909872 (1055)	total: 37.5s	remaining: 1m 44s
    1056:	learn: 11.0126984	test: 14.2871093	best: 14.2871093 (1056)	total: 37.5s	remaining: 1m 44s
    1057:	learn: 11.0080855	test: 14.2851263	best: 14.2851263 (1057)	total: 37.6s	remaining: 1m 44s
    1058:	learn: 11.0030257	test: 14.2843319	best: 14.2843319 (1058)	total: 37.6s	remaining: 1m 44s
    1059:	learn: 11.0019888	test: 14.2831102	best: 14.2831102 (1059)	total: 37.7s	remaining: 1m 44s
    1060:	learn: 11.0012582	test: 14.2840580	best: 14.2831102 (1059)	total: 37.7s	remaining: 1m 44s
    1061:	learn: 10.9979448	test: 14.2831293	best: 14.2831102 (1059)	total: 37.7s	remaining: 1m 44s
    1062:	learn: 10.9963646	test: 14.2833480	best: 14.2831102 (1059)	total: 37.8s	remaining: 1m 44s
    1063:	learn: 10.9934271	test: 14.2834251	best: 14.2831102 (1059)	total: 37.8s	remaining: 1m 44s
    1064:	learn: 10.9923893	test: 14.2832469	best: 14.2831102 (1059)	total: 37.8s	remaining: 1m 44s
    1065:	learn: 10.9895790	test: 14.2806153	best: 14.2806153 (1065)	total: 37.8s	remaining: 1m 44s
    1066:	learn: 10.9870812	test: 14.2806688	best: 14.2806153 (1065)	total: 37.9s	remaining: 1m 44s
    1067:	learn: 10.9826968	test: 14.2787643	best: 14.2787643 (1067)	total: 37.9s	remaining: 1m 44s
    1068:	learn: 10.9815064	test: 14.2780529	best: 14.2780529 (1068)	total: 37.9s	remaining: 1m 44s
    1069:	learn: 10.9802547	test: 14.2789188	best: 14.2780529 (1068)	total: 38s	remaining: 1m 44s
    1070:	learn: 10.9769720	test: 14.2796028	best: 14.2780529 (1068)	total: 38s	remaining: 1m 43s
    1071:	learn: 10.9756672	test: 14.2787096	best: 14.2780529 (1068)	total: 38s	remaining: 1m 43s
    1072:	learn: 10.9745572	test: 14.2795845	best: 14.2780529 (1068)	total: 38.1s	remaining: 1m 43s
    1073:	learn: 10.9710103	test: 14.2801067	best: 14.2780529 (1068)	total: 38.1s	remaining: 1m 43s
    1074:	learn: 10.9699739	test: 14.2799376	best: 14.2780529 (1068)	total: 38.1s	remaining: 1m 43s
    1075:	learn: 10.9690604	test: 14.2799732	best: 14.2780529 (1068)	total: 38.2s	remaining: 1m 43s
    1076:	learn: 10.9670140	test: 14.2776074	best: 14.2776074 (1076)	total: 38.2s	remaining: 1m 43s
    1077:	learn: 10.9634331	test: 14.2778271	best: 14.2776074 (1076)	total: 38.2s	remaining: 1m 43s
    1078:	learn: 10.9622728	test: 14.2763915	best: 14.2763915 (1078)	total: 38.3s	remaining: 1m 43s
    1079:	learn: 10.9604097	test: 14.2746808	best: 14.2746808 (1079)	total: 38.3s	remaining: 1m 43s
    1080:	learn: 10.9556604	test: 14.2728302	best: 14.2728302 (1080)	total: 38.3s	remaining: 1m 43s
    1081:	learn: 10.9544546	test: 14.2742651	best: 14.2728302 (1080)	total: 38.4s	remaining: 1m 43s
    1082:	learn: 10.9524386	test: 14.2743101	best: 14.2728302 (1080)	total: 38.4s	remaining: 1m 43s
    1083:	learn: 10.9506536	test: 14.2755993	best: 14.2728302 (1080)	total: 38.4s	remaining: 1m 43s
    1084:	learn: 10.9481845	test: 14.2754477	best: 14.2728302 (1080)	total: 38.4s	remaining: 1m 43s
    1085:	learn: 10.9473177	test: 14.2753945	best: 14.2728302 (1080)	total: 38.5s	remaining: 1m 43s
    1086:	learn: 10.9467606	test: 14.2754649	best: 14.2728302 (1080)	total: 38.5s	remaining: 1m 43s
    1087:	learn: 10.9371876	test: 14.2697870	best: 14.2697870 (1087)	total: 38.5s	remaining: 1m 43s
    1088:	learn: 10.9359998	test: 14.2697549	best: 14.2697549 (1088)	total: 38.5s	remaining: 1m 43s
    1089:	learn: 10.9328862	test: 14.2690525	best: 14.2690525 (1089)	total: 38.6s	remaining: 1m 42s
    1090:	learn: 10.9312086	test: 14.2672428	best: 14.2672428 (1090)	total: 38.6s	remaining: 1m 42s
    1091:	learn: 10.9288386	test: 14.2682469	best: 14.2672428 (1090)	total: 38.6s	remaining: 1m 42s
    1092:	learn: 10.9275990	test: 14.2674279	best: 14.2672428 (1090)	total: 38.6s	remaining: 1m 42s
    1093:	learn: 10.9262285	test: 14.2661016	best: 14.2661016 (1093)	total: 38.7s	remaining: 1m 42s
    1094:	learn: 10.9254614	test: 14.2660244	best: 14.2660244 (1094)	total: 38.7s	remaining: 1m 42s
    1095:	learn: 10.9193177	test: 14.2646927	best: 14.2646927 (1095)	total: 38.7s	remaining: 1m 42s
    1096:	learn: 10.9159466	test: 14.2623118	best: 14.2623118 (1096)	total: 38.8s	remaining: 1m 42s
    1097:	learn: 10.9117343	test: 14.2613786	best: 14.2613786 (1097)	total: 38.8s	remaining: 1m 42s
    1098:	learn: 10.9039608	test: 14.2554659	best: 14.2554659 (1098)	total: 38.9s	remaining: 1m 42s
    1099:	learn: 10.9014191	test: 14.2556564	best: 14.2554659 (1098)	total: 38.9s	remaining: 1m 42s
    1100:	learn: 10.8996961	test: 14.2554201	best: 14.2554201 (1100)	total: 38.9s	remaining: 1m 42s
    1101:	learn: 10.8980339	test: 14.2573786	best: 14.2554201 (1100)	total: 39s	remaining: 1m 42s
    1102:	learn: 10.8966814	test: 14.2574783	best: 14.2554201 (1100)	total: 39s	remaining: 1m 42s
    1103:	learn: 10.8947851	test: 14.2583101	best: 14.2554201 (1100)	total: 39s	remaining: 1m 42s
    1104:	learn: 10.8931324	test: 14.2574141	best: 14.2554201 (1100)	total: 39.1s	remaining: 1m 42s
    1105:	learn: 10.8903470	test: 14.2556367	best: 14.2554201 (1100)	total: 39.1s	remaining: 1m 42s
    1106:	learn: 10.8888927	test: 14.2568454	best: 14.2554201 (1100)	total: 39.1s	remaining: 1m 42s
    1107:	learn: 10.8859455	test: 14.2542843	best: 14.2542843 (1107)	total: 39.2s	remaining: 1m 42s
    1108:	learn: 10.8825372	test: 14.2521627	best: 14.2521627 (1108)	total: 39.2s	remaining: 1m 42s
    1109:	learn: 10.8808922	test: 14.2523502	best: 14.2521627 (1108)	total: 39.3s	remaining: 1m 42s
    1110:	learn: 10.8735035	test: 14.2478763	best: 14.2478763 (1110)	total: 39.3s	remaining: 1m 42s
    1111:	learn: 10.8716332	test: 14.2489034	best: 14.2478763 (1110)	total: 39.4s	remaining: 1m 42s
    1112:	learn: 10.8693247	test: 14.2460118	best: 14.2460118 (1112)	total: 39.4s	remaining: 1m 42s
    1113:	learn: 10.8679220	test: 14.2452232	best: 14.2452232 (1113)	total: 39.4s	remaining: 1m 42s
    1114:	learn: 10.8659559	test: 14.2456338	best: 14.2452232 (1113)	total: 39.4s	remaining: 1m 42s
    1115:	learn: 10.8653578	test: 14.2456933	best: 14.2452232 (1113)	total: 39.5s	remaining: 1m 41s
    1116:	learn: 10.8634637	test: 14.2457938	best: 14.2452232 (1113)	total: 39.5s	remaining: 1m 41s
    1117:	learn: 10.8624509	test: 14.2457182	best: 14.2452232 (1113)	total: 39.5s	remaining: 1m 41s
    1118:	learn: 10.8599416	test: 14.2455021	best: 14.2452232 (1113)	total: 39.5s	remaining: 1m 41s
    1119:	learn: 10.8589406	test: 14.2444357	best: 14.2444357 (1119)	total: 39.6s	remaining: 1m 41s
    1120:	learn: 10.8577529	test: 14.2435846	best: 14.2435846 (1120)	total: 39.6s	remaining: 1m 41s
    1121:	learn: 10.8559231	test: 14.2424616	best: 14.2424616 (1121)	total: 39.6s	remaining: 1m 41s
    1122:	learn: 10.8523507	test: 14.2430505	best: 14.2424616 (1121)	total: 39.7s	remaining: 1m 41s
    1123:	learn: 10.8515111	test: 14.2429089	best: 14.2424616 (1121)	total: 39.7s	remaining: 1m 41s
    1124:	learn: 10.8496339	test: 14.2418130	best: 14.2418130 (1124)	total: 39.7s	remaining: 1m 41s
    1125:	learn: 10.8465987	test: 14.2424179	best: 14.2418130 (1124)	total: 39.8s	remaining: 1m 41s
    1126:	learn: 10.8457443	test: 14.2424183	best: 14.2418130 (1124)	total: 39.8s	remaining: 1m 41s
    1127:	learn: 10.8396840	test: 14.2378652	best: 14.2378652 (1127)	total: 39.8s	remaining: 1m 41s
    1128:	learn: 10.8366912	test: 14.2371343	best: 14.2371343 (1128)	total: 39.8s	remaining: 1m 41s
    1129:	learn: 10.8334904	test: 14.2357668	best: 14.2357668 (1129)	total: 39.9s	remaining: 1m 41s
    1130:	learn: 10.8289804	test: 14.2308071	best: 14.2308071 (1130)	total: 39.9s	remaining: 1m 41s
    1131:	learn: 10.8275542	test: 14.2313191	best: 14.2308071 (1130)	total: 40s	remaining: 1m 41s
    1132:	learn: 10.8219592	test: 14.2289247	best: 14.2289247 (1132)	total: 40s	remaining: 1m 41s
    1133:	learn: 10.8205744	test: 14.2293229	best: 14.2289247 (1132)	total: 40.1s	remaining: 1m 41s
    1134:	learn: 10.8146286	test: 14.2283702	best: 14.2283702 (1134)	total: 40.1s	remaining: 1m 41s
    1135:	learn: 10.8136808	test: 14.2291525	best: 14.2283702 (1134)	total: 40.1s	remaining: 1m 41s
    1136:	learn: 10.8124582	test: 14.2286525	best: 14.2283702 (1134)	total: 40.2s	remaining: 1m 41s
    1137:	learn: 10.8114875	test: 14.2273602	best: 14.2273602 (1137)	total: 40.2s	remaining: 1m 41s
    1138:	learn: 10.8040679	test: 14.2227980	best: 14.2227980 (1138)	total: 40.3s	remaining: 1m 41s
    1139:	learn: 10.8016167	test: 14.2242095	best: 14.2227980 (1138)	total: 40.3s	remaining: 1m 41s
    1140:	learn: 10.8012048	test: 14.2237666	best: 14.2227980 (1138)	total: 40.3s	remaining: 1m 41s
    1141:	learn: 10.7988262	test: 14.2249552	best: 14.2227980 (1138)	total: 40.3s	remaining: 1m 40s
    1142:	learn: 10.7969705	test: 14.2231595	best: 14.2227980 (1138)	total: 40.4s	remaining: 1m 40s
    1143:	learn: 10.7954742	test: 14.2223670	best: 14.2223670 (1143)	total: 40.4s	remaining: 1m 40s
    1144:	learn: 10.7941057	test: 14.2215106	best: 14.2215106 (1144)	total: 40.4s	remaining: 1m 40s
    1145:	learn: 10.7895307	test: 14.2182222	best: 14.2182222 (1145)	total: 40.5s	remaining: 1m 40s
    1146:	learn: 10.7873137	test: 14.2182011	best: 14.2182011 (1146)	total: 40.5s	remaining: 1m 40s
    1147:	learn: 10.7830554	test: 14.2154033	best: 14.2154033 (1147)	total: 40.5s	remaining: 1m 40s
    1148:	learn: 10.7823863	test: 14.2164379	best: 14.2154033 (1147)	total: 40.5s	remaining: 1m 40s
    1149:	learn: 10.7796207	test: 14.2153072	best: 14.2153072 (1149)	total: 40.6s	remaining: 1m 40s
    1150:	learn: 10.7756524	test: 14.2118589	best: 14.2118589 (1150)	total: 40.6s	remaining: 1m 40s
    1151:	learn: 10.7725567	test: 14.2112982	best: 14.2112982 (1151)	total: 40.6s	remaining: 1m 40s
    1152:	learn: 10.7699213	test: 14.2116679	best: 14.2112982 (1151)	total: 40.6s	remaining: 1m 40s
    1153:	learn: 10.7686412	test: 14.2113887	best: 14.2112982 (1151)	total: 40.7s	remaining: 1m 40s
    1154:	learn: 10.7667495	test: 14.2108563	best: 14.2108563 (1154)	total: 40.7s	remaining: 1m 40s
    1155:	learn: 10.7661470	test: 14.2099925	best: 14.2099925 (1155)	total: 40.7s	remaining: 1m 40s
    1156:	learn: 10.7646744	test: 14.2100693	best: 14.2099925 (1155)	total: 40.8s	remaining: 1m 40s
    1157:	learn: 10.7619835	test: 14.2110303	best: 14.2099925 (1155)	total: 40.8s	remaining: 1m 40s
    1158:	learn: 10.7595808	test: 14.2104368	best: 14.2099925 (1155)	total: 40.8s	remaining: 1m 40s
    1159:	learn: 10.7581326	test: 14.2093310	best: 14.2093310 (1159)	total: 40.8s	remaining: 1m 39s
    1160:	learn: 10.7572479	test: 14.2096278	best: 14.2093310 (1159)	total: 40.9s	remaining: 1m 39s
    1161:	learn: 10.7563584	test: 14.2090571	best: 14.2090571 (1161)	total: 40.9s	remaining: 1m 39s
    1162:	learn: 10.7548986	test: 14.2078424	best: 14.2078424 (1162)	total: 40.9s	remaining: 1m 39s
    1163:	learn: 10.7487333	test: 14.2039863	best: 14.2039863 (1163)	total: 41s	remaining: 1m 39s
    1164:	learn: 10.7475852	test: 14.2031601	best: 14.2031601 (1164)	total: 41s	remaining: 1m 39s
    1165:	learn: 10.7453837	test: 14.2022736	best: 14.2022736 (1165)	total: 41s	remaining: 1m 39s
    1166:	learn: 10.7443959	test: 14.2019636	best: 14.2019636 (1166)	total: 41.1s	remaining: 1m 39s
    1167:	learn: 10.7434383	test: 14.2013233	best: 14.2013233 (1167)	total: 41.1s	remaining: 1m 39s
    1168:	learn: 10.7415690	test: 14.2005629	best: 14.2005629 (1168)	total: 41.2s	remaining: 1m 39s
    1169:	learn: 10.7395986	test: 14.2008930	best: 14.2005629 (1168)	total: 41.2s	remaining: 1m 39s
    1170:	learn: 10.7379668	test: 14.1985412	best: 14.1985412 (1170)	total: 41.3s	remaining: 1m 39s
    1171:	learn: 10.7362788	test: 14.1985132	best: 14.1985132 (1171)	total: 41.3s	remaining: 1m 39s
    1172:	learn: 10.7318847	test: 14.1962982	best: 14.1962982 (1172)	total: 41.4s	remaining: 1m 39s
    1173:	learn: 10.7292746	test: 14.1951470	best: 14.1951470 (1173)	total: 41.4s	remaining: 1m 39s
    1174:	learn: 10.7248751	test: 14.1920058	best: 14.1920058 (1174)	total: 41.5s	remaining: 1m 39s
    1175:	learn: 10.7241625	test: 14.1920675	best: 14.1920058 (1174)	total: 41.5s	remaining: 1m 39s
    1176:	learn: 10.7223323	test: 14.1906533	best: 14.1906533 (1176)	total: 41.6s	remaining: 1m 39s
    1177:	learn: 10.7208807	test: 14.1899664	best: 14.1899664 (1177)	total: 41.6s	remaining: 1m 39s
    1178:	learn: 10.7196316	test: 14.1907359	best: 14.1899664 (1177)	total: 41.6s	remaining: 1m 39s
    1179:	learn: 10.7181603	test: 14.1904275	best: 14.1899664 (1177)	total: 41.7s	remaining: 1m 39s
    1180:	learn: 10.7163777	test: 14.1902800	best: 14.1899664 (1177)	total: 41.7s	remaining: 1m 39s
    1181:	learn: 10.7153131	test: 14.1891809	best: 14.1891809 (1181)	total: 41.7s	remaining: 1m 39s
    1182:	learn: 10.7132082	test: 14.1896155	best: 14.1891809 (1181)	total: 41.8s	remaining: 1m 39s
    1183:	learn: 10.7102071	test: 14.1901930	best: 14.1891809 (1181)	total: 41.8s	remaining: 1m 39s
    1184:	learn: 10.7087103	test: 14.1900533	best: 14.1891809 (1181)	total: 41.8s	remaining: 1m 39s
    1185:	learn: 10.7074896	test: 14.1895281	best: 14.1891809 (1181)	total: 41.9s	remaining: 1m 39s
    1186:	learn: 10.7051279	test: 14.1873628	best: 14.1873628 (1186)	total: 41.9s	remaining: 1m 39s
    1187:	learn: 10.7036932	test: 14.1854967	best: 14.1854967 (1187)	total: 42s	remaining: 1m 39s
    1188:	learn: 10.7017724	test: 14.1855941	best: 14.1854967 (1187)	total: 42s	remaining: 1m 39s
    1189:	learn: 10.6974066	test: 14.1821808	best: 14.1821808 (1189)	total: 42s	remaining: 1m 39s
    1190:	learn: 10.6927923	test: 14.1781361	best: 14.1781361 (1190)	total: 42.1s	remaining: 1m 39s
    1191:	learn: 10.6910835	test: 14.1763256	best: 14.1763256 (1191)	total: 42.1s	remaining: 1m 39s
    1192:	learn: 10.6892205	test: 14.1738435	best: 14.1738435 (1192)	total: 42.1s	remaining: 1m 39s
    1193:	learn: 10.6891139	test: 14.1734493	best: 14.1734493 (1193)	total: 42.1s	remaining: 1m 39s
    1194:	learn: 10.6863053	test: 14.1713433	best: 14.1713433 (1194)	total: 42.2s	remaining: 1m 39s
    1195:	learn: 10.6830879	test: 14.1704959	best: 14.1704959 (1195)	total: 42.2s	remaining: 1m 38s
    1196:	learn: 10.6808246	test: 14.1692055	best: 14.1692055 (1196)	total: 42.2s	remaining: 1m 38s
    1197:	learn: 10.6792968	test: 14.1697768	best: 14.1692055 (1196)	total: 42.3s	remaining: 1m 38s
    1198:	learn: 10.6773705	test: 14.1690936	best: 14.1690936 (1198)	total: 42.3s	remaining: 1m 38s
    1199:	learn: 10.6749084	test: 14.1682429	best: 14.1682429 (1199)	total: 42.3s	remaining: 1m 38s
    1200:	learn: 10.6720260	test: 14.1676745	best: 14.1676745 (1200)	total: 42.4s	remaining: 1m 38s
    1201:	learn: 10.6655271	test: 14.1657264	best: 14.1657264 (1201)	total: 42.4s	remaining: 1m 38s
    1202:	learn: 10.6624771	test: 14.1666825	best: 14.1657264 (1201)	total: 42.5s	remaining: 1m 38s
    1203:	learn: 10.6595689	test: 14.1660941	best: 14.1657264 (1201)	total: 42.5s	remaining: 1m 38s
    1204:	learn: 10.6587817	test: 14.1647251	best: 14.1647251 (1204)	total: 42.6s	remaining: 1m 38s
    1205:	learn: 10.6565793	test: 14.1639170	best: 14.1639170 (1205)	total: 42.6s	remaining: 1m 38s
    1206:	learn: 10.6555637	test: 14.1642014	best: 14.1639170 (1205)	total: 42.7s	remaining: 1m 38s
    1207:	learn: 10.6547072	test: 14.1638828	best: 14.1638828 (1207)	total: 42.7s	remaining: 1m 38s
    1208:	learn: 10.6537770	test: 14.1642636	best: 14.1638828 (1207)	total: 42.7s	remaining: 1m 38s
    1209:	learn: 10.6529937	test: 14.1640989	best: 14.1638828 (1207)	total: 42.8s	remaining: 1m 38s
    1210:	learn: 10.6514153	test: 14.1643148	best: 14.1638828 (1207)	total: 42.8s	remaining: 1m 38s
    1211:	learn: 10.6503900	test: 14.1646775	best: 14.1638828 (1207)	total: 42.8s	remaining: 1m 38s
    1212:	learn: 10.6455666	test: 14.1627391	best: 14.1627391 (1212)	total: 42.9s	remaining: 1m 38s
    1213:	learn: 10.6438177	test: 14.1639762	best: 14.1627391 (1212)	total: 42.9s	remaining: 1m 38s
    1214:	learn: 10.6425814	test: 14.1645747	best: 14.1627391 (1212)	total: 43s	remaining: 1m 38s
    1215:	learn: 10.6413848	test: 14.1648174	best: 14.1627391 (1212)	total: 43s	remaining: 1m 38s
    1216:	learn: 10.6384445	test: 14.1640316	best: 14.1627391 (1212)	total: 43s	remaining: 1m 38s
    1217:	learn: 10.6366549	test: 14.1644584	best: 14.1627391 (1212)	total: 43s	remaining: 1m 38s
    1218:	learn: 10.6348632	test: 14.1637871	best: 14.1627391 (1212)	total: 43.1s	remaining: 1m 38s
    1219:	learn: 10.6333096	test: 14.1628319	best: 14.1627391 (1212)	total: 43.1s	remaining: 1m 38s
    1220:	learn: 10.6302497	test: 14.1637619	best: 14.1627391 (1212)	total: 43.1s	remaining: 1m 38s
    1221:	learn: 10.6297286	test: 14.1640868	best: 14.1627391 (1212)	total: 43.2s	remaining: 1m 38s
    1222:	learn: 10.6283170	test: 14.1629854	best: 14.1627391 (1212)	total: 43.2s	remaining: 1m 38s
    1223:	learn: 10.6246152	test: 14.1615420	best: 14.1615420 (1223)	total: 43.2s	remaining: 1m 38s
    1224:	learn: 10.6233565	test: 14.1618516	best: 14.1615420 (1223)	total: 43.3s	remaining: 1m 38s
    1225:	learn: 10.6222748	test: 14.1613593	best: 14.1613593 (1225)	total: 43.3s	remaining: 1m 38s
    1226:	learn: 10.6195687	test: 14.1603567	best: 14.1603567 (1226)	total: 43.4s	remaining: 1m 37s
    1227:	learn: 10.6179287	test: 14.1604214	best: 14.1603567 (1226)	total: 43.4s	remaining: 1m 37s
    1228:	learn: 10.6174254	test: 14.1600833	best: 14.1600833 (1228)	total: 43.4s	remaining: 1m 37s
    1229:	learn: 10.6157193	test: 14.1601404	best: 14.1600833 (1228)	total: 43.5s	remaining: 1m 37s
    1230:	learn: 10.6124701	test: 14.1589791	best: 14.1589791 (1230)	total: 43.5s	remaining: 1m 37s
    1231:	learn: 10.6104138	test: 14.1599021	best: 14.1589791 (1230)	total: 43.5s	remaining: 1m 37s
    1232:	learn: 10.6088878	test: 14.1570896	best: 14.1570896 (1232)	total: 43.6s	remaining: 1m 37s
    1233:	learn: 10.6079162	test: 14.1568418	best: 14.1568418 (1233)	total: 43.6s	remaining: 1m 37s
    1234:	learn: 10.6061406	test: 14.1558940	best: 14.1558940 (1234)	total: 43.6s	remaining: 1m 37s
    1235:	learn: 10.6021608	test: 14.1545243	best: 14.1545243 (1235)	total: 43.7s	remaining: 1m 37s
    1236:	learn: 10.5999404	test: 14.1537693	best: 14.1537693 (1236)	total: 43.7s	remaining: 1m 37s
    1237:	learn: 10.5975411	test: 14.1528037	best: 14.1528037 (1237)	total: 43.7s	remaining: 1m 37s
    1238:	learn: 10.5951802	test: 14.1530264	best: 14.1528037 (1237)	total: 43.8s	remaining: 1m 37s
    1239:	learn: 10.5943671	test: 14.1531850	best: 14.1528037 (1237)	total: 43.8s	remaining: 1m 37s
    1240:	learn: 10.5923762	test: 14.1530239	best: 14.1528037 (1237)	total: 43.8s	remaining: 1m 37s
    1241:	learn: 10.5899541	test: 14.1506051	best: 14.1506051 (1241)	total: 43.9s	remaining: 1m 37s
    1242:	learn: 10.5882743	test: 14.1505529	best: 14.1505529 (1242)	total: 43.9s	remaining: 1m 37s
    1243:	learn: 10.5879162	test: 14.1504377	best: 14.1504377 (1243)	total: 43.9s	remaining: 1m 37s
    1244:	learn: 10.5867307	test: 14.1506478	best: 14.1504377 (1243)	total: 44s	remaining: 1m 37s
    1245:	learn: 10.5858951	test: 14.1507180	best: 14.1504377 (1243)	total: 44s	remaining: 1m 37s
    1246:	learn: 10.5849198	test: 14.1506990	best: 14.1504377 (1243)	total: 44s	remaining: 1m 37s
    1247:	learn: 10.5842408	test: 14.1494691	best: 14.1494691 (1247)	total: 44.1s	remaining: 1m 37s
    1248:	learn: 10.5830673	test: 14.1500655	best: 14.1494691 (1247)	total: 44.1s	remaining: 1m 37s
    1249:	learn: 10.5823009	test: 14.1499271	best: 14.1494691 (1247)	total: 44.2s	remaining: 1m 37s
    1250:	learn: 10.5802289	test: 14.1511199	best: 14.1494691 (1247)	total: 44.2s	remaining: 1m 37s
    1251:	learn: 10.5796513	test: 14.1515772	best: 14.1494691 (1247)	total: 44.2s	remaining: 1m 37s
    1252:	learn: 10.5786364	test: 14.1517659	best: 14.1494691 (1247)	total: 44.3s	remaining: 1m 37s
    1253:	learn: 10.5781662	test: 14.1510460	best: 14.1494691 (1247)	total: 44.3s	remaining: 1m 36s
    1254:	learn: 10.5776075	test: 14.1508352	best: 14.1494691 (1247)	total: 44.3s	remaining: 1m 36s
    1255:	learn: 10.5768916	test: 14.1511995	best: 14.1494691 (1247)	total: 44.3s	remaining: 1m 36s
    1256:	learn: 10.5755966	test: 14.1501391	best: 14.1494691 (1247)	total: 44.4s	remaining: 1m 36s
    1257:	learn: 10.5736340	test: 14.1493349	best: 14.1493349 (1257)	total: 44.4s	remaining: 1m 36s
    1258:	learn: 10.5712548	test: 14.1487830	best: 14.1487830 (1258)	total: 44.4s	remaining: 1m 36s
    1259:	learn: 10.5687428	test: 14.1478772	best: 14.1478772 (1259)	total: 44.5s	remaining: 1m 36s
    1260:	learn: 10.5673180	test: 14.1474130	best: 14.1474130 (1260)	total: 44.5s	remaining: 1m 36s
    1261:	learn: 10.5662146	test: 14.1469845	best: 14.1469845 (1261)	total: 44.5s	remaining: 1m 36s
    1262:	learn: 10.5649501	test: 14.1462197	best: 14.1462197 (1262)	total: 44.6s	remaining: 1m 36s
    1263:	learn: 10.5637060	test: 14.1457067	best: 14.1457067 (1263)	total: 44.6s	remaining: 1m 36s
    1264:	learn: 10.5632053	test: 14.1450774	best: 14.1450774 (1264)	total: 44.7s	remaining: 1m 36s
    1265:	learn: 10.5603999	test: 14.1435936	best: 14.1435936 (1265)	total: 44.7s	remaining: 1m 36s
    1266:	learn: 10.5579755	test: 14.1446951	best: 14.1435936 (1265)	total: 44.7s	remaining: 1m 36s
    1267:	learn: 10.5575613	test: 14.1449430	best: 14.1435936 (1265)	total: 44.8s	remaining: 1m 36s
    1268:	learn: 10.5556109	test: 14.1454439	best: 14.1435936 (1265)	total: 44.8s	remaining: 1m 36s
    1269:	learn: 10.5548188	test: 14.1459004	best: 14.1435936 (1265)	total: 44.8s	remaining: 1m 36s
    1270:	learn: 10.5543464	test: 14.1456090	best: 14.1435936 (1265)	total: 44.9s	remaining: 1m 36s
    1271:	learn: 10.5510600	test: 14.1455159	best: 14.1435936 (1265)	total: 44.9s	remaining: 1m 36s
    1272:	learn: 10.5477661	test: 14.1440166	best: 14.1435936 (1265)	total: 44.9s	remaining: 1m 36s
    1273:	learn: 10.5445338	test: 14.1413175	best: 14.1413175 (1273)	total: 45s	remaining: 1m 36s
    1274:	learn: 10.5434059	test: 14.1399916	best: 14.1399916 (1274)	total: 45s	remaining: 1m 36s
    1275:	learn: 10.5402819	test: 14.1356826	best: 14.1356826 (1275)	total: 45s	remaining: 1m 36s
    1276:	learn: 10.5360445	test: 14.1357694	best: 14.1356826 (1275)	total: 45.1s	remaining: 1m 36s
    1277:	learn: 10.5355817	test: 14.1364396	best: 14.1356826 (1275)	total: 45.1s	remaining: 1m 36s
    1278:	learn: 10.5349322	test: 14.1356656	best: 14.1356656 (1278)	total: 45.2s	remaining: 1m 36s
    1279:	learn: 10.5332890	test: 14.1361041	best: 14.1356656 (1278)	total: 45.2s	remaining: 1m 36s
    1280:	learn: 10.5329687	test: 14.1362781	best: 14.1356656 (1278)	total: 45.3s	remaining: 1m 36s
    1281:	learn: 10.5316994	test: 14.1359305	best: 14.1356656 (1278)	total: 45.3s	remaining: 1m 36s
    1282:	learn: 10.5311114	test: 14.1355778	best: 14.1355778 (1282)	total: 45.3s	remaining: 1m 36s
    1283:	learn: 10.5293974	test: 14.1332419	best: 14.1332419 (1283)	total: 45.4s	remaining: 1m 35s
    1284:	learn: 10.5275796	test: 14.1327746	best: 14.1327746 (1284)	total: 45.4s	remaining: 1m 35s
    1285:	learn: 10.5266587	test: 14.1325763	best: 14.1325763 (1285)	total: 45.4s	remaining: 1m 35s
    1286:	learn: 10.5265335	test: 14.1314438	best: 14.1314438 (1286)	total: 45.5s	remaining: 1m 35s
    1287:	learn: 10.5255887	test: 14.1318067	best: 14.1314438 (1286)	total: 45.5s	remaining: 1m 35s
    1288:	learn: 10.5242024	test: 14.1308154	best: 14.1308154 (1288)	total: 45.6s	remaining: 1m 35s
    1289:	learn: 10.5231380	test: 14.1301481	best: 14.1301481 (1289)	total: 45.6s	remaining: 1m 35s
    1290:	learn: 10.5220726	test: 14.1301097	best: 14.1301097 (1290)	total: 45.7s	remaining: 1m 35s
    1291:	learn: 10.5205349	test: 14.1292393	best: 14.1292393 (1291)	total: 45.7s	remaining: 1m 35s
    1292:	learn: 10.5199329	test: 14.1296618	best: 14.1292393 (1291)	total: 45.8s	remaining: 1m 35s
    1293:	learn: 10.5180568	test: 14.1287407	best: 14.1287407 (1293)	total: 45.8s	remaining: 1m 35s
    1294:	learn: 10.5161539	test: 14.1272288	best: 14.1272288 (1294)	total: 45.8s	remaining: 1m 35s
    1295:	learn: 10.5140888	test: 14.1252717	best: 14.1252717 (1295)	total: 45.9s	remaining: 1m 35s
    1296:	learn: 10.5132056	test: 14.1255193	best: 14.1252717 (1295)	total: 45.9s	remaining: 1m 35s
    1297:	learn: 10.5093090	test: 14.1209009	best: 14.1209009 (1297)	total: 45.9s	remaining: 1m 35s
    1298:	learn: 10.5067317	test: 14.1197901	best: 14.1197901 (1298)	total: 45.9s	remaining: 1m 35s
    1299:	learn: 10.5060529	test: 14.1187885	best: 14.1187885 (1299)	total: 46s	remaining: 1m 35s
    1300:	learn: 10.5048619	test: 14.1181081	best: 14.1181081 (1300)	total: 46s	remaining: 1m 35s
    1301:	learn: 10.5044483	test: 14.1178068	best: 14.1178068 (1301)	total: 46s	remaining: 1m 35s
    1302:	learn: 10.5033242	test: 14.1186135	best: 14.1178068 (1301)	total: 46.1s	remaining: 1m 35s
    1303:	learn: 10.5019651	test: 14.1186056	best: 14.1178068 (1301)	total: 46.1s	remaining: 1m 35s
    1304:	learn: 10.5008155	test: 14.1192527	best: 14.1178068 (1301)	total: 46.1s	remaining: 1m 35s
    1305:	learn: 10.4966377	test: 14.1163923	best: 14.1163923 (1305)	total: 46.1s	remaining: 1m 35s
    1306:	learn: 10.4955111	test: 14.1162399	best: 14.1162399 (1306)	total: 46.2s	remaining: 1m 35s
    1307:	learn: 10.4918791	test: 14.1129790	best: 14.1129790 (1307)	total: 46.2s	remaining: 1m 35s
    1308:	learn: 10.4907191	test: 14.1117370	best: 14.1117370 (1308)	total: 46.2s	remaining: 1m 35s
    1309:	learn: 10.4890307	test: 14.1107452	best: 14.1107452 (1309)	total: 46.3s	remaining: 1m 34s
    1310:	learn: 10.4881894	test: 14.1110516	best: 14.1107452 (1309)	total: 46.3s	remaining: 1m 34s
    1311:	learn: 10.4844430	test: 14.1083049	best: 14.1083049 (1311)	total: 46.3s	remaining: 1m 34s
    1312:	learn: 10.4836298	test: 14.1080746	best: 14.1080746 (1312)	total: 46.3s	remaining: 1m 34s
    1313:	learn: 10.4824916	test: 14.1078980	best: 14.1078980 (1313)	total: 46.4s	remaining: 1m 34s
    1314:	learn: 10.4783481	test: 14.1062259	best: 14.1062259 (1314)	total: 46.4s	remaining: 1m 34s
    1315:	learn: 10.4777979	test: 14.1067367	best: 14.1062259 (1314)	total: 46.4s	remaining: 1m 34s
    1316:	learn: 10.4766914	test: 14.1062805	best: 14.1062259 (1314)	total: 46.5s	remaining: 1m 34s
    1317:	learn: 10.4741179	test: 14.1056026	best: 14.1056026 (1317)	total: 46.5s	remaining: 1m 34s
    1318:	learn: 10.4738180	test: 14.1066764	best: 14.1056026 (1317)	total: 46.5s	remaining: 1m 34s
    1319:	learn: 10.4724197	test: 14.1050821	best: 14.1050821 (1319)	total: 46.5s	remaining: 1m 34s
    1320:	learn: 10.4708093	test: 14.1055837	best: 14.1050821 (1319)	total: 46.6s	remaining: 1m 34s
    1321:	learn: 10.4696450	test: 14.1054920	best: 14.1050821 (1319)	total: 46.6s	remaining: 1m 34s
    1322:	learn: 10.4677108	test: 14.1045101	best: 14.1045101 (1322)	total: 46.6s	remaining: 1m 34s
    1323:	learn: 10.4660451	test: 14.1049186	best: 14.1045101 (1322)	total: 46.7s	remaining: 1m 34s
    1324:	learn: 10.4621798	test: 14.1008565	best: 14.1008565 (1324)	total: 46.7s	remaining: 1m 34s
    1325:	learn: 10.4603326	test: 14.1008212	best: 14.1008212 (1325)	total: 46.7s	remaining: 1m 34s
    1326:	learn: 10.4601954	test: 14.1000007	best: 14.1000007 (1326)	total: 46.8s	remaining: 1m 34s
    1327:	learn: 10.4588257	test: 14.1001518	best: 14.1000007 (1326)	total: 46.8s	remaining: 1m 34s
    1328:	learn: 10.4572444	test: 14.0981338	best: 14.0981338 (1328)	total: 46.9s	remaining: 1m 34s
    1329:	learn: 10.4561007	test: 14.0989541	best: 14.0981338 (1328)	total: 46.9s	remaining: 1m 34s
    1330:	learn: 10.4552003	test: 14.0983756	best: 14.0981338 (1328)	total: 46.9s	remaining: 1m 34s
    1331:	learn: 10.4515160	test: 14.0994586	best: 14.0981338 (1328)	total: 47s	remaining: 1m 34s
    1332:	learn: 10.4505811	test: 14.0980491	best: 14.0980491 (1332)	total: 47s	remaining: 1m 34s
    1333:	learn: 10.4466020	test: 14.0956341	best: 14.0956341 (1333)	total: 47s	remaining: 1m 34s
    1334:	learn: 10.4455316	test: 14.0960256	best: 14.0956341 (1333)	total: 47.1s	remaining: 1m 33s
    1335:	learn: 10.4417867	test: 14.0932637	best: 14.0932637 (1335)	total: 47.1s	remaining: 1m 33s
    1336:	learn: 10.4393855	test: 14.0923409	best: 14.0923409 (1336)	total: 47.1s	remaining: 1m 33s
    1337:	learn: 10.4368238	test: 14.0917725	best: 14.0917725 (1337)	total: 47.2s	remaining: 1m 33s
    1338:	learn: 10.4361469	test: 14.0917166	best: 14.0917166 (1338)	total: 47.2s	remaining: 1m 33s
    1339:	learn: 10.4345793	test: 14.0919933	best: 14.0917166 (1338)	total: 47.2s	remaining: 1m 33s
    1340:	learn: 10.4333998	test: 14.0924511	best: 14.0917166 (1338)	total: 47.2s	remaining: 1m 33s
    1341:	learn: 10.4321830	test: 14.0914803	best: 14.0914803 (1341)	total: 47.3s	remaining: 1m 33s
    1342:	learn: 10.4297971	test: 14.0902049	best: 14.0902049 (1342)	total: 47.3s	remaining: 1m 33s
    1343:	learn: 10.4283560	test: 14.0913056	best: 14.0902049 (1342)	total: 47.3s	remaining: 1m 33s
    1344:	learn: 10.4273041	test: 14.0907055	best: 14.0902049 (1342)	total: 47.4s	remaining: 1m 33s
    1345:	learn: 10.4235522	test: 14.0912453	best: 14.0902049 (1342)	total: 47.4s	remaining: 1m 33s
    1346:	learn: 10.4222822	test: 14.0912122	best: 14.0902049 (1342)	total: 47.4s	remaining: 1m 33s
    1347:	learn: 10.4178682	test: 14.0881396	best: 14.0881396 (1347)	total: 47.5s	remaining: 1m 33s
    1348:	learn: 10.4116174	test: 14.0817341	best: 14.0817341 (1348)	total: 47.5s	remaining: 1m 33s
    1349:	learn: 10.4106248	test: 14.0823248	best: 14.0817341 (1348)	total: 47.6s	remaining: 1m 33s
    1350:	learn: 10.4087451	test: 14.0809031	best: 14.0809031 (1350)	total: 47.6s	remaining: 1m 33s
    1351:	learn: 10.4078737	test: 14.0814162	best: 14.0809031 (1350)	total: 47.6s	remaining: 1m 33s
    1352:	learn: 10.4072161	test: 14.0804296	best: 14.0804296 (1352)	total: 47.7s	remaining: 1m 33s
    1353:	learn: 10.4059331	test: 14.0798904	best: 14.0798904 (1353)	total: 47.7s	remaining: 1m 33s
    1354:	learn: 10.4019152	test: 14.0780684	best: 14.0780684 (1354)	total: 47.7s	remaining: 1m 33s
    1355:	learn: 10.4010487	test: 14.0780706	best: 14.0780684 (1354)	total: 47.8s	remaining: 1m 33s
    1356:	learn: 10.3982629	test: 14.0761559	best: 14.0761559 (1356)	total: 47.8s	remaining: 1m 33s
    1357:	learn: 10.3959976	test: 14.0766533	best: 14.0761559 (1356)	total: 47.9s	remaining: 1m 33s
    1358:	learn: 10.3927330	test: 14.0740301	best: 14.0740301 (1358)	total: 47.9s	remaining: 1m 33s
    1359:	learn: 10.3916669	test: 14.0736769	best: 14.0736769 (1359)	total: 48s	remaining: 1m 33s
    1360:	learn: 10.3908756	test: 14.0730719	best: 14.0730719 (1360)	total: 48s	remaining: 1m 33s
    1361:	learn: 10.3881206	test: 14.0733025	best: 14.0730719 (1360)	total: 48.1s	remaining: 1m 33s
    1362:	learn: 10.3872017	test: 14.0726029	best: 14.0726029 (1362)	total: 48.1s	remaining: 1m 33s
    1363:	learn: 10.3859594	test: 14.0726879	best: 14.0726029 (1362)	total: 48.1s	remaining: 1m 33s
    1364:	learn: 10.3835924	test: 14.0728355	best: 14.0726029 (1362)	total: 48.2s	remaining: 1m 33s
    1365:	learn: 10.3796738	test: 14.0697396	best: 14.0697396 (1365)	total: 48.2s	remaining: 1m 32s
    1366:	learn: 10.3774243	test: 14.0700208	best: 14.0697396 (1365)	total: 48.3s	remaining: 1m 32s
    1367:	learn: 10.3761952	test: 14.0709399	best: 14.0697396 (1365)	total: 48.3s	remaining: 1m 32s
    1368:	learn: 10.3734885	test: 14.0703212	best: 14.0697396 (1365)	total: 48.3s	remaining: 1m 32s
    1369:	learn: 10.3722340	test: 14.0696143	best: 14.0696143 (1369)	total: 48.4s	remaining: 1m 32s
    1370:	learn: 10.3714281	test: 14.0702002	best: 14.0696143 (1369)	total: 48.4s	remaining: 1m 32s
    1371:	learn: 10.3682807	test: 14.0685326	best: 14.0685326 (1371)	total: 48.4s	remaining: 1m 32s
    1372:	learn: 10.3677422	test: 14.0683057	best: 14.0683057 (1372)	total: 48.5s	remaining: 1m 32s
    1373:	learn: 10.3665313	test: 14.0683718	best: 14.0683057 (1372)	total: 48.5s	remaining: 1m 32s
    1374:	learn: 10.3648163	test: 14.0690308	best: 14.0683057 (1372)	total: 48.5s	remaining: 1m 32s
    1375:	learn: 10.3637926	test: 14.0690897	best: 14.0683057 (1372)	total: 48.5s	remaining: 1m 32s
    1376:	learn: 10.3631197	test: 14.0690939	best: 14.0683057 (1372)	total: 48.6s	remaining: 1m 32s
    1377:	learn: 10.3619517	test: 14.0673304	best: 14.0673304 (1377)	total: 48.6s	remaining: 1m 32s
    1378:	learn: 10.3609678	test: 14.0678257	best: 14.0673304 (1377)	total: 48.6s	remaining: 1m 32s
    1379:	learn: 10.3603111	test: 14.0673427	best: 14.0673304 (1377)	total: 48.7s	remaining: 1m 32s
    1380:	learn: 10.3597124	test: 14.0676741	best: 14.0673304 (1377)	total: 48.7s	remaining: 1m 32s
    1381:	learn: 10.3564222	test: 14.0649422	best: 14.0649422 (1381)	total: 48.7s	remaining: 1m 32s
    1382:	learn: 10.3555297	test: 14.0643102	best: 14.0643102 (1382)	total: 48.8s	remaining: 1m 32s
    1383:	learn: 10.3539876	test: 14.0631698	best: 14.0631698 (1383)	total: 48.8s	remaining: 1m 32s
    1384:	learn: 10.3532642	test: 14.0636796	best: 14.0631698 (1383)	total: 48.8s	remaining: 1m 32s
    1385:	learn: 10.3521256	test: 14.0628154	best: 14.0628154 (1385)	total: 48.8s	remaining: 1m 32s
    1386:	learn: 10.3504469	test: 14.0607475	best: 14.0607475 (1386)	total: 48.9s	remaining: 1m 32s
    1387:	learn: 10.3495731	test: 14.0612639	best: 14.0607475 (1386)	total: 48.9s	remaining: 1m 32s
    1388:	learn: 10.3478222	test: 14.0603389	best: 14.0603389 (1388)	total: 48.9s	remaining: 1m 31s
    1389:	learn: 10.3444980	test: 14.0577593	best: 14.0577593 (1389)	total: 49s	remaining: 1m 31s
    1390:	learn: 10.3418127	test: 14.0575538	best: 14.0575538 (1390)	total: 49s	remaining: 1m 31s
    1391:	learn: 10.3406917	test: 14.0573862	best: 14.0573862 (1391)	total: 49s	remaining: 1m 31s
    1392:	learn: 10.3399014	test: 14.0573603	best: 14.0573603 (1392)	total: 49s	remaining: 1m 31s
    1393:	learn: 10.3373869	test: 14.0550312	best: 14.0550312 (1393)	total: 49.1s	remaining: 1m 31s
    1394:	learn: 10.3346778	test: 14.0529683	best: 14.0529683 (1394)	total: 49.1s	remaining: 1m 31s
    1395:	learn: 10.3323502	test: 14.0522210	best: 14.0522210 (1395)	total: 49.1s	remaining: 1m 31s
    1396:	learn: 10.3302015	test: 14.0507850	best: 14.0507850 (1396)	total: 49.2s	remaining: 1m 31s
    1397:	learn: 10.3293585	test: 14.0504370	best: 14.0504370 (1397)	total: 49.2s	remaining: 1m 31s
    1398:	learn: 10.3270309	test: 14.0501660	best: 14.0501660 (1398)	total: 49.2s	remaining: 1m 31s
    1399:	learn: 10.3243371	test: 14.0493826	best: 14.0493826 (1399)	total: 49.2s	remaining: 1m 31s
    1400:	learn: 10.3224502	test: 14.0477547	best: 14.0477547 (1400)	total: 49.3s	remaining: 1m 31s
    1401:	learn: 10.3214450	test: 14.0469579	best: 14.0469579 (1401)	total: 49.3s	remaining: 1m 31s
    1402:	learn: 10.3211204	test: 14.0465733	best: 14.0465733 (1402)	total: 49.3s	remaining: 1m 31s
    1403:	learn: 10.3190160	test: 14.0468360	best: 14.0465733 (1402)	total: 49.4s	remaining: 1m 31s
    1404:	learn: 10.3187558	test: 14.0467037	best: 14.0465733 (1402)	total: 49.4s	remaining: 1m 31s
    1405:	learn: 10.3170105	test: 14.0466150	best: 14.0465733 (1402)	total: 49.4s	remaining: 1m 31s
    1406:	learn: 10.3164560	test: 14.0466806	best: 14.0465733 (1402)	total: 49.5s	remaining: 1m 31s
    1407:	learn: 10.3148974	test: 14.0462602	best: 14.0462602 (1407)	total: 49.5s	remaining: 1m 31s
    1408:	learn: 10.3145062	test: 14.0457295	best: 14.0457295 (1408)	total: 49.5s	remaining: 1m 31s
    1409:	learn: 10.3131886	test: 14.0448697	best: 14.0448697 (1409)	total: 49.6s	remaining: 1m 31s
    1410:	learn: 10.3110190	test: 14.0428870	best: 14.0428870 (1410)	total: 49.6s	remaining: 1m 31s
    1411:	learn: 10.3082810	test: 14.0418069	best: 14.0418069 (1411)	total: 49.6s	remaining: 1m 30s
    1412:	learn: 10.3078262	test: 14.0416077	best: 14.0416077 (1412)	total: 49.7s	remaining: 1m 30s
    1413:	learn: 10.3047804	test: 14.0404414	best: 14.0404414 (1413)	total: 49.7s	remaining: 1m 30s
    1414:	learn: 10.3038700	test: 14.0403965	best: 14.0403965 (1414)	total: 49.7s	remaining: 1m 30s
    1415:	learn: 10.3031238	test: 14.0402469	best: 14.0402469 (1415)	total: 49.7s	remaining: 1m 30s
    1416:	learn: 10.3016012	test: 14.0407280	best: 14.0402469 (1415)	total: 49.8s	remaining: 1m 30s
    1417:	learn: 10.2991772	test: 14.0420116	best: 14.0402469 (1415)	total: 49.8s	remaining: 1m 30s
    1418:	learn: 10.2977096	test: 14.0410949	best: 14.0402469 (1415)	total: 49.8s	remaining: 1m 30s
    1419:	learn: 10.2961940	test: 14.0401810	best: 14.0401810 (1419)	total: 49.9s	remaining: 1m 30s
    1420:	learn: 10.2949902	test: 14.0407757	best: 14.0401810 (1419)	total: 49.9s	remaining: 1m 30s
    1421:	learn: 10.2943515	test: 14.0403075	best: 14.0401810 (1419)	total: 49.9s	remaining: 1m 30s
    1422:	learn: 10.2917647	test: 14.0398835	best: 14.0398835 (1422)	total: 50s	remaining: 1m 30s
    1423:	learn: 10.2906764	test: 14.0399557	best: 14.0398835 (1422)	total: 50s	remaining: 1m 30s
    1424:	learn: 10.2891549	test: 14.0398646	best: 14.0398646 (1424)	total: 50s	remaining: 1m 30s
    1425:	learn: 10.2881068	test: 14.0388993	best: 14.0388993 (1425)	total: 50s	remaining: 1m 30s
    1426:	learn: 10.2848996	test: 14.0369969	best: 14.0369969 (1426)	total: 50.1s	remaining: 1m 30s
    1427:	learn: 10.2842013	test: 14.0376428	best: 14.0369969 (1426)	total: 50.1s	remaining: 1m 30s
    1428:	learn: 10.2837132	test: 14.0379651	best: 14.0369969 (1426)	total: 50.1s	remaining: 1m 30s
    1429:	learn: 10.2822953	test: 14.0383680	best: 14.0369969 (1426)	total: 50.1s	remaining: 1m 30s
    1430:	learn: 10.2818138	test: 14.0379839	best: 14.0369969 (1426)	total: 50.2s	remaining: 1m 30s
    1431:	learn: 10.2803221	test: 14.0379809	best: 14.0369969 (1426)	total: 50.2s	remaining: 1m 30s
    1432:	learn: 10.2801120	test: 14.0377832	best: 14.0369969 (1426)	total: 50.2s	remaining: 1m 29s
    1433:	learn: 10.2789523	test: 14.0385413	best: 14.0369969 (1426)	total: 50.3s	remaining: 1m 29s
    1434:	learn: 10.2764244	test: 14.0401011	best: 14.0369969 (1426)	total: 50.3s	remaining: 1m 29s
    1435:	learn: 10.2749161	test: 14.0405690	best: 14.0369969 (1426)	total: 50.3s	remaining: 1m 29s
    1436:	learn: 10.2723820	test: 14.0382503	best: 14.0369969 (1426)	total: 50.3s	remaining: 1m 29s
    1437:	learn: 10.2709240	test: 14.0377587	best: 14.0369969 (1426)	total: 50.4s	remaining: 1m 29s
    1438:	learn: 10.2701794	test: 14.0379449	best: 14.0369969 (1426)	total: 50.4s	remaining: 1m 29s
    1439:	learn: 10.2685011	test: 14.0368175	best: 14.0368175 (1439)	total: 50.4s	remaining: 1m 29s
    1440:	learn: 10.2669761	test: 14.0373795	best: 14.0368175 (1439)	total: 50.5s	remaining: 1m 29s
    1441:	learn: 10.2659767	test: 14.0371434	best: 14.0368175 (1439)	total: 50.5s	remaining: 1m 29s
    1442:	learn: 10.2655569	test: 14.0378059	best: 14.0368175 (1439)	total: 50.5s	remaining: 1m 29s
    1443:	learn: 10.2631033	test: 14.0385268	best: 14.0368175 (1439)	total: 50.5s	remaining: 1m 29s
    1444:	learn: 10.2604300	test: 14.0371484	best: 14.0368175 (1439)	total: 50.6s	remaining: 1m 29s
    1445:	learn: 10.2580523	test: 14.0368714	best: 14.0368175 (1439)	total: 50.6s	remaining: 1m 29s
    1446:	learn: 10.2559517	test: 14.0344023	best: 14.0344023 (1446)	total: 50.6s	remaining: 1m 29s
    1447:	learn: 10.2525311	test: 14.0330987	best: 14.0330987 (1447)	total: 50.7s	remaining: 1m 29s
    1448:	learn: 10.2497948	test: 14.0308770	best: 14.0308770 (1448)	total: 50.7s	remaining: 1m 29s
    1449:	learn: 10.2457619	test: 14.0294522	best: 14.0294522 (1449)	total: 50.7s	remaining: 1m 29s
    1450:	learn: 10.2436788	test: 14.0278608	best: 14.0278608 (1450)	total: 50.7s	remaining: 1m 29s
    1451:	learn: 10.2424791	test: 14.0281724	best: 14.0278608 (1450)	total: 50.8s	remaining: 1m 29s
    1452:	learn: 10.2388090	test: 14.0268332	best: 14.0268332 (1452)	total: 50.8s	remaining: 1m 29s
    1453:	learn: 10.2380640	test: 14.0271317	best: 14.0268332 (1452)	total: 50.8s	remaining: 1m 29s
    1454:	learn: 10.2376161	test: 14.0270116	best: 14.0268332 (1452)	total: 50.9s	remaining: 1m 28s
    1455:	learn: 10.2367316	test: 14.0275013	best: 14.0268332 (1452)	total: 50.9s	remaining: 1m 28s
    1456:	learn: 10.2340238	test: 14.0249065	best: 14.0249065 (1456)	total: 50.9s	remaining: 1m 28s
    1457:	learn: 10.2334880	test: 14.0250840	best: 14.0249065 (1456)	total: 50.9s	remaining: 1m 28s
    1458:	learn: 10.2330392	test: 14.0254797	best: 14.0249065 (1456)	total: 51s	remaining: 1m 28s
    1459:	learn: 10.2319504	test: 14.0247712	best: 14.0247712 (1459)	total: 51s	remaining: 1m 28s
    1460:	learn: 10.2302810	test: 14.0243044	best: 14.0243044 (1460)	total: 51s	remaining: 1m 28s
    1461:	learn: 10.2253997	test: 14.0189034	best: 14.0189034 (1461)	total: 51s	remaining: 1m 28s
    1462:	learn: 10.2236477	test: 14.0183059	best: 14.0183059 (1462)	total: 51.1s	remaining: 1m 28s
    1463:	learn: 10.2203253	test: 14.0162039	best: 14.0162039 (1463)	total: 51.1s	remaining: 1m 28s
    1464:	learn: 10.2169535	test: 14.0146216	best: 14.0146216 (1464)	total: 51.1s	remaining: 1m 28s
    1465:	learn: 10.2156778	test: 14.0131964	best: 14.0131964 (1465)	total: 51.1s	remaining: 1m 28s
    1466:	learn: 10.2138713	test: 14.0118988	best: 14.0118988 (1466)	total: 51.2s	remaining: 1m 28s
    1467:	learn: 10.2130896	test: 14.0119381	best: 14.0118988 (1466)	total: 51.2s	remaining: 1m 28s
    1468:	learn: 10.2113241	test: 14.0117541	best: 14.0117541 (1468)	total: 51.3s	remaining: 1m 28s
    1469:	learn: 10.2099646	test: 14.0097919	best: 14.0097919 (1469)	total: 51.3s	remaining: 1m 28s
    1470:	learn: 10.2084628	test: 14.0101546	best: 14.0097919 (1469)	total: 51.3s	remaining: 1m 28s
    1471:	learn: 10.2077722	test: 14.0096303	best: 14.0096303 (1471)	total: 51.3s	remaining: 1m 28s
    1472:	learn: 10.2070135	test: 14.0100054	best: 14.0096303 (1471)	total: 51.4s	remaining: 1m 28s
    1473:	learn: 10.2066530	test: 14.0097247	best: 14.0096303 (1471)	total: 51.4s	remaining: 1m 28s
    1474:	learn: 10.2049450	test: 14.0090165	best: 14.0090165 (1474)	total: 51.4s	remaining: 1m 28s
    1475:	learn: 10.2037502	test: 14.0090035	best: 14.0090035 (1475)	total: 51.5s	remaining: 1m 28s
    1476:	learn: 10.2026515	test: 14.0092496	best: 14.0090035 (1475)	total: 51.5s	remaining: 1m 27s
    1477:	learn: 10.2021175	test: 14.0096904	best: 14.0090035 (1475)	total: 51.5s	remaining: 1m 27s
    1478:	learn: 10.2018802	test: 14.0089194	best: 14.0089194 (1478)	total: 51.6s	remaining: 1m 27s
    1479:	learn: 10.1991064	test: 14.0086042	best: 14.0086042 (1479)	total: 51.6s	remaining: 1m 27s
    1480:	learn: 10.1974592	test: 14.0106858	best: 14.0086042 (1479)	total: 51.6s	remaining: 1m 27s
    1481:	learn: 10.1965817	test: 14.0117945	best: 14.0086042 (1479)	total: 51.6s	remaining: 1m 27s
    1482:	learn: 10.1957465	test: 14.0114495	best: 14.0086042 (1479)	total: 51.7s	remaining: 1m 27s
    1483:	learn: 10.1952870	test: 14.0118954	best: 14.0086042 (1479)	total: 51.7s	remaining: 1m 27s
    1484:	learn: 10.1946691	test: 14.0112262	best: 14.0086042 (1479)	total: 51.7s	remaining: 1m 27s
    1485:	learn: 10.1935972	test: 14.0118756	best: 14.0086042 (1479)	total: 51.8s	remaining: 1m 27s
    1486:	learn: 10.1917394	test: 14.0092950	best: 14.0086042 (1479)	total: 51.8s	remaining: 1m 27s
    1487:	learn: 10.1904676	test: 14.0081139	best: 14.0081139 (1487)	total: 51.8s	remaining: 1m 27s
    1488:	learn: 10.1890683	test: 14.0079832	best: 14.0079832 (1488)	total: 51.8s	remaining: 1m 27s
    1489:	learn: 10.1871227	test: 14.0074651	best: 14.0074651 (1489)	total: 51.9s	remaining: 1m 27s
    1490:	learn: 10.1863764	test: 14.0072359	best: 14.0072359 (1490)	total: 51.9s	remaining: 1m 27s
    1491:	learn: 10.1856907	test: 14.0068424	best: 14.0068424 (1491)	total: 51.9s	remaining: 1m 27s
    1492:	learn: 10.1853305	test: 14.0054223	best: 14.0054223 (1492)	total: 51.9s	remaining: 1m 27s
    1493:	learn: 10.1812569	test: 14.0018323	best: 14.0018323 (1493)	total: 52s	remaining: 1m 27s
    1494:	learn: 10.1797634	test: 14.0014662	best: 14.0014662 (1494)	total: 52s	remaining: 1m 27s
    1495:	learn: 10.1774248	test: 14.0000824	best: 14.0000824 (1495)	total: 52s	remaining: 1m 27s
    1496:	learn: 10.1753149	test: 13.9985434	best: 13.9985434 (1496)	total: 52.1s	remaining: 1m 27s
    1497:	learn: 10.1751307	test: 13.9990625	best: 13.9985434 (1496)	total: 52.1s	remaining: 1m 26s
    1498:	learn: 10.1744310	test: 13.9982458	best: 13.9982458 (1498)	total: 52.1s	remaining: 1m 26s
    1499:	learn: 10.1733019	test: 13.9981127	best: 13.9981127 (1499)	total: 52.1s	remaining: 1m 26s
    1500:	learn: 10.1726884	test: 13.9978781	best: 13.9978781 (1500)	total: 52.2s	remaining: 1m 26s
    1501:	learn: 10.1711860	test: 13.9980847	best: 13.9978781 (1500)	total: 52.2s	remaining: 1m 26s
    1502:	learn: 10.1702516	test: 13.9971934	best: 13.9971934 (1502)	total: 52.2s	remaining: 1m 26s
    1503:	learn: 10.1692732	test: 13.9984802	best: 13.9971934 (1502)	total: 52.3s	remaining: 1m 26s
    1504:	learn: 10.1662774	test: 13.9984651	best: 13.9971934 (1502)	total: 52.3s	remaining: 1m 26s
    1505:	learn: 10.1654135	test: 13.9993740	best: 13.9971934 (1502)	total: 52.3s	remaining: 1m 26s
    1506:	learn: 10.1645559	test: 13.9986133	best: 13.9971934 (1502)	total: 52.4s	remaining: 1m 26s
    1507:	learn: 10.1611354	test: 13.9999682	best: 13.9971934 (1502)	total: 52.4s	remaining: 1m 26s
    1508:	learn: 10.1589778	test: 13.9990570	best: 13.9971934 (1502)	total: 52.4s	remaining: 1m 26s
    1509:	learn: 10.1573358	test: 13.9967492	best: 13.9967492 (1509)	total: 52.4s	remaining: 1m 26s
    1510:	learn: 10.1553436	test: 13.9948903	best: 13.9948903 (1510)	total: 52.5s	remaining: 1m 26s
    1511:	learn: 10.1522297	test: 13.9950843	best: 13.9948903 (1510)	total: 52.5s	remaining: 1m 26s
    1512:	learn: 10.1500688	test: 13.9957015	best: 13.9948903 (1510)	total: 52.5s	remaining: 1m 26s
    1513:	learn: 10.1476363	test: 13.9961324	best: 13.9948903 (1510)	total: 52.6s	remaining: 1m 26s
    1514:	learn: 10.1464897	test: 13.9959752	best: 13.9948903 (1510)	total: 52.6s	remaining: 1m 26s
    1515:	learn: 10.1453045	test: 13.9962405	best: 13.9948903 (1510)	total: 52.6s	remaining: 1m 26s
    1516:	learn: 10.1444332	test: 13.9955026	best: 13.9948903 (1510)	total: 52.6s	remaining: 1m 26s
    1517:	learn: 10.1428445	test: 13.9945175	best: 13.9945175 (1517)	total: 52.7s	remaining: 1m 26s
    1518:	learn: 10.1421360	test: 13.9940851	best: 13.9940851 (1518)	total: 52.7s	remaining: 1m 26s
    1519:	learn: 10.1384291	test: 13.9908549	best: 13.9908549 (1519)	total: 52.7s	remaining: 1m 26s
    1520:	learn: 10.1367569	test: 13.9899738	best: 13.9899738 (1520)	total: 52.7s	remaining: 1m 25s
    1521:	learn: 10.1348478	test: 13.9910626	best: 13.9899738 (1520)	total: 52.8s	remaining: 1m 25s
    1522:	learn: 10.1346159	test: 13.9929331	best: 13.9899738 (1520)	total: 52.8s	remaining: 1m 25s
    1523:	learn: 10.1316934	test: 13.9929326	best: 13.9899738 (1520)	total: 52.8s	remaining: 1m 25s
    1524:	learn: 10.1309591	test: 13.9925121	best: 13.9899738 (1520)	total: 52.9s	remaining: 1m 25s
    1525:	learn: 10.1303962	test: 13.9918636	best: 13.9899738 (1520)	total: 52.9s	remaining: 1m 25s
    1526:	learn: 10.1300710	test: 13.9917790	best: 13.9899738 (1520)	total: 52.9s	remaining: 1m 25s
    1527:	learn: 10.1286365	test: 13.9938933	best: 13.9899738 (1520)	total: 53s	remaining: 1m 25s
    1528:	learn: 10.1281528	test: 13.9939000	best: 13.9899738 (1520)	total: 53s	remaining: 1m 25s
    1529:	learn: 10.1277276	test: 13.9929685	best: 13.9899738 (1520)	total: 53.1s	remaining: 1m 25s
    1530:	learn: 10.1259940	test: 13.9936906	best: 13.9899738 (1520)	total: 53.1s	remaining: 1m 25s
    1531:	learn: 10.1231094	test: 13.9948148	best: 13.9899738 (1520)	total: 53.1s	remaining: 1m 25s
    1532:	learn: 10.1192825	test: 13.9930238	best: 13.9899738 (1520)	total: 53.1s	remaining: 1m 25s
    1533:	learn: 10.1175569	test: 13.9927946	best: 13.9899738 (1520)	total: 53.2s	remaining: 1m 25s
    1534:	learn: 10.1162852	test: 13.9915238	best: 13.9899738 (1520)	total: 53.2s	remaining: 1m 25s
    1535:	learn: 10.1145261	test: 13.9904513	best: 13.9899738 (1520)	total: 53.2s	remaining: 1m 25s
    1536:	learn: 10.1129692	test: 13.9909000	best: 13.9899738 (1520)	total: 53.3s	remaining: 1m 25s
    1537:	learn: 10.1117422	test: 13.9898909	best: 13.9898909 (1537)	total: 53.3s	remaining: 1m 25s
    1538:	learn: 10.1114349	test: 13.9903025	best: 13.9898909 (1537)	total: 53.3s	remaining: 1m 25s
    1539:	learn: 10.1104381	test: 13.9914261	best: 13.9898909 (1537)	total: 53.3s	remaining: 1m 25s
    1540:	learn: 10.1086336	test: 13.9887033	best: 13.9887033 (1540)	total: 53.4s	remaining: 1m 25s
    1541:	learn: 10.1081349	test: 13.9885001	best: 13.9885001 (1541)	total: 53.4s	remaining: 1m 25s
    1542:	learn: 10.1040076	test: 13.9873261	best: 13.9873261 (1542)	total: 53.4s	remaining: 1m 25s
    1543:	learn: 10.1028466	test: 13.9877445	best: 13.9873261 (1542)	total: 53.5s	remaining: 1m 25s
    1544:	learn: 10.1020218	test: 13.9879998	best: 13.9873261 (1542)	total: 53.5s	remaining: 1m 25s
    1545:	learn: 10.1013122	test: 13.9867380	best: 13.9867380 (1545)	total: 53.5s	remaining: 1m 24s
    1546:	learn: 10.0989069	test: 13.9863240	best: 13.9863240 (1546)	total: 53.6s	remaining: 1m 24s
    1547:	learn: 10.0971113	test: 13.9856474	best: 13.9856474 (1547)	total: 53.6s	remaining: 1m 24s
    1548:	learn: 10.0958099	test: 13.9859075	best: 13.9856474 (1547)	total: 53.6s	remaining: 1m 24s
    1549:	learn: 10.0950289	test: 13.9866462	best: 13.9856474 (1547)	total: 53.7s	remaining: 1m 24s
    1550:	learn: 10.0938013	test: 13.9860028	best: 13.9856474 (1547)	total: 53.7s	remaining: 1m 24s
    1551:	learn: 10.0915597	test: 13.9843686	best: 13.9843686 (1551)	total: 53.7s	remaining: 1m 24s
    1552:	learn: 10.0872143	test: 13.9848747	best: 13.9843686 (1551)	total: 53.8s	remaining: 1m 24s
    1553:	learn: 10.0865293	test: 13.9845948	best: 13.9843686 (1551)	total: 53.8s	remaining: 1m 24s
    1554:	learn: 10.0858262	test: 13.9843912	best: 13.9843686 (1551)	total: 53.8s	remaining: 1m 24s
    1555:	learn: 10.0854169	test: 13.9845813	best: 13.9843686 (1551)	total: 53.9s	remaining: 1m 24s
    1556:	learn: 10.0824140	test: 13.9838821	best: 13.9838821 (1556)	total: 53.9s	remaining: 1m 24s
    1557:	learn: 10.0809593	test: 13.9838539	best: 13.9838539 (1557)	total: 53.9s	remaining: 1m 24s
    1558:	learn: 10.0791385	test: 13.9825835	best: 13.9825835 (1558)	total: 54s	remaining: 1m 24s
    1559:	learn: 10.0774288	test: 13.9816362	best: 13.9816362 (1559)	total: 54s	remaining: 1m 24s
    1560:	learn: 10.0737965	test: 13.9796513	best: 13.9796513 (1560)	total: 54s	remaining: 1m 24s
    1561:	learn: 10.0726491	test: 13.9785667	best: 13.9785667 (1561)	total: 54s	remaining: 1m 24s
    1562:	learn: 10.0710231	test: 13.9781803	best: 13.9781803 (1562)	total: 54.1s	remaining: 1m 24s
    1563:	learn: 10.0684617	test: 13.9792145	best: 13.9781803 (1562)	total: 54.1s	remaining: 1m 24s
    1564:	learn: 10.0677212	test: 13.9780684	best: 13.9780684 (1564)	total: 54.1s	remaining: 1m 24s
    1565:	learn: 10.0634417	test: 13.9758395	best: 13.9758395 (1565)	total: 54.2s	remaining: 1m 24s
    1566:	learn: 10.0621063	test: 13.9769175	best: 13.9758395 (1565)	total: 54.2s	remaining: 1m 24s
    1567:	learn: 10.0592285	test: 13.9743851	best: 13.9743851 (1567)	total: 54.3s	remaining: 1m 24s
    1568:	learn: 10.0560524	test: 13.9727141	best: 13.9727141 (1568)	total: 54.3s	remaining: 1m 24s
    1569:	learn: 10.0546260	test: 13.9718357	best: 13.9718357 (1569)	total: 54.3s	remaining: 1m 24s
    1570:	learn: 10.0531807	test: 13.9715127	best: 13.9715127 (1570)	total: 54.4s	remaining: 1m 24s
    1571:	learn: 10.0523057	test: 13.9707847	best: 13.9707847 (1571)	total: 54.4s	remaining: 1m 24s
    1572:	learn: 10.0515277	test: 13.9710735	best: 13.9707847 (1571)	total: 54.4s	remaining: 1m 23s
    1573:	learn: 10.0511663	test: 13.9713105	best: 13.9707847 (1571)	total: 54.5s	remaining: 1m 23s
    1574:	learn: 10.0493763	test: 13.9688251	best: 13.9688251 (1574)	total: 54.5s	remaining: 1m 23s
    1575:	learn: 10.0464729	test: 13.9675132	best: 13.9675132 (1575)	total: 54.6s	remaining: 1m 23s
    1576:	learn: 10.0453665	test: 13.9659429	best: 13.9659429 (1576)	total: 54.6s	remaining: 1m 23s
    1577:	learn: 10.0441881	test: 13.9639401	best: 13.9639401 (1577)	total: 54.7s	remaining: 1m 23s
    1578:	learn: 10.0428660	test: 13.9646243	best: 13.9639401 (1577)	total: 54.8s	remaining: 1m 23s
    1579:	learn: 10.0404744	test: 13.9645427	best: 13.9639401 (1577)	total: 54.8s	remaining: 1m 23s
    1580:	learn: 10.0386013	test: 13.9640914	best: 13.9639401 (1577)	total: 54.9s	remaining: 1m 23s
    1581:	learn: 10.0372165	test: 13.9642832	best: 13.9639401 (1577)	total: 54.9s	remaining: 1m 23s
    1582:	learn: 10.0361683	test: 13.9635692	best: 13.9635692 (1582)	total: 55s	remaining: 1m 23s
    1583:	learn: 10.0349693	test: 13.9641018	best: 13.9635692 (1582)	total: 55s	remaining: 1m 23s
    1584:	learn: 10.0330025	test: 13.9644604	best: 13.9635692 (1582)	total: 55.1s	remaining: 1m 23s
    1585:	learn: 10.0314686	test: 13.9634741	best: 13.9634741 (1585)	total: 55.1s	remaining: 1m 23s
    1586:	learn: 10.0287790	test: 13.9621981	best: 13.9621981 (1586)	total: 55.1s	remaining: 1m 23s
    1587:	learn: 10.0275666	test: 13.9614268	best: 13.9614268 (1587)	total: 55.2s	remaining: 1m 23s
    1588:	learn: 10.0253871	test: 13.9598000	best: 13.9598000 (1588)	total: 55.2s	remaining: 1m 23s
    1589:	learn: 10.0229282	test: 13.9588080	best: 13.9588080 (1589)	total: 55.3s	remaining: 1m 23s
    1590:	learn: 10.0216519	test: 13.9585793	best: 13.9585793 (1590)	total: 55.3s	remaining: 1m 23s
    1591:	learn: 10.0196636	test: 13.9587272	best: 13.9585793 (1590)	total: 55.3s	remaining: 1m 23s
    1592:	learn: 10.0174354	test: 13.9603388	best: 13.9585793 (1590)	total: 55.4s	remaining: 1m 23s
    1593:	learn: 10.0160416	test: 13.9583253	best: 13.9583253 (1593)	total: 55.4s	remaining: 1m 23s
    1594:	learn: 10.0141687	test: 13.9589821	best: 13.9583253 (1593)	total: 55.5s	remaining: 1m 23s
    1595:	learn: 10.0130158	test: 13.9596043	best: 13.9583253 (1593)	total: 55.5s	remaining: 1m 23s
    1596:	learn: 10.0116651	test: 13.9592175	best: 13.9583253 (1593)	total: 55.5s	remaining: 1m 23s
    1597:	learn: 10.0100928	test: 13.9601565	best: 13.9583253 (1593)	total: 55.5s	remaining: 1m 23s
    1598:	learn: 10.0094032	test: 13.9600277	best: 13.9583253 (1593)	total: 55.6s	remaining: 1m 23s
    1599:	learn: 10.0078042	test: 13.9605616	best: 13.9583253 (1593)	total: 55.6s	remaining: 1m 23s
    1600:	learn: 10.0069134	test: 13.9602101	best: 13.9583253 (1593)	total: 55.6s	remaining: 1m 23s
    1601:	learn: 10.0040463	test: 13.9580640	best: 13.9580640 (1601)	total: 55.7s	remaining: 1m 23s
    1602:	learn: 10.0015935	test: 13.9574691	best: 13.9574691 (1602)	total: 55.7s	remaining: 1m 23s
    1603:	learn: 10.0000297	test: 13.9559646	best: 13.9559646 (1603)	total: 55.7s	remaining: 1m 23s
    1604:	learn: 9.9980104	test: 13.9559138	best: 13.9559138 (1604)	total: 55.8s	remaining: 1m 23s
    1605:	learn: 9.9972618	test: 13.9560728	best: 13.9559138 (1604)	total: 55.8s	remaining: 1m 23s
    1606:	learn: 9.9968201	test: 13.9572827	best: 13.9559138 (1604)	total: 55.9s	remaining: 1m 23s
    1607:	learn: 9.9955139	test: 13.9561806	best: 13.9559138 (1604)	total: 55.9s	remaining: 1m 23s
    1608:	learn: 9.9940571	test: 13.9551392	best: 13.9551392 (1608)	total: 55.9s	remaining: 1m 23s
    1609:	learn: 9.9936263	test: 13.9562668	best: 13.9551392 (1608)	total: 56s	remaining: 1m 23s
    1610:	learn: 9.9928114	test: 13.9554787	best: 13.9551392 (1608)	total: 56s	remaining: 1m 23s
    1611:	learn: 9.9923225	test: 13.9545730	best: 13.9545730 (1611)	total: 56.1s	remaining: 1m 23s
    1612:	learn: 9.9911932	test: 13.9534178	best: 13.9534178 (1612)	total: 56.1s	remaining: 1m 23s
    1613:	learn: 9.9903587	test: 13.9534955	best: 13.9534178 (1612)	total: 56.1s	remaining: 1m 22s
    1614:	learn: 9.9894043	test: 13.9535704	best: 13.9534178 (1612)	total: 56.2s	remaining: 1m 22s
    1615:	learn: 9.9873385	test: 13.9536741	best: 13.9534178 (1612)	total: 56.2s	remaining: 1m 22s
    1616:	learn: 9.9857393	test: 13.9539786	best: 13.9534178 (1612)	total: 56.2s	remaining: 1m 22s
    1617:	learn: 9.9843389	test: 13.9539226	best: 13.9534178 (1612)	total: 56.3s	remaining: 1m 22s
    1618:	learn: 9.9823555	test: 13.9528549	best: 13.9528549 (1618)	total: 56.3s	remaining: 1m 22s
    1619:	learn: 9.9812612	test: 13.9522351	best: 13.9522351 (1619)	total: 56.3s	remaining: 1m 22s
    1620:	learn: 9.9787932	test: 13.9514859	best: 13.9514859 (1620)	total: 56.4s	remaining: 1m 22s
    1621:	learn: 9.9774108	test: 13.9495222	best: 13.9495222 (1621)	total: 56.4s	remaining: 1m 22s
    1622:	learn: 9.9763583	test: 13.9486453	best: 13.9486453 (1622)	total: 56.4s	remaining: 1m 22s
    1623:	learn: 9.9747496	test: 13.9482166	best: 13.9482166 (1623)	total: 56.5s	remaining: 1m 22s
    1624:	learn: 9.9710752	test: 13.9460926	best: 13.9460926 (1624)	total: 56.6s	remaining: 1m 22s
    1625:	learn: 9.9703148	test: 13.9463448	best: 13.9460926 (1624)	total: 56.6s	remaining: 1m 22s
    1626:	learn: 9.9701569	test: 13.9463629	best: 13.9460926 (1624)	total: 56.7s	remaining: 1m 22s
    1627:	learn: 9.9691090	test: 13.9457143	best: 13.9457143 (1627)	total: 56.7s	remaining: 1m 22s
    1628:	learn: 9.9683672	test: 13.9449771	best: 13.9449771 (1628)	total: 56.7s	remaining: 1m 22s
    1629:	learn: 9.9675044	test: 13.9443284	best: 13.9443284 (1629)	total: 56.8s	remaining: 1m 22s
    1630:	learn: 9.9659712	test: 13.9429318	best: 13.9429318 (1630)	total: 56.8s	remaining: 1m 22s
    1631:	learn: 9.9654714	test: 13.9424914	best: 13.9424914 (1631)	total: 56.8s	remaining: 1m 22s
    1632:	learn: 9.9634402	test: 13.9405373	best: 13.9405373 (1632)	total: 56.9s	remaining: 1m 22s
    1633:	learn: 9.9626055	test: 13.9403299	best: 13.9403299 (1633)	total: 56.9s	remaining: 1m 22s
    1634:	learn: 9.9612962	test: 13.9407695	best: 13.9403299 (1633)	total: 56.9s	remaining: 1m 22s
    1635:	learn: 9.9602428	test: 13.9410946	best: 13.9403299 (1633)	total: 57s	remaining: 1m 22s
    1636:	learn: 9.9593181	test: 13.9401855	best: 13.9401855 (1636)	total: 57s	remaining: 1m 22s
    1637:	learn: 9.9581604	test: 13.9404734	best: 13.9401855 (1636)	total: 57s	remaining: 1m 22s
    1638:	learn: 9.9573574	test: 13.9397715	best: 13.9397715 (1638)	total: 57s	remaining: 1m 22s
    1639:	learn: 9.9567253	test: 13.9395153	best: 13.9395153 (1639)	total: 57.1s	remaining: 1m 22s
    1640:	learn: 9.9554581	test: 13.9398095	best: 13.9395153 (1639)	total: 57.1s	remaining: 1m 22s
    1641:	learn: 9.9539430	test: 13.9410993	best: 13.9395153 (1639)	total: 57.1s	remaining: 1m 22s
    1642:	learn: 9.9529936	test: 13.9406239	best: 13.9395153 (1639)	total: 57.2s	remaining: 1m 22s
    1643:	learn: 9.9515864	test: 13.9399368	best: 13.9395153 (1639)	total: 57.2s	remaining: 1m 21s
    1644:	learn: 9.9485492	test: 13.9378066	best: 13.9378066 (1644)	total: 57.2s	remaining: 1m 21s
    1645:	learn: 9.9466371	test: 13.9382261	best: 13.9378066 (1644)	total: 57.2s	remaining: 1m 21s
    1646:	learn: 9.9449270	test: 13.9373110	best: 13.9373110 (1646)	total: 57.3s	remaining: 1m 21s
    1647:	learn: 9.9434531	test: 13.9358367	best: 13.9358367 (1647)	total: 57.3s	remaining: 1m 21s
    1648:	learn: 9.9416423	test: 13.9345962	best: 13.9345962 (1648)	total: 57.3s	remaining: 1m 21s
    1649:	learn: 9.9385540	test: 13.9343047	best: 13.9343047 (1649)	total: 57.4s	remaining: 1m 21s
    1650:	learn: 9.9377299	test: 13.9335501	best: 13.9335501 (1650)	total: 57.4s	remaining: 1m 21s
    1651:	learn: 9.9362283	test: 13.9340340	best: 13.9335501 (1650)	total: 57.4s	remaining: 1m 21s
    1652:	learn: 9.9353588	test: 13.9334615	best: 13.9334615 (1652)	total: 57.4s	remaining: 1m 21s
    1653:	learn: 9.9331986	test: 13.9323427	best: 13.9323427 (1653)	total: 57.5s	remaining: 1m 21s
    1654:	learn: 9.9306730	test: 13.9300526	best: 13.9300526 (1654)	total: 57.5s	remaining: 1m 21s
    1655:	learn: 9.9298756	test: 13.9302737	best: 13.9300526 (1654)	total: 57.5s	remaining: 1m 21s
    1656:	learn: 9.9283895	test: 13.9307396	best: 13.9300526 (1654)	total: 57.6s	remaining: 1m 21s
    1657:	learn: 9.9272129	test: 13.9313639	best: 13.9300526 (1654)	total: 57.6s	remaining: 1m 21s
    1658:	learn: 9.9260094	test: 13.9310445	best: 13.9300526 (1654)	total: 57.6s	remaining: 1m 21s
    1659:	learn: 9.9247143	test: 13.9298716	best: 13.9298716 (1659)	total: 57.7s	remaining: 1m 21s
    1660:	learn: 9.9238843	test: 13.9298442	best: 13.9298442 (1660)	total: 57.7s	remaining: 1m 21s
    1661:	learn: 9.9224017	test: 13.9276592	best: 13.9276592 (1661)	total: 57.7s	remaining: 1m 21s
    1662:	learn: 9.9196026	test: 13.9243715	best: 13.9243715 (1662)	total: 57.8s	remaining: 1m 21s
    1663:	learn: 9.9173278	test: 13.9241331	best: 13.9241331 (1663)	total: 57.8s	remaining: 1m 21s
    1664:	learn: 9.9154857	test: 13.9219480	best: 13.9219480 (1664)	total: 57.8s	remaining: 1m 21s
    1665:	learn: 9.9137793	test: 13.9210725	best: 13.9210725 (1665)	total: 57.9s	remaining: 1m 21s
    1666:	learn: 9.9129481	test: 13.9197997	best: 13.9197997 (1666)	total: 57.9s	remaining: 1m 21s
    1667:	learn: 9.9110018	test: 13.9179995	best: 13.9179995 (1667)	total: 57.9s	remaining: 1m 20s
    1668:	learn: 9.9094290	test: 13.9189494	best: 13.9179995 (1667)	total: 57.9s	remaining: 1m 20s
    1669:	learn: 9.9085074	test: 13.9189346	best: 13.9179995 (1667)	total: 58s	remaining: 1m 20s
    1670:	learn: 9.9068577	test: 13.9189511	best: 13.9179995 (1667)	total: 58s	remaining: 1m 20s
    1671:	learn: 9.9053834	test: 13.9184434	best: 13.9179995 (1667)	total: 58s	remaining: 1m 20s
    1672:	learn: 9.9046892	test: 13.9188110	best: 13.9179995 (1667)	total: 58.1s	remaining: 1m 20s
    1673:	learn: 9.9019589	test: 13.9189229	best: 13.9179995 (1667)	total: 58.1s	remaining: 1m 20s
    1674:	learn: 9.9001595	test: 13.9178413	best: 13.9178413 (1674)	total: 58.2s	remaining: 1m 20s
    1675:	learn: 9.8973298	test: 13.9146460	best: 13.9146460 (1675)	total: 58.2s	remaining: 1m 20s
    1676:	learn: 9.8954158	test: 13.9146624	best: 13.9146460 (1675)	total: 58.2s	remaining: 1m 20s
    1677:	learn: 9.8948475	test: 13.9144316	best: 13.9144316 (1677)	total: 58.2s	remaining: 1m 20s
    1678:	learn: 9.8932783	test: 13.9119875	best: 13.9119875 (1678)	total: 58.3s	remaining: 1m 20s
    1679:	learn: 9.8927162	test: 13.9120758	best: 13.9119875 (1678)	total: 58.3s	remaining: 1m 20s
    1680:	learn: 9.8890965	test: 13.9082326	best: 13.9082326 (1680)	total: 58.4s	remaining: 1m 20s
    1681:	learn: 9.8885391	test: 13.9075752	best: 13.9075752 (1681)	total: 58.4s	remaining: 1m 20s
    1682:	learn: 9.8871960	test: 13.9077918	best: 13.9075752 (1681)	total: 58.4s	remaining: 1m 20s
    1683:	learn: 9.8861490	test: 13.9087948	best: 13.9075752 (1681)	total: 58.5s	remaining: 1m 20s
    1684:	learn: 9.8854314	test: 13.9087659	best: 13.9075752 (1681)	total: 58.5s	remaining: 1m 20s
    1685:	learn: 9.8842785	test: 13.9082281	best: 13.9075752 (1681)	total: 58.5s	remaining: 1m 20s
    1686:	learn: 9.8830127	test: 13.9084070	best: 13.9075752 (1681)	total: 58.6s	remaining: 1m 20s
    1687:	learn: 9.8815451	test: 13.9060047	best: 13.9060047 (1687)	total: 58.6s	remaining: 1m 20s
    1688:	learn: 9.8786607	test: 13.9051725	best: 13.9051725 (1688)	total: 58.7s	remaining: 1m 20s
    1689:	learn: 9.8775290	test: 13.9053620	best: 13.9051725 (1688)	total: 58.7s	remaining: 1m 20s
    1690:	learn: 9.8761454	test: 13.9071333	best: 13.9051725 (1688)	total: 58.7s	remaining: 1m 20s
    1691:	learn: 9.8744891	test: 13.9052456	best: 13.9051725 (1688)	total: 58.8s	remaining: 1m 20s
    1692:	learn: 9.8707529	test: 13.9031656	best: 13.9031656 (1692)	total: 58.8s	remaining: 1m 20s
    1693:	learn: 9.8691966	test: 13.9036429	best: 13.9031656 (1692)	total: 58.8s	remaining: 1m 20s
    1694:	learn: 9.8689974	test: 13.9040175	best: 13.9031656 (1692)	total: 58.9s	remaining: 1m 20s
    1695:	learn: 9.8684645	test: 13.9029752	best: 13.9029752 (1695)	total: 58.9s	remaining: 1m 20s
    1696:	learn: 9.8672241	test: 13.9019689	best: 13.9019689 (1696)	total: 58.9s	remaining: 1m 19s
    1697:	learn: 9.8664573	test: 13.9024822	best: 13.9019689 (1696)	total: 59s	remaining: 1m 19s
    1698:	learn: 9.8647558	test: 13.9030508	best: 13.9019689 (1696)	total: 59s	remaining: 1m 19s
    1699:	learn: 9.8632771	test: 13.9020924	best: 13.9019689 (1696)	total: 59s	remaining: 1m 19s
    1700:	learn: 9.8603925	test: 13.9002630	best: 13.9002630 (1700)	total: 59.1s	remaining: 1m 19s
    1701:	learn: 9.8600555	test: 13.8999517	best: 13.8999517 (1701)	total: 59.1s	remaining: 1m 19s
    1702:	learn: 9.8586640	test: 13.8996208	best: 13.8996208 (1702)	total: 59.1s	remaining: 1m 19s
    1703:	learn: 9.8561614	test: 13.9009178	best: 13.8996208 (1702)	total: 59.2s	remaining: 1m 19s
    1704:	learn: 9.8535033	test: 13.9006035	best: 13.8996208 (1702)	total: 59.2s	remaining: 1m 19s
    1705:	learn: 9.8521910	test: 13.8990199	best: 13.8990199 (1705)	total: 59.3s	remaining: 1m 19s
    1706:	learn: 9.8501899	test: 13.8983874	best: 13.8983874 (1706)	total: 59.3s	remaining: 1m 19s
    1707:	learn: 9.8492973	test: 13.8965481	best: 13.8965481 (1707)	total: 59.3s	remaining: 1m 19s
    1708:	learn: 9.8474213	test: 13.8965057	best: 13.8965057 (1708)	total: 59.4s	remaining: 1m 19s
    1709:	learn: 9.8462917	test: 13.8970714	best: 13.8965057 (1708)	total: 59.4s	remaining: 1m 19s
    1710:	learn: 9.8449854	test: 13.8972651	best: 13.8965057 (1708)	total: 59.4s	remaining: 1m 19s
    1711:	learn: 9.8446552	test: 13.8974060	best: 13.8965057 (1708)	total: 59.5s	remaining: 1m 19s
    1712:	learn: 9.8435442	test: 13.8965284	best: 13.8965057 (1708)	total: 59.6s	remaining: 1m 19s
    1713:	learn: 9.8405638	test: 13.8926493	best: 13.8926493 (1713)	total: 59.7s	remaining: 1m 19s
    1714:	learn: 9.8395664	test: 13.8928903	best: 13.8926493 (1713)	total: 59.7s	remaining: 1m 19s
    1715:	learn: 9.8387845	test: 13.8927526	best: 13.8926493 (1713)	total: 59.7s	remaining: 1m 19s
    1716:	learn: 9.8366151	test: 13.8907426	best: 13.8907426 (1716)	total: 59.8s	remaining: 1m 19s
    1717:	learn: 9.8352431	test: 13.8902969	best: 13.8902969 (1717)	total: 59.8s	remaining: 1m 19s
    1718:	learn: 9.8332346	test: 13.8906223	best: 13.8902969 (1717)	total: 59.9s	remaining: 1m 19s
    1719:	learn: 9.8323406	test: 13.8911099	best: 13.8902969 (1717)	total: 59.9s	remaining: 1m 19s
    1720:	learn: 9.8319338	test: 13.8905961	best: 13.8902969 (1717)	total: 59.9s	remaining: 1m 19s
    1721:	learn: 9.8287829	test: 13.8874152	best: 13.8874152 (1721)	total: 60s	remaining: 1m 19s
    1722:	learn: 9.8254535	test: 13.8854470	best: 13.8854470 (1722)	total: 1m	remaining: 1m 19s
    1723:	learn: 9.8249473	test: 13.8851640	best: 13.8851640 (1723)	total: 1m	remaining: 1m 19s
    1724:	learn: 9.8239173	test: 13.8848850	best: 13.8848850 (1724)	total: 1m	remaining: 1m 19s
    1725:	learn: 9.8227475	test: 13.8847130	best: 13.8847130 (1725)	total: 1m	remaining: 1m 19s
    1726:	learn: 9.8219583	test: 13.8848519	best: 13.8847130 (1725)	total: 1m	remaining: 1m 19s
    1727:	learn: 9.8208997	test: 13.8838737	best: 13.8838737 (1727)	total: 1m	remaining: 1m 19s
    1728:	learn: 9.8203627	test: 13.8841857	best: 13.8838737 (1727)	total: 1m	remaining: 1m 19s
    1729:	learn: 9.8193303	test: 13.8842838	best: 13.8838737 (1727)	total: 1m	remaining: 1m 19s
    1730:	learn: 9.8187866	test: 13.8849218	best: 13.8838737 (1727)	total: 1m	remaining: 1m 19s
    1731:	learn: 9.8183023	test: 13.8849964	best: 13.8838737 (1727)	total: 1m	remaining: 1m 19s
    1732:	learn: 9.8173796	test: 13.8839432	best: 13.8838737 (1727)	total: 1m	remaining: 1m 19s
    1733:	learn: 9.8164536	test: 13.8818408	best: 13.8818408 (1733)	total: 1m	remaining: 1m 19s
    1734:	learn: 9.8154810	test: 13.8813955	best: 13.8813955 (1734)	total: 1m	remaining: 1m 19s
    1735:	learn: 9.8147599	test: 13.8811391	best: 13.8811391 (1735)	total: 1m	remaining: 1m 19s
    1736:	learn: 9.8121588	test: 13.8780875	best: 13.8780875 (1736)	total: 1m	remaining: 1m 19s
    1737:	learn: 9.8109272	test: 13.8784572	best: 13.8780875 (1736)	total: 1m	remaining: 1m 19s
    1738:	learn: 9.8093100	test: 13.8780963	best: 13.8780875 (1736)	total: 1m	remaining: 1m 19s
    1739:	learn: 9.8080437	test: 13.8783342	best: 13.8780875 (1736)	total: 1m	remaining: 1m 19s
    1740:	learn: 9.8072730	test: 13.8781204	best: 13.8780875 (1736)	total: 1m 1s	remaining: 1m 19s
    1741:	learn: 9.8046586	test: 13.8789780	best: 13.8780875 (1736)	total: 1m 1s	remaining: 1m 19s
    1742:	learn: 9.8035910	test: 13.8783792	best: 13.8780875 (1736)	total: 1m 1s	remaining: 1m 19s
    1743:	learn: 9.8029397	test: 13.8793164	best: 13.8780875 (1736)	total: 1m 1s	remaining: 1m 19s
    1744:	learn: 9.7990109	test: 13.8772199	best: 13.8772199 (1744)	total: 1m 1s	remaining: 1m 19s
    1745:	learn: 9.7981237	test: 13.8763306	best: 13.8763306 (1745)	total: 1m 1s	remaining: 1m 19s
    1746:	learn: 9.7971122	test: 13.8768741	best: 13.8763306 (1745)	total: 1m 1s	remaining: 1m 19s
    1747:	learn: 9.7958139	test: 13.8772878	best: 13.8763306 (1745)	total: 1m 1s	remaining: 1m 19s
    1748:	learn: 9.7954061	test: 13.8776246	best: 13.8763306 (1745)	total: 1m 1s	remaining: 1m 19s
    1749:	learn: 9.7946405	test: 13.8777939	best: 13.8763306 (1745)	total: 1m 1s	remaining: 1m 19s
    1750:	learn: 9.7925472	test: 13.8777944	best: 13.8763306 (1745)	total: 1m 1s	remaining: 1m 19s
    1751:	learn: 9.7912524	test: 13.8763901	best: 13.8763306 (1745)	total: 1m 1s	remaining: 1m 19s
    1752:	learn: 9.7903948	test: 13.8751989	best: 13.8751989 (1752)	total: 1m 1s	remaining: 1m 19s
    1753:	learn: 9.7895033	test: 13.8750401	best: 13.8750401 (1753)	total: 1m 1s	remaining: 1m 19s
    1754:	learn: 9.7889419	test: 13.8751359	best: 13.8750401 (1753)	total: 1m 1s	remaining: 1m 19s
    1755:	learn: 9.7878844	test: 13.8756770	best: 13.8750401 (1753)	total: 1m 1s	remaining: 1m 19s
    1756:	learn: 9.7844851	test: 13.8750022	best: 13.8750022 (1756)	total: 1m 1s	remaining: 1m 19s
    1757:	learn: 9.7832752	test: 13.8743736	best: 13.8743736 (1757)	total: 1m 1s	remaining: 1m 19s
    1758:	learn: 9.7821959	test: 13.8750572	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1759:	learn: 9.7817636	test: 13.8754090	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1760:	learn: 9.7804056	test: 13.8763328	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1761:	learn: 9.7760299	test: 13.8766582	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1762:	learn: 9.7751652	test: 13.8783812	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1763:	learn: 9.7737817	test: 13.8784642	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1764:	learn: 9.7731760	test: 13.8791509	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1765:	learn: 9.7720353	test: 13.8791341	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1766:	learn: 9.7714938	test: 13.8786506	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1767:	learn: 9.7684740	test: 13.8779856	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1768:	learn: 9.7670117	test: 13.8773139	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1769:	learn: 9.7667706	test: 13.8770954	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1770:	learn: 9.7658206	test: 13.8770130	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1771:	learn: 9.7644324	test: 13.8771620	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1772:	learn: 9.7624033	test: 13.8781057	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1773:	learn: 9.7600413	test: 13.8785433	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1774:	learn: 9.7569295	test: 13.8782105	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1775:	learn: 9.7538259	test: 13.8772242	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1776:	learn: 9.7523970	test: 13.8770774	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1777:	learn: 9.7500666	test: 13.8761639	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1778:	learn: 9.7487717	test: 13.8772407	best: 13.8743736 (1757)	total: 1m 2s	remaining: 1m 18s
    1779:	learn: 9.7480475	test: 13.8779947	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1780:	learn: 9.7477621	test: 13.8778822	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1781:	learn: 9.7474017	test: 13.8778512	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1782:	learn: 9.7462594	test: 13.8786346	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1783:	learn: 9.7439419	test: 13.8785922	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1784:	learn: 9.7428369	test: 13.8786482	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1785:	learn: 9.7417915	test: 13.8777539	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1786:	learn: 9.7406752	test: 13.8751457	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1787:	learn: 9.7402793	test: 13.8755195	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1788:	learn: 9.7388185	test: 13.8756924	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1789:	learn: 9.7382421	test: 13.8759337	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1790:	learn: 9.7367443	test: 13.8746014	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1791:	learn: 9.7357720	test: 13.8752817	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1792:	learn: 9.7349971	test: 13.8754945	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1793:	learn: 9.7332735	test: 13.8763989	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1794:	learn: 9.7322903	test: 13.8755061	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1795:	learn: 9.7317415	test: 13.8754049	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1796:	learn: 9.7300111	test: 13.8746099	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1797:	learn: 9.7273156	test: 13.8759724	best: 13.8743736 (1757)	total: 1m 3s	remaining: 1m 18s
    1798:	learn: 9.7243684	test: 13.8739414	best: 13.8739414 (1798)	total: 1m 3s	remaining: 1m 18s
    1799:	learn: 9.7214278	test: 13.8730803	best: 13.8730803 (1799)	total: 1m 3s	remaining: 1m 18s
    1800:	learn: 9.7205135	test: 13.8729863	best: 13.8729863 (1800)	total: 1m 3s	remaining: 1m 18s
    1801:	learn: 9.7197395	test: 13.8730474	best: 13.8729863 (1800)	total: 1m 3s	remaining: 1m 17s
    1802:	learn: 9.7185014	test: 13.8744668	best: 13.8729863 (1800)	total: 1m 3s	remaining: 1m 17s
    1803:	learn: 9.7172761	test: 13.8740469	best: 13.8729863 (1800)	total: 1m 4s	remaining: 1m 17s
    1804:	learn: 9.7160606	test: 13.8726107	best: 13.8726107 (1804)	total: 1m 4s	remaining: 1m 17s
    1805:	learn: 9.7150250	test: 13.8727142	best: 13.8726107 (1804)	total: 1m 4s	remaining: 1m 17s
    1806:	learn: 9.7136357	test: 13.8732283	best: 13.8726107 (1804)	total: 1m 4s	remaining: 1m 17s
    1807:	learn: 9.7115650	test: 13.8723893	best: 13.8723893 (1807)	total: 1m 4s	remaining: 1m 17s
    1808:	learn: 9.7103445	test: 13.8723548	best: 13.8723548 (1808)	total: 1m 4s	remaining: 1m 17s
    1809:	learn: 9.7096189	test: 13.8714327	best: 13.8714327 (1809)	total: 1m 4s	remaining: 1m 17s
    1810:	learn: 9.7077166	test: 13.8708066	best: 13.8708066 (1810)	total: 1m 4s	remaining: 1m 17s
    1811:	learn: 9.7068278	test: 13.8706491	best: 13.8706491 (1811)	total: 1m 4s	remaining: 1m 17s
    1812:	learn: 9.7059149	test: 13.8706451	best: 13.8706451 (1812)	total: 1m 4s	remaining: 1m 17s
    1813:	learn: 9.7042857	test: 13.8713657	best: 13.8706451 (1812)	total: 1m 4s	remaining: 1m 17s
    1814:	learn: 9.7034202	test: 13.8712466	best: 13.8706451 (1812)	total: 1m 4s	remaining: 1m 17s
    1815:	learn: 9.7028883	test: 13.8710981	best: 13.8706451 (1812)	total: 1m 4s	remaining: 1m 17s
    1816:	learn: 9.7025578	test: 13.8697993	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1817:	learn: 9.7001636	test: 13.8703247	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1818:	learn: 9.6974578	test: 13.8710232	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1819:	learn: 9.6962702	test: 13.8705138	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1820:	learn: 9.6950748	test: 13.8719176	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1821:	learn: 9.6933827	test: 13.8719114	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1822:	learn: 9.6925577	test: 13.8706002	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1823:	learn: 9.6911254	test: 13.8700264	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1824:	learn: 9.6901572	test: 13.8704515	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1825:	learn: 9.6895814	test: 13.8707284	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1826:	learn: 9.6887962	test: 13.8709407	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1827:	learn: 9.6875393	test: 13.8714908	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 17s
    1828:	learn: 9.6858610	test: 13.8720496	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 16s
    1829:	learn: 9.6852467	test: 13.8718307	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 16s
    1830:	learn: 9.6848555	test: 13.8717839	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 16s
    1831:	learn: 9.6830938	test: 13.8708747	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 16s
    1832:	learn: 9.6821965	test: 13.8707752	best: 13.8697993 (1816)	total: 1m 4s	remaining: 1m 16s
    1833:	learn: 9.6815669	test: 13.8707069	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1834:	learn: 9.6800107	test: 13.8713214	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1835:	learn: 9.6793154	test: 13.8706170	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1836:	learn: 9.6760678	test: 13.8718197	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1837:	learn: 9.6745442	test: 13.8721490	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1838:	learn: 9.6740534	test: 13.8729745	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1839:	learn: 9.6726249	test: 13.8734021	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1840:	learn: 9.6723044	test: 13.8732395	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1841:	learn: 9.6715006	test: 13.8733020	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1842:	learn: 9.6698409	test: 13.8748769	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1843:	learn: 9.6682188	test: 13.8753718	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1844:	learn: 9.6669921	test: 13.8730117	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1845:	learn: 9.6665811	test: 13.8719076	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1846:	learn: 9.6647985	test: 13.8702823	best: 13.8697993 (1816)	total: 1m 5s	remaining: 1m 16s
    1847:	learn: 9.6639868	test: 13.8696147	best: 13.8696147 (1847)	total: 1m 5s	remaining: 1m 16s
    1848:	learn: 9.6628467	test: 13.8685236	best: 13.8685236 (1848)	total: 1m 5s	remaining: 1m 16s
    1849:	learn: 9.6613765	test: 13.8684918	best: 13.8684918 (1849)	total: 1m 5s	remaining: 1m 16s
    1850:	learn: 9.6574077	test: 13.8629882	best: 13.8629882 (1850)	total: 1m 5s	remaining: 1m 16s
    1851:	learn: 9.6560724	test: 13.8633217	best: 13.8629882 (1850)	total: 1m 5s	remaining: 1m 16s
    1852:	learn: 9.6536282	test: 13.8619587	best: 13.8619587 (1852)	total: 1m 5s	remaining: 1m 16s
    1853:	learn: 9.6522830	test: 13.8611011	best: 13.8611011 (1853)	total: 1m 5s	remaining: 1m 16s
    1854:	learn: 9.6513465	test: 13.8608282	best: 13.8608282 (1854)	total: 1m 5s	remaining: 1m 15s
    1855:	learn: 9.6493850	test: 13.8618822	best: 13.8608282 (1854)	total: 1m 5s	remaining: 1m 15s
    1856:	learn: 9.6482675	test: 13.8624939	best: 13.8608282 (1854)	total: 1m 5s	remaining: 1m 15s
    1857:	learn: 9.6478866	test: 13.8618315	best: 13.8608282 (1854)	total: 1m 5s	remaining: 1m 15s
    1858:	learn: 9.6475881	test: 13.8624597	best: 13.8608282 (1854)	total: 1m 5s	remaining: 1m 15s
    1859:	learn: 9.6469296	test: 13.8627868	best: 13.8608282 (1854)	total: 1m 5s	remaining: 1m 15s
    1860:	learn: 9.6446558	test: 13.8612154	best: 13.8608282 (1854)	total: 1m 5s	remaining: 1m 15s
    1861:	learn: 9.6433131	test: 13.8599097	best: 13.8599097 (1861)	total: 1m 5s	remaining: 1m 15s
    1862:	learn: 9.6427286	test: 13.8602209	best: 13.8599097 (1861)	total: 1m 5s	remaining: 1m 15s
    1863:	learn: 9.6391895	test: 13.8560498	best: 13.8560498 (1863)	total: 1m 6s	remaining: 1m 15s
    1864:	learn: 9.6367010	test: 13.8559362	best: 13.8559362 (1864)	total: 1m 6s	remaining: 1m 15s
    1865:	learn: 9.6352899	test: 13.8537658	best: 13.8537658 (1865)	total: 1m 6s	remaining: 1m 15s
    1866:	learn: 9.6350114	test: 13.8540824	best: 13.8537658 (1865)	total: 1m 6s	remaining: 1m 15s
    1867:	learn: 9.6324826	test: 13.8540452	best: 13.8537658 (1865)	total: 1m 6s	remaining: 1m 15s
    1868:	learn: 9.6313059	test: 13.8535858	best: 13.8535858 (1868)	total: 1m 6s	remaining: 1m 15s
    1869:	learn: 9.6294641	test: 13.8522344	best: 13.8522344 (1869)	total: 1m 6s	remaining: 1m 15s
    1870:	learn: 9.6269912	test: 13.8506888	best: 13.8506888 (1870)	total: 1m 6s	remaining: 1m 15s
    1871:	learn: 9.6267242	test: 13.8511727	best: 13.8506888 (1870)	total: 1m 6s	remaining: 1m 15s
    1872:	learn: 9.6249333	test: 13.8493466	best: 13.8493466 (1872)	total: 1m 6s	remaining: 1m 15s
    1873:	learn: 9.6231299	test: 13.8511793	best: 13.8493466 (1872)	total: 1m 6s	remaining: 1m 15s
    1874:	learn: 9.6217006	test: 13.8522978	best: 13.8493466 (1872)	total: 1m 6s	remaining: 1m 15s
    1875:	learn: 9.6196881	test: 13.8527457	best: 13.8493466 (1872)	total: 1m 6s	remaining: 1m 15s
    1876:	learn: 9.6182799	test: 13.8529028	best: 13.8493466 (1872)	total: 1m 6s	remaining: 1m 15s
    1877:	learn: 9.6167624	test: 13.8531449	best: 13.8493466 (1872)	total: 1m 6s	remaining: 1m 15s
    1878:	learn: 9.6157379	test: 13.8541997	best: 13.8493466 (1872)	total: 1m 6s	remaining: 1m 15s
    1879:	learn: 9.6139544	test: 13.8518481	best: 13.8493466 (1872)	total: 1m 6s	remaining: 1m 14s
    1880:	learn: 9.6137924	test: 13.8511445	best: 13.8493466 (1872)	total: 1m 6s	remaining: 1m 14s
    1881:	learn: 9.6106995	test: 13.8487274	best: 13.8487274 (1881)	total: 1m 6s	remaining: 1m 14s
    1882:	learn: 9.6095514	test: 13.8479577	best: 13.8479577 (1882)	total: 1m 6s	remaining: 1m 14s
    1883:	learn: 9.6088357	test: 13.8482544	best: 13.8479577 (1882)	total: 1m 6s	remaining: 1m 14s
    1884:	learn: 9.6082941	test: 13.8482903	best: 13.8479577 (1882)	total: 1m 6s	remaining: 1m 14s
    1885:	learn: 9.6069484	test: 13.8489335	best: 13.8479577 (1882)	total: 1m 6s	remaining: 1m 14s
    1886:	learn: 9.6050857	test: 13.8480043	best: 13.8479577 (1882)	total: 1m 6s	remaining: 1m 14s
    1887:	learn: 9.6034964	test: 13.8476699	best: 13.8476699 (1887)	total: 1m 6s	remaining: 1m 14s
    1888:	learn: 9.6015801	test: 13.8464148	best: 13.8464148 (1888)	total: 1m 6s	remaining: 1m 14s
    1889:	learn: 9.6009829	test: 13.8472935	best: 13.8464148 (1888)	total: 1m 6s	remaining: 1m 14s
    1890:	learn: 9.6000136	test: 13.8476125	best: 13.8464148 (1888)	total: 1m 6s	remaining: 1m 14s
    1891:	learn: 9.5990811	test: 13.8484530	best: 13.8464148 (1888)	total: 1m 6s	remaining: 1m 14s
    1892:	learn: 9.5973269	test: 13.8490060	best: 13.8464148 (1888)	total: 1m 6s	remaining: 1m 14s
    1893:	learn: 9.5961835	test: 13.8480093	best: 13.8464148 (1888)	total: 1m 6s	remaining: 1m 14s
    1894:	learn: 9.5951099	test: 13.8470179	best: 13.8464148 (1888)	total: 1m 6s	remaining: 1m 14s
    1895:	learn: 9.5921332	test: 13.8476649	best: 13.8464148 (1888)	total: 1m 6s	remaining: 1m 14s
    1896:	learn: 9.5913011	test: 13.8476769	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 14s
    1897:	learn: 9.5897549	test: 13.8486744	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 14s
    1898:	learn: 9.5889681	test: 13.8483588	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 14s
    1899:	learn: 9.5875178	test: 13.8489922	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 14s
    1900:	learn: 9.5861771	test: 13.8493903	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 14s
    1901:	learn: 9.5850264	test: 13.8500432	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 14s
    1902:	learn: 9.5825205	test: 13.8484914	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 14s
    1903:	learn: 9.5800410	test: 13.8491631	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 14s
    1904:	learn: 9.5774711	test: 13.8475068	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 14s
    1905:	learn: 9.5761974	test: 13.8469168	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 13s
    1906:	learn: 9.5757568	test: 13.8466043	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 13s
    1907:	learn: 9.5750099	test: 13.8473664	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 13s
    1908:	learn: 9.5745182	test: 13.8468821	best: 13.8464148 (1888)	total: 1m 7s	remaining: 1m 13s
    1909:	learn: 9.5727441	test: 13.8463064	best: 13.8463064 (1909)	total: 1m 7s	remaining: 1m 13s
    1910:	learn: 9.5725395	test: 13.8464321	best: 13.8463064 (1909)	total: 1m 7s	remaining: 1m 13s
    1911:	learn: 9.5720461	test: 13.8463188	best: 13.8463064 (1909)	total: 1m 7s	remaining: 1m 13s
    1912:	learn: 9.5715116	test: 13.8457307	best: 13.8457307 (1912)	total: 1m 7s	remaining: 1m 13s
    1913:	learn: 9.5701790	test: 13.8453005	best: 13.8453005 (1913)	total: 1m 7s	remaining: 1m 13s
    1914:	learn: 9.5697642	test: 13.8446993	best: 13.8446993 (1914)	total: 1m 7s	remaining: 1m 13s
    1915:	learn: 9.5686435	test: 13.8459259	best: 13.8446993 (1914)	total: 1m 7s	remaining: 1m 13s
    1916:	learn: 9.5674363	test: 13.8459391	best: 13.8446993 (1914)	total: 1m 7s	remaining: 1m 13s
    1917:	learn: 9.5666111	test: 13.8443928	best: 13.8443928 (1917)	total: 1m 7s	remaining: 1m 13s
    1918:	learn: 9.5650221	test: 13.8456363	best: 13.8443928 (1917)	total: 1m 7s	remaining: 1m 13s
    1919:	learn: 9.5643912	test: 13.8452005	best: 13.8443928 (1917)	total: 1m 7s	remaining: 1m 13s
    1920:	learn: 9.5641954	test: 13.8444883	best: 13.8443928 (1917)	total: 1m 7s	remaining: 1m 13s
    1921:	learn: 9.5630534	test: 13.8428775	best: 13.8428775 (1921)	total: 1m 7s	remaining: 1m 13s
    1922:	learn: 9.5619777	test: 13.8421446	best: 13.8421446 (1922)	total: 1m 7s	remaining: 1m 13s
    1923:	learn: 9.5610440	test: 13.8431544	best: 13.8421446 (1922)	total: 1m 7s	remaining: 1m 13s
    1924:	learn: 9.5605453	test: 13.8432867	best: 13.8421446 (1922)	total: 1m 7s	remaining: 1m 13s
    1925:	learn: 9.5571991	test: 13.8414986	best: 13.8414986 (1925)	total: 1m 7s	remaining: 1m 13s
    1926:	learn: 9.5557156	test: 13.8411278	best: 13.8411278 (1926)	total: 1m 8s	remaining: 1m 13s
    1927:	learn: 9.5543567	test: 13.8428597	best: 13.8411278 (1926)	total: 1m 8s	remaining: 1m 13s
    1928:	learn: 9.5530350	test: 13.8422636	best: 13.8411278 (1926)	total: 1m 8s	remaining: 1m 13s
    1929:	learn: 9.5511923	test: 13.8405170	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 13s
    1930:	learn: 9.5501432	test: 13.8413220	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 13s
    1931:	learn: 9.5482093	test: 13.8427704	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1932:	learn: 9.5473084	test: 13.8441130	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1933:	learn: 9.5467524	test: 13.8438355	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1934:	learn: 9.5452218	test: 13.8446462	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1935:	learn: 9.5430916	test: 13.8449318	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1936:	learn: 9.5426779	test: 13.8443774	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1937:	learn: 9.5419062	test: 13.8441191	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1938:	learn: 9.5398697	test: 13.8427367	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1939:	learn: 9.5386358	test: 13.8426605	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1940:	learn: 9.5363767	test: 13.8433580	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1941:	learn: 9.5350259	test: 13.8448023	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1942:	learn: 9.5334212	test: 13.8436882	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1943:	learn: 9.5327617	test: 13.8432167	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1944:	learn: 9.5319077	test: 13.8428731	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1945:	learn: 9.5286860	test: 13.8409224	best: 13.8405170 (1929)	total: 1m 8s	remaining: 1m 12s
    1946:	learn: 9.5269770	test: 13.8393949	best: 13.8393949 (1946)	total: 1m 8s	remaining: 1m 12s
    1947:	learn: 9.5261452	test: 13.8406746	best: 13.8393949 (1946)	total: 1m 8s	remaining: 1m 12s
    1948:	learn: 9.5242821	test: 13.8391860	best: 13.8391860 (1948)	total: 1m 8s	remaining: 1m 12s
    1949:	learn: 9.5220909	test: 13.8389122	best: 13.8389122 (1949)	total: 1m 8s	remaining: 1m 12s
    1950:	learn: 9.5214419	test: 13.8379152	best: 13.8379152 (1950)	total: 1m 8s	remaining: 1m 12s
    1951:	learn: 9.5192707	test: 13.8354932	best: 13.8354932 (1951)	total: 1m 8s	remaining: 1m 12s
    1952:	learn: 9.5189551	test: 13.8352501	best: 13.8352501 (1952)	total: 1m 8s	remaining: 1m 12s
    1953:	learn: 9.5175587	test: 13.8326883	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 12s
    1954:	learn: 9.5162020	test: 13.8341783	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 12s
    1955:	learn: 9.5146149	test: 13.8335211	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 12s
    1956:	learn: 9.5130979	test: 13.8338009	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 12s
    1957:	learn: 9.5114197	test: 13.8344882	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 12s
    1958:	learn: 9.5108504	test: 13.8352348	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 12s
    1959:	learn: 9.5097043	test: 13.8348607	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 12s
    1960:	learn: 9.5077208	test: 13.8364554	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 12s
    1961:	learn: 9.5040164	test: 13.8332032	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1962:	learn: 9.5031096	test: 13.8344406	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1963:	learn: 9.5020430	test: 13.8332458	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1964:	learn: 9.5009923	test: 13.8331280	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1965:	learn: 9.4989418	test: 13.8345824	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1966:	learn: 9.4983555	test: 13.8355044	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1967:	learn: 9.4969596	test: 13.8358854	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1968:	learn: 9.4959647	test: 13.8364647	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1969:	learn: 9.4945359	test: 13.8381520	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1970:	learn: 9.4932885	test: 13.8386847	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1971:	learn: 9.4914597	test: 13.8383710	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1972:	learn: 9.4902417	test: 13.8383066	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1973:	learn: 9.4874142	test: 13.8390453	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1974:	learn: 9.4866144	test: 13.8381472	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1975:	learn: 9.4861832	test: 13.8385147	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1976:	learn: 9.4842690	test: 13.8363307	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1977:	learn: 9.4832486	test: 13.8362058	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1978:	learn: 9.4820753	test: 13.8351008	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1979:	learn: 9.4809712	test: 13.8344536	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1980:	learn: 9.4798720	test: 13.8339321	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1981:	learn: 9.4792738	test: 13.8336044	best: 13.8326883 (1953)	total: 1m 9s	remaining: 1m 11s
    1982:	learn: 9.4786382	test: 13.8343955	best: 13.8326883 (1953)	total: 1m 10s	remaining: 1m 11s
    1983:	learn: 9.4777926	test: 13.8352744	best: 13.8326883 (1953)	total: 1m 10s	remaining: 1m 11s
    1984:	learn: 9.4770675	test: 13.8348971	best: 13.8326883 (1953)	total: 1m 10s	remaining: 1m 11s
    1985:	learn: 9.4755459	test: 13.8355181	best: 13.8326883 (1953)	total: 1m 10s	remaining: 1m 11s
    1986:	learn: 9.4742986	test: 13.8351248	best: 13.8326883 (1953)	total: 1m 10s	remaining: 1m 11s
    1987:	learn: 9.4714866	test: 13.8330015	best: 13.8326883 (1953)	total: 1m 10s	remaining: 1m 11s
    1988:	learn: 9.4687831	test: 13.8321735	best: 13.8321735 (1988)	total: 1m 10s	remaining: 1m 10s
    1989:	learn: 9.4652541	test: 13.8344167	best: 13.8321735 (1988)	total: 1m 10s	remaining: 1m 10s
    1990:	learn: 9.4623151	test: 13.8306743	best: 13.8306743 (1990)	total: 1m 10s	remaining: 1m 10s
    1991:	learn: 9.4601889	test: 13.8308725	best: 13.8306743 (1990)	total: 1m 10s	remaining: 1m 10s
    1992:	learn: 9.4589404	test: 13.8306532	best: 13.8306532 (1992)	total: 1m 10s	remaining: 1m 10s
    1993:	learn: 9.4582244	test: 13.8312067	best: 13.8306532 (1992)	total: 1m 10s	remaining: 1m 10s
    1994:	learn: 9.4575439	test: 13.8308835	best: 13.8306532 (1992)	total: 1m 10s	remaining: 1m 10s
    1995:	learn: 9.4565615	test: 13.8315751	best: 13.8306532 (1992)	total: 1m 10s	remaining: 1m 10s
    1996:	learn: 9.4550449	test: 13.8303531	best: 13.8303531 (1996)	total: 1m 10s	remaining: 1m 10s
    1997:	learn: 9.4533803	test: 13.8302265	best: 13.8302265 (1997)	total: 1m 10s	remaining: 1m 10s
    1998:	learn: 9.4530709	test: 13.8305975	best: 13.8302265 (1997)	total: 1m 10s	remaining: 1m 10s
    1999:	learn: 9.4515748	test: 13.8306942	best: 13.8302265 (1997)	total: 1m 10s	remaining: 1m 10s
    2000:	learn: 9.4508778	test: 13.8295513	best: 13.8295513 (2000)	total: 1m 10s	remaining: 1m 10s
    2001:	learn: 9.4491014	test: 13.8300099	best: 13.8295513 (2000)	total: 1m 10s	remaining: 1m 10s
    2002:	learn: 9.4485145	test: 13.8297260	best: 13.8295513 (2000)	total: 1m 10s	remaining: 1m 10s
    2003:	learn: 9.4462285	test: 13.8292922	best: 13.8292922 (2003)	total: 1m 10s	remaining: 1m 10s
    2004:	learn: 9.4444684	test: 13.8300197	best: 13.8292922 (2003)	total: 1m 10s	remaining: 1m 10s
    2005:	learn: 9.4407571	test: 13.8293260	best: 13.8292922 (2003)	total: 1m 10s	remaining: 1m 10s
    2006:	learn: 9.4399319	test: 13.8287913	best: 13.8287913 (2006)	total: 1m 10s	remaining: 1m 10s
    2007:	learn: 9.4384624	test: 13.8287613	best: 13.8287613 (2007)	total: 1m 10s	remaining: 1m 10s
    2008:	learn: 9.4370093	test: 13.8289200	best: 13.8287613 (2007)	total: 1m 10s	remaining: 1m 10s
    2009:	learn: 9.4360867	test: 13.8292675	best: 13.8287613 (2007)	total: 1m 10s	remaining: 1m 10s
    2010:	learn: 9.4332762	test: 13.8301224	best: 13.8287613 (2007)	total: 1m 10s	remaining: 1m 10s
    2011:	learn: 9.4312587	test: 13.8297373	best: 13.8287613 (2007)	total: 1m 10s	remaining: 1m 10s
    2012:	learn: 9.4303833	test: 13.8291532	best: 13.8287613 (2007)	total: 1m 10s	remaining: 1m 10s
    2013:	learn: 9.4273394	test: 13.8280726	best: 13.8280726 (2013)	total: 1m 10s	remaining: 1m 9s
    2014:	learn: 9.4266696	test: 13.8282603	best: 13.8280726 (2013)	total: 1m 10s	remaining: 1m 9s
    2015:	learn: 9.4258883	test: 13.8269095	best: 13.8269095 (2015)	total: 1m 11s	remaining: 1m 9s
    2016:	learn: 9.4255569	test: 13.8264301	best: 13.8264301 (2016)	total: 1m 11s	remaining: 1m 9s
    2017:	learn: 9.4230303	test: 13.8241263	best: 13.8241263 (2017)	total: 1m 11s	remaining: 1m 9s
    2018:	learn: 9.4214495	test: 13.8218389	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2019:	learn: 9.4210299	test: 13.8218397	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2020:	learn: 9.4197420	test: 13.8238514	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2021:	learn: 9.4190165	test: 13.8244876	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2022:	learn: 9.4175975	test: 13.8247854	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2023:	learn: 9.4170380	test: 13.8251873	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2024:	learn: 9.4159614	test: 13.8244238	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2025:	learn: 9.4137653	test: 13.8233910	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2026:	learn: 9.4126404	test: 13.8239799	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2027:	learn: 9.4108351	test: 13.8223248	best: 13.8218389 (2018)	total: 1m 11s	remaining: 1m 9s
    2028:	learn: 9.4081148	test: 13.8216223	best: 13.8216223 (2028)	total: 1m 11s	remaining: 1m 9s
    2029:	learn: 9.4071866	test: 13.8213327	best: 13.8213327 (2029)	total: 1m 11s	remaining: 1m 9s
    2030:	learn: 9.4062319	test: 13.8208431	best: 13.8208431 (2030)	total: 1m 11s	remaining: 1m 9s
    2031:	learn: 9.4054094	test: 13.8219741	best: 13.8208431 (2030)	total: 1m 11s	remaining: 1m 9s
    2032:	learn: 9.4050164	test: 13.8217480	best: 13.8208431 (2030)	total: 1m 11s	remaining: 1m 9s
    2033:	learn: 9.4038899	test: 13.8217402	best: 13.8208431 (2030)	total: 1m 11s	remaining: 1m 9s
    2034:	learn: 9.4020899	test: 13.8210890	best: 13.8208431 (2030)	total: 1m 11s	remaining: 1m 9s
    2035:	learn: 9.4007031	test: 13.8216976	best: 13.8208431 (2030)	total: 1m 11s	remaining: 1m 9s
    2036:	learn: 9.3996437	test: 13.8224089	best: 13.8208431 (2030)	total: 1m 11s	remaining: 1m 9s
    2037:	learn: 9.3962974	test: 13.8198642	best: 13.8198642 (2037)	total: 1m 11s	remaining: 1m 9s
    2038:	learn: 9.3958104	test: 13.8200551	best: 13.8198642 (2037)	total: 1m 11s	remaining: 1m 8s
    2039:	learn: 9.3949367	test: 13.8198104	best: 13.8198104 (2039)	total: 1m 11s	remaining: 1m 8s
    2040:	learn: 9.3939921	test: 13.8190916	best: 13.8190916 (2040)	total: 1m 11s	remaining: 1m 8s
    2041:	learn: 9.3935529	test: 13.8193273	best: 13.8190916 (2040)	total: 1m 11s	remaining: 1m 8s
    2042:	learn: 9.3929269	test: 13.8196177	best: 13.8190916 (2040)	total: 1m 11s	remaining: 1m 8s
    2043:	learn: 9.3916333	test: 13.8192321	best: 13.8190916 (2040)	total: 1m 11s	remaining: 1m 8s
    2044:	learn: 9.3910220	test: 13.8191050	best: 13.8190916 (2040)	total: 1m 11s	remaining: 1m 8s
    2045:	learn: 9.3873176	test: 13.8179535	best: 13.8179535 (2045)	total: 1m 11s	remaining: 1m 8s
    2046:	learn: 9.3865036	test: 13.8174239	best: 13.8174239 (2046)	total: 1m 11s	remaining: 1m 8s
    2047:	learn: 9.3860319	test: 13.8180412	best: 13.8174239 (2046)	total: 1m 11s	remaining: 1m 8s
    2048:	learn: 9.3855359	test: 13.8180240	best: 13.8174239 (2046)	total: 1m 12s	remaining: 1m 8s
    2049:	learn: 9.3830426	test: 13.8161433	best: 13.8161433 (2049)	total: 1m 12s	remaining: 1m 8s
    2050:	learn: 9.3814118	test: 13.8164453	best: 13.8161433 (2049)	total: 1m 12s	remaining: 1m 8s
    2051:	learn: 9.3797468	test: 13.8168460	best: 13.8161433 (2049)	total: 1m 12s	remaining: 1m 8s
    2052:	learn: 9.3777618	test: 13.8154172	best: 13.8154172 (2052)	total: 1m 12s	remaining: 1m 8s
    2053:	learn: 9.3771293	test: 13.8160493	best: 13.8154172 (2052)	total: 1m 12s	remaining: 1m 8s
    2054:	learn: 9.3751582	test: 13.8146996	best: 13.8146996 (2054)	total: 1m 12s	remaining: 1m 8s
    2055:	learn: 9.3730181	test: 13.8151598	best: 13.8146996 (2054)	total: 1m 12s	remaining: 1m 8s
    2056:	learn: 9.3725038	test: 13.8156828	best: 13.8146996 (2054)	total: 1m 12s	remaining: 1m 8s
    2057:	learn: 9.3720809	test: 13.8149703	best: 13.8146996 (2054)	total: 1m 12s	remaining: 1m 8s
    2058:	learn: 9.3713047	test: 13.8142198	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 8s
    2059:	learn: 9.3693042	test: 13.8143785	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 8s
    2060:	learn: 9.3691343	test: 13.8158542	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 8s
    2061:	learn: 9.3686706	test: 13.8160613	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 8s
    2062:	learn: 9.3683546	test: 13.8165480	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 8s
    2063:	learn: 9.3677030	test: 13.8160561	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 8s
    2064:	learn: 9.3670554	test: 13.8173559	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2065:	learn: 9.3667849	test: 13.8182718	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2066:	learn: 9.3659278	test: 13.8185256	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2067:	learn: 9.3654440	test: 13.8173664	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2068:	learn: 9.3628744	test: 13.8171656	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2069:	learn: 9.3618750	test: 13.8165687	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2070:	learn: 9.3589295	test: 13.8158998	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2071:	learn: 9.3572909	test: 13.8170432	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2072:	learn: 9.3555950	test: 13.8166121	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2073:	learn: 9.3547313	test: 13.8163634	best: 13.8142198 (2058)	total: 1m 12s	remaining: 1m 7s
    2074:	learn: 9.3526840	test: 13.8139703	best: 13.8139703 (2074)	total: 1m 12s	remaining: 1m 7s
    2075:	learn: 9.3507463	test: 13.8153083	best: 13.8139703 (2074)	total: 1m 12s	remaining: 1m 7s
    2076:	learn: 9.3497928	test: 13.8153557	best: 13.8139703 (2074)	total: 1m 12s	remaining: 1m 7s
    2077:	learn: 9.3477442	test: 13.8150837	best: 13.8139703 (2074)	total: 1m 12s	remaining: 1m 7s
    2078:	learn: 9.3469704	test: 13.8145899	best: 13.8139703 (2074)	total: 1m 13s	remaining: 1m 7s
    2079:	learn: 9.3458827	test: 13.8155799	best: 13.8139703 (2074)	total: 1m 13s	remaining: 1m 7s
    2080:	learn: 9.3426289	test: 13.8130150	best: 13.8130150 (2080)	total: 1m 13s	remaining: 1m 7s
    2081:	learn: 9.3417518	test: 13.8135711	best: 13.8130150 (2080)	total: 1m 13s	remaining: 1m 7s
    2082:	learn: 9.3390017	test: 13.8128699	best: 13.8128699 (2082)	total: 1m 13s	remaining: 1m 7s
    2083:	learn: 9.3379342	test: 13.8138389	best: 13.8128699 (2082)	total: 1m 13s	remaining: 1m 7s
    2084:	learn: 9.3367231	test: 13.8139845	best: 13.8128699 (2082)	total: 1m 13s	remaining: 1m 7s
    2085:	learn: 9.3361588	test: 13.8132433	best: 13.8128699 (2082)	total: 1m 13s	remaining: 1m 7s
    2086:	learn: 9.3326850	test: 13.8099094	best: 13.8099094 (2086)	total: 1m 13s	remaining: 1m 7s
    2087:	learn: 9.3307971	test: 13.8095539	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 7s
    2088:	learn: 9.3296630	test: 13.8099697	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 7s
    2089:	learn: 9.3292099	test: 13.8106889	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 7s
    2090:	learn: 9.3266636	test: 13.8106246	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 7s
    2091:	learn: 9.3254204	test: 13.8096613	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2092:	learn: 9.3247762	test: 13.8107927	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2093:	learn: 9.3242518	test: 13.8110670	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2094:	learn: 9.3235444	test: 13.8103704	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2095:	learn: 9.3227157	test: 13.8113229	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2096:	learn: 9.3208279	test: 13.8106195	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2097:	learn: 9.3200981	test: 13.8108818	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2098:	learn: 9.3191361	test: 13.8109539	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2099:	learn: 9.3186547	test: 13.8113568	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2100:	learn: 9.3175166	test: 13.8125544	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2101:	learn: 9.3156299	test: 13.8137390	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2102:	learn: 9.3150172	test: 13.8125643	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2103:	learn: 9.3141804	test: 13.8125804	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2104:	learn: 9.3122081	test: 13.8113987	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2105:	learn: 9.3114205	test: 13.8128803	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2106:	learn: 9.3104623	test: 13.8117648	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2107:	learn: 9.3102579	test: 13.8106464	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2108:	learn: 9.3094360	test: 13.8111463	best: 13.8095539 (2087)	total: 1m 13s	remaining: 1m 6s
    2109:	learn: 9.3080709	test: 13.8115126	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2110:	learn: 9.3074244	test: 13.8114946	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2111:	learn: 9.3054264	test: 13.8115493	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2112:	learn: 9.3040089	test: 13.8127907	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2113:	learn: 9.3023323	test: 13.8147736	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2114:	learn: 9.3008401	test: 13.8141919	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2115:	learn: 9.2997940	test: 13.8137073	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2116:	learn: 9.2991562	test: 13.8141607	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2117:	learn: 9.2983797	test: 13.8153839	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2118:	learn: 9.2975600	test: 13.8137040	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 6s
    2119:	learn: 9.2946756	test: 13.8134494	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 5s
    2120:	learn: 9.2927182	test: 13.8111560	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 5s
    2121:	learn: 9.2911542	test: 13.8106857	best: 13.8095539 (2087)	total: 1m 14s	remaining: 1m 5s
    2122:	learn: 9.2894865	test: 13.8090231	best: 13.8090231 (2122)	total: 1m 14s	remaining: 1m 5s
    2123:	learn: 9.2865640	test: 13.8083551	best: 13.8083551 (2123)	total: 1m 14s	remaining: 1m 5s
    2124:	learn: 9.2862932	test: 13.8080492	best: 13.8080492 (2124)	total: 1m 14s	remaining: 1m 5s
    2125:	learn: 9.2854684	test: 13.8074253	best: 13.8074253 (2125)	total: 1m 14s	remaining: 1m 5s
    2126:	learn: 9.2834199	test: 13.8086749	best: 13.8074253 (2125)	total: 1m 14s	remaining: 1m 5s
    2127:	learn: 9.2818673	test: 13.8081992	best: 13.8074253 (2125)	total: 1m 14s	remaining: 1m 5s
    2128:	learn: 9.2799173	test: 13.8072641	best: 13.8072641 (2128)	total: 1m 14s	remaining: 1m 5s
    2129:	learn: 9.2783132	test: 13.8070107	best: 13.8070107 (2129)	total: 1m 14s	remaining: 1m 5s
    2130:	learn: 9.2768723	test: 13.8066057	best: 13.8066057 (2130)	total: 1m 14s	remaining: 1m 5s
    2131:	learn: 9.2757470	test: 13.8053028	best: 13.8053028 (2131)	total: 1m 14s	remaining: 1m 5s
    2132:	learn: 9.2738921	test: 13.8042530	best: 13.8042530 (2132)	total: 1m 14s	remaining: 1m 5s
    2133:	learn: 9.2722929	test: 13.8050570	best: 13.8042530 (2132)	total: 1m 14s	remaining: 1m 5s
    2134:	learn: 9.2719364	test: 13.8029520	best: 13.8029520 (2134)	total: 1m 14s	remaining: 1m 5s
    2135:	learn: 9.2695302	test: 13.8034871	best: 13.8029520 (2134)	total: 1m 14s	remaining: 1m 5s
    2136:	learn: 9.2694316	test: 13.8027716	best: 13.8027716 (2136)	total: 1m 14s	remaining: 1m 5s
    2137:	learn: 9.2686232	test: 13.8033964	best: 13.8027716 (2136)	total: 1m 15s	remaining: 1m 5s
    2138:	learn: 9.2672624	test: 13.8029430	best: 13.8027716 (2136)	total: 1m 15s	remaining: 1m 5s
    2139:	learn: 9.2665558	test: 13.8038092	best: 13.8027716 (2136)	total: 1m 15s	remaining: 1m 5s
    2140:	learn: 9.2650947	test: 13.8049262	best: 13.8027716 (2136)	total: 1m 15s	remaining: 1m 5s
    2141:	learn: 9.2634444	test: 13.8043477	best: 13.8027716 (2136)	total: 1m 15s	remaining: 1m 5s
    2142:	learn: 9.2617621	test: 13.8036875	best: 13.8027716 (2136)	total: 1m 15s	remaining: 1m 5s
    2143:	learn: 9.2609879	test: 13.8030715	best: 13.8027716 (2136)	total: 1m 15s	remaining: 1m 5s
    2144:	learn: 9.2596142	test: 13.8017003	best: 13.8017003 (2144)	total: 1m 15s	remaining: 1m 5s
    2145:	learn: 9.2583677	test: 13.8018544	best: 13.8017003 (2144)	total: 1m 15s	remaining: 1m 5s
    2146:	learn: 9.2577967	test: 13.8016037	best: 13.8016037 (2146)	total: 1m 15s	remaining: 1m 4s
    2147:	learn: 9.2570816	test: 13.8022792	best: 13.8016037 (2146)	total: 1m 15s	remaining: 1m 4s
    2148:	learn: 9.2569647	test: 13.8027621	best: 13.8016037 (2146)	total: 1m 15s	remaining: 1m 4s
    2149:	learn: 9.2550825	test: 13.8031575	best: 13.8016037 (2146)	total: 1m 15s	remaining: 1m 4s
    2150:	learn: 9.2545658	test: 13.8020328	best: 13.8016037 (2146)	total: 1m 15s	remaining: 1m 4s
    2151:	learn: 9.2542536	test: 13.8028826	best: 13.8016037 (2146)	total: 1m 15s	remaining: 1m 4s
    2152:	learn: 9.2533185	test: 13.8014410	best: 13.8014410 (2152)	total: 1m 15s	remaining: 1m 4s
    2153:	learn: 9.2521768	test: 13.8013153	best: 13.8013153 (2153)	total: 1m 15s	remaining: 1m 4s
    2154:	learn: 9.2506466	test: 13.8002900	best: 13.8002900 (2154)	total: 1m 15s	remaining: 1m 4s
    2155:	learn: 9.2495835	test: 13.8000274	best: 13.8000274 (2155)	total: 1m 15s	remaining: 1m 4s
    2156:	learn: 9.2488436	test: 13.7996349	best: 13.7996349 (2156)	total: 1m 15s	remaining: 1m 4s
    2157:	learn: 9.2469722	test: 13.7988158	best: 13.7988158 (2157)	total: 1m 15s	remaining: 1m 4s
    2158:	learn: 9.2453168	test: 13.7973715	best: 13.7973715 (2158)	total: 1m 15s	remaining: 1m 4s
    2159:	learn: 9.2434898	test: 13.7947733	best: 13.7947733 (2159)	total: 1m 15s	remaining: 1m 4s
    2160:	learn: 9.2422644	test: 13.7959603	best: 13.7947733 (2159)	total: 1m 15s	remaining: 1m 4s
    2161:	learn: 9.2415576	test: 13.7953790	best: 13.7947733 (2159)	total: 1m 15s	remaining: 1m 4s
    2162:	learn: 9.2409344	test: 13.7964140	best: 13.7947733 (2159)	total: 1m 15s	remaining: 1m 4s
    2163:	learn: 9.2394913	test: 13.7962442	best: 13.7947733 (2159)	total: 1m 15s	remaining: 1m 4s
    2164:	learn: 9.2379802	test: 13.7955746	best: 13.7947733 (2159)	total: 1m 15s	remaining: 1m 4s
    2165:	learn: 9.2364283	test: 13.7948296	best: 13.7947733 (2159)	total: 1m 15s	remaining: 1m 4s
    2166:	learn: 9.2361571	test: 13.7950071	best: 13.7947733 (2159)	total: 1m 15s	remaining: 1m 4s
    2167:	learn: 9.2344652	test: 13.7932939	best: 13.7932939 (2167)	total: 1m 16s	remaining: 1m 4s
    2168:	learn: 9.2334868	test: 13.7924278	best: 13.7924278 (2168)	total: 1m 16s	remaining: 1m 4s
    2169:	learn: 9.2329624	test: 13.7915680	best: 13.7915680 (2169)	total: 1m 16s	remaining: 1m 4s
    2170:	learn: 9.2299360	test: 13.7915357	best: 13.7915357 (2170)	total: 1m 16s	remaining: 1m 4s
    2171:	learn: 9.2281284	test: 13.7904485	best: 13.7904485 (2171)	total: 1m 16s	remaining: 1m 4s
    2172:	learn: 9.2276434	test: 13.7892794	best: 13.7892794 (2172)	total: 1m 16s	remaining: 1m 4s
    2173:	learn: 9.2253222	test: 13.7876177	best: 13.7876177 (2173)	total: 1m 16s	remaining: 1m 4s
    2174:	learn: 9.2247194	test: 13.7873093	best: 13.7873093 (2174)	total: 1m 16s	remaining: 1m 3s
    2175:	learn: 9.2245120	test: 13.7873424	best: 13.7873093 (2174)	total: 1m 16s	remaining: 1m 3s
    2176:	learn: 9.2237561	test: 13.7876911	best: 13.7873093 (2174)	total: 1m 16s	remaining: 1m 3s
    2177:	learn: 9.2228127	test: 13.7875034	best: 13.7873093 (2174)	total: 1m 16s	remaining: 1m 3s
    2178:	learn: 9.2201571	test: 13.7875536	best: 13.7873093 (2174)	total: 1m 16s	remaining: 1m 3s
    2179:	learn: 9.2190977	test: 13.7875116	best: 13.7873093 (2174)	total: 1m 16s	remaining: 1m 3s
    2180:	learn: 9.2181688	test: 13.7869837	best: 13.7869837 (2180)	total: 1m 16s	remaining: 1m 3s
    2181:	learn: 9.2172506	test: 13.7883246	best: 13.7869837 (2180)	total: 1m 16s	remaining: 1m 3s
    2182:	learn: 9.2162988	test: 13.7888439	best: 13.7869837 (2180)	total: 1m 16s	remaining: 1m 3s
    2183:	learn: 9.2142852	test: 13.7865284	best: 13.7865284 (2183)	total: 1m 16s	remaining: 1m 3s
    2184:	learn: 9.2132051	test: 13.7866861	best: 13.7865284 (2183)	total: 1m 16s	remaining: 1m 3s
    2185:	learn: 9.2125272	test: 13.7870869	best: 13.7865284 (2183)	total: 1m 16s	remaining: 1m 3s
    2186:	learn: 9.2114456	test: 13.7863413	best: 13.7863413 (2186)	total: 1m 16s	remaining: 1m 3s
    2187:	learn: 9.2091764	test: 13.7859898	best: 13.7859898 (2187)	total: 1m 16s	remaining: 1m 3s
    2188:	learn: 9.2075385	test: 13.7871568	best: 13.7859898 (2187)	total: 1m 16s	remaining: 1m 3s
    2189:	learn: 9.2042447	test: 13.7862700	best: 13.7859898 (2187)	total: 1m 16s	remaining: 1m 3s
    2190:	learn: 9.2022083	test: 13.7863260	best: 13.7859898 (2187)	total: 1m 16s	remaining: 1m 3s
    2191:	learn: 9.2013044	test: 13.7852751	best: 13.7852751 (2191)	total: 1m 16s	remaining: 1m 3s
    2192:	learn: 9.2005662	test: 13.7853516	best: 13.7852751 (2191)	total: 1m 16s	remaining: 1m 3s
    2193:	learn: 9.1983876	test: 13.7856594	best: 13.7852751 (2191)	total: 1m 16s	remaining: 1m 3s
    2194:	learn: 9.1973778	test: 13.7868012	best: 13.7852751 (2191)	total: 1m 16s	remaining: 1m 3s
    2195:	learn: 9.1966747	test: 13.7863263	best: 13.7852751 (2191)	total: 1m 16s	remaining: 1m 3s
    2196:	learn: 9.1956653	test: 13.7878557	best: 13.7852751 (2191)	total: 1m 16s	remaining: 1m 3s
    2197:	learn: 9.1939537	test: 13.7866423	best: 13.7852751 (2191)	total: 1m 16s	remaining: 1m 3s
    2198:	learn: 9.1920546	test: 13.7858482	best: 13.7852751 (2191)	total: 1m 17s	remaining: 1m 3s
    2199:	learn: 9.1911850	test: 13.7867014	best: 13.7852751 (2191)	total: 1m 17s	remaining: 1m 3s
    2200:	learn: 9.1902645	test: 13.7867610	best: 13.7852751 (2191)	total: 1m 17s	remaining: 1m 3s
    2201:	learn: 9.1894470	test: 13.7852297	best: 13.7852297 (2201)	total: 1m 17s	remaining: 1m 2s
    2202:	learn: 9.1884543	test: 13.7860210	best: 13.7852297 (2201)	total: 1m 17s	remaining: 1m 2s
    2203:	learn: 9.1876579	test: 13.7875374	best: 13.7852297 (2201)	total: 1m 17s	remaining: 1m 2s
    2204:	learn: 9.1863356	test: 13.7869443	best: 13.7852297 (2201)	total: 1m 17s	remaining: 1m 2s
    2205:	learn: 9.1851139	test: 13.7862066	best: 13.7852297 (2201)	total: 1m 17s	remaining: 1m 2s
    2206:	learn: 9.1811825	test: 13.7829346	best: 13.7829346 (2206)	total: 1m 17s	remaining: 1m 2s
    2207:	learn: 9.1789509	test: 13.7815748	best: 13.7815748 (2207)	total: 1m 17s	remaining: 1m 2s
    2208:	learn: 9.1782258	test: 13.7819496	best: 13.7815748 (2207)	total: 1m 17s	remaining: 1m 2s
    2209:	learn: 9.1774047	test: 13.7824483	best: 13.7815748 (2207)	total: 1m 17s	remaining: 1m 2s
    2210:	learn: 9.1764954	test: 13.7814777	best: 13.7814777 (2210)	total: 1m 17s	remaining: 1m 2s
    2211:	learn: 9.1749203	test: 13.7801952	best: 13.7801952 (2211)	total: 1m 17s	remaining: 1m 2s
    2212:	learn: 9.1739430	test: 13.7791841	best: 13.7791841 (2212)	total: 1m 17s	remaining: 1m 2s
    2213:	learn: 9.1715774	test: 13.7790204	best: 13.7790204 (2213)	total: 1m 17s	remaining: 1m 2s
    2214:	learn: 9.1710212	test: 13.7794562	best: 13.7790204 (2213)	total: 1m 17s	remaining: 1m 2s
    2215:	learn: 9.1698167	test: 13.7786049	best: 13.7786049 (2215)	total: 1m 17s	remaining: 1m 2s
    2216:	learn: 9.1687695	test: 13.7777850	best: 13.7777850 (2216)	total: 1m 17s	remaining: 1m 2s
    2217:	learn: 9.1652695	test: 13.7779372	best: 13.7777850 (2216)	total: 1m 17s	remaining: 1m 2s
    2218:	learn: 9.1643209	test: 13.7781371	best: 13.7777850 (2216)	total: 1m 17s	remaining: 1m 2s
    2219:	learn: 9.1634687	test: 13.7784106	best: 13.7777850 (2216)	total: 1m 17s	remaining: 1m 2s
    2220:	learn: 9.1611674	test: 13.7758780	best: 13.7758780 (2220)	total: 1m 17s	remaining: 1m 2s
    2221:	learn: 9.1601351	test: 13.7770997	best: 13.7758780 (2220)	total: 1m 17s	remaining: 1m 2s
    2222:	learn: 9.1593161	test: 13.7763673	best: 13.7758780 (2220)	total: 1m 17s	remaining: 1m 2s
    2223:	learn: 9.1585118	test: 13.7779409	best: 13.7758780 (2220)	total: 1m 17s	remaining: 1m 2s
    2224:	learn: 9.1567830	test: 13.7773506	best: 13.7758780 (2220)	total: 1m 17s	remaining: 1m 2s
    2225:	learn: 9.1554165	test: 13.7768832	best: 13.7758780 (2220)	total: 1m 17s	remaining: 1m 2s
    2226:	learn: 9.1525449	test: 13.7766943	best: 13.7758780 (2220)	total: 1m 17s	remaining: 1m 2s
    2227:	learn: 9.1517809	test: 13.7772742	best: 13.7758780 (2220)	total: 1m 17s	remaining: 1m 2s
    2228:	learn: 9.1513124	test: 13.7773005	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2229:	learn: 9.1501003	test: 13.7773515	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2230:	learn: 9.1488408	test: 13.7772992	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2231:	learn: 9.1485861	test: 13.7778444	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2232:	learn: 9.1472019	test: 13.7775431	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2233:	learn: 9.1464084	test: 13.7773222	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2234:	learn: 9.1453417	test: 13.7774995	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2235:	learn: 9.1444313	test: 13.7783243	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2236:	learn: 9.1427856	test: 13.7800304	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2237:	learn: 9.1400245	test: 13.7776793	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2238:	learn: 9.1393941	test: 13.7775418	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2239:	learn: 9.1390183	test: 13.7774997	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2240:	learn: 9.1385078	test: 13.7782063	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2241:	learn: 9.1374675	test: 13.7784033	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2242:	learn: 9.1366503	test: 13.7770576	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2243:	learn: 9.1359747	test: 13.7782250	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2244:	learn: 9.1348580	test: 13.7786121	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2245:	learn: 9.1340595	test: 13.7792101	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2246:	learn: 9.1314734	test: 13.7816203	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2247:	learn: 9.1308256	test: 13.7814447	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2248:	learn: 9.1289535	test: 13.7819206	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2249:	learn: 9.1269094	test: 13.7793794	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2250:	learn: 9.1261614	test: 13.7804215	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2251:	learn: 9.1257453	test: 13.7800469	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2252:	learn: 9.1251794	test: 13.7808615	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2253:	learn: 9.1240007	test: 13.7805459	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2254:	learn: 9.1233729	test: 13.7817778	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2255:	learn: 9.1222182	test: 13.7826930	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m 1s
    2256:	learn: 9.1210820	test: 13.7834683	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m
    2257:	learn: 9.1209137	test: 13.7834914	best: 13.7758780 (2220)	total: 1m 18s	remaining: 1m
    2258:	learn: 9.1204535	test: 13.7826408	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2259:	learn: 9.1195245	test: 13.7824598	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2260:	learn: 9.1192708	test: 13.7830358	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2261:	learn: 9.1187124	test: 13.7828961	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2262:	learn: 9.1162926	test: 13.7800106	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2263:	learn: 9.1146887	test: 13.7791285	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2264:	learn: 9.1144546	test: 13.7791867	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2265:	learn: 9.1141252	test: 13.7791341	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2266:	learn: 9.1132606	test: 13.7801093	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2267:	learn: 9.1120171	test: 13.7781468	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2268:	learn: 9.1105891	test: 13.7781857	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2269:	learn: 9.1097812	test: 13.7779585	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2270:	learn: 9.1090135	test: 13.7802786	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2271:	learn: 9.1076335	test: 13.7831384	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2272:	learn: 9.1072221	test: 13.7826711	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2273:	learn: 9.1057276	test: 13.7828250	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2274:	learn: 9.1048708	test: 13.7827721	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2275:	learn: 9.1043771	test: 13.7828739	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2276:	learn: 9.1029115	test: 13.7835032	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2277:	learn: 9.1006262	test: 13.7837221	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2278:	learn: 9.1000938	test: 13.7838761	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2279:	learn: 9.0987290	test: 13.7835361	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2280:	learn: 9.0974568	test: 13.7837299	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2281:	learn: 9.0963282	test: 13.7812211	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2282:	learn: 9.0955203	test: 13.7812984	best: 13.7758780 (2220)	total: 1m 19s	remaining: 1m
    2283:	learn: 9.0932813	test: 13.7800733	best: 13.7758780 (2220)	total: 1m 19s	remaining: 60s
    2284:	learn: 9.0933731	test: 13.7799751	best: 13.7758780 (2220)	total: 1m 19s	remaining: 60s
    2285:	learn: 9.0925946	test: 13.7796475	best: 13.7758780 (2220)	total: 1m 19s	remaining: 59.9s
    2286:	learn: 9.0908473	test: 13.7774217	best: 13.7758780 (2220)	total: 1m 19s	remaining: 59.9s
    2287:	learn: 9.0893672	test: 13.7761112	best: 13.7758780 (2220)	total: 1m 19s	remaining: 59.9s
    2288:	learn: 9.0887995	test: 13.7754706	best: 13.7754706 (2288)	total: 1m 20s	remaining: 59.8s
    2289:	learn: 9.0882965	test: 13.7751018	best: 13.7751018 (2289)	total: 1m 20s	remaining: 59.8s
    2290:	learn: 9.0875050	test: 13.7754672	best: 13.7751018 (2289)	total: 1m 20s	remaining: 59.7s
    2291:	learn: 9.0866283	test: 13.7744733	best: 13.7744733 (2291)	total: 1m 20s	remaining: 59.7s
    2292:	learn: 9.0849979	test: 13.7753702	best: 13.7744733 (2291)	total: 1m 20s	remaining: 59.7s
    2293:	learn: 9.0833483	test: 13.7750380	best: 13.7744733 (2291)	total: 1m 20s	remaining: 59.6s
    2294:	learn: 9.0818745	test: 13.7741725	best: 13.7741725 (2294)	total: 1m 20s	remaining: 59.6s
    2295:	learn: 9.0809381	test: 13.7730600	best: 13.7730600 (2295)	total: 1m 20s	remaining: 59.6s
    2296:	learn: 9.0795707	test: 13.7729642	best: 13.7729642 (2296)	total: 1m 20s	remaining: 59.5s
    2297:	learn: 9.0789503	test: 13.7728131	best: 13.7728131 (2297)	total: 1m 20s	remaining: 59.5s
    2298:	learn: 9.0775971	test: 13.7733232	best: 13.7728131 (2297)	total: 1m 20s	remaining: 59.4s
    2299:	learn: 9.0770414	test: 13.7737809	best: 13.7728131 (2297)	total: 1m 20s	remaining: 59.4s
    2300:	learn: 9.0756684	test: 13.7707546	best: 13.7707546 (2300)	total: 1m 20s	remaining: 59.4s
    2301:	learn: 9.0748163	test: 13.7708025	best: 13.7707546 (2300)	total: 1m 20s	remaining: 59.3s
    2302:	learn: 9.0732942	test: 13.7710400	best: 13.7707546 (2300)	total: 1m 20s	remaining: 59.3s
    2303:	learn: 9.0718245	test: 13.7699257	best: 13.7699257 (2303)	total: 1m 20s	remaining: 59.2s
    2304:	learn: 9.0714823	test: 13.7699559	best: 13.7699257 (2303)	total: 1m 20s	remaining: 59.2s
    2305:	learn: 9.0707621	test: 13.7687271	best: 13.7687271 (2305)	total: 1m 20s	remaining: 59.2s
    2306:	learn: 9.0703932	test: 13.7689972	best: 13.7687271 (2305)	total: 1m 20s	remaining: 59.1s
    2307:	learn: 9.0681181	test: 13.7682708	best: 13.7682708 (2307)	total: 1m 20s	remaining: 59.1s
    2308:	learn: 9.0670239	test: 13.7681577	best: 13.7681577 (2308)	total: 1m 20s	remaining: 59.1s
    2309:	learn: 9.0655492	test: 13.7697458	best: 13.7681577 (2308)	total: 1m 20s	remaining: 59s
    2310:	learn: 9.0634298	test: 13.7701065	best: 13.7681577 (2308)	total: 1m 20s	remaining: 59s
    2311:	learn: 9.0630547	test: 13.7693553	best: 13.7681577 (2308)	total: 1m 20s	remaining: 58.9s
    2312:	learn: 9.0613686	test: 13.7699201	best: 13.7681577 (2308)	total: 1m 20s	remaining: 58.9s
    2313:	learn: 9.0607745	test: 13.7701164	best: 13.7681577 (2308)	total: 1m 20s	remaining: 58.9s
    2314:	learn: 9.0597166	test: 13.7698800	best: 13.7681577 (2308)	total: 1m 20s	remaining: 58.8s
    2315:	learn: 9.0578867	test: 13.7710913	best: 13.7681577 (2308)	total: 1m 20s	remaining: 58.8s
    2316:	learn: 9.0566323	test: 13.7722224	best: 13.7681577 (2308)	total: 1m 20s	remaining: 58.8s
    2317:	learn: 9.0553439	test: 13.7713638	best: 13.7681577 (2308)	total: 1m 20s	remaining: 58.7s
    2318:	learn: 9.0534818	test: 13.7707877	best: 13.7681577 (2308)	total: 1m 20s	remaining: 58.7s
    2319:	learn: 9.0529877	test: 13.7724590	best: 13.7681577 (2308)	total: 1m 21s	remaining: 58.7s
    2320:	learn: 9.0522362	test: 13.7710436	best: 13.7681577 (2308)	total: 1m 21s	remaining: 58.6s
    2321:	learn: 9.0513545	test: 13.7707970	best: 13.7681577 (2308)	total: 1m 21s	remaining: 58.6s
    2322:	learn: 9.0504870	test: 13.7706925	best: 13.7681577 (2308)	total: 1m 21s	remaining: 58.6s
    2323:	learn: 9.0499907	test: 13.7705312	best: 13.7681577 (2308)	total: 1m 21s	remaining: 58.5s
    2324:	learn: 9.0474951	test: 13.7689608	best: 13.7681577 (2308)	total: 1m 21s	remaining: 58.5s
    2325:	learn: 9.0471128	test: 13.7689769	best: 13.7681577 (2308)	total: 1m 21s	remaining: 58.4s
    2326:	learn: 9.0452979	test: 13.7683691	best: 13.7681577 (2308)	total: 1m 21s	remaining: 58.4s
    2327:	learn: 9.0432915	test: 13.7681780	best: 13.7681577 (2308)	total: 1m 21s	remaining: 58.4s
    2328:	learn: 9.0428598	test: 13.7676510	best: 13.7676510 (2328)	total: 1m 21s	remaining: 58.3s
    2329:	learn: 9.0409382	test: 13.7671821	best: 13.7671821 (2329)	total: 1m 21s	remaining: 58.3s
    2330:	learn: 9.0404535	test: 13.7668883	best: 13.7668883 (2330)	total: 1m 21s	remaining: 58.2s
    2331:	learn: 9.0387369	test: 13.7659051	best: 13.7659051 (2331)	total: 1m 21s	remaining: 58.2s
    2332:	learn: 9.0379162	test: 13.7665478	best: 13.7659051 (2331)	total: 1m 21s	remaining: 58.2s
    2333:	learn: 9.0370974	test: 13.7667382	best: 13.7659051 (2331)	total: 1m 21s	remaining: 58.1s
    2334:	learn: 9.0363584	test: 13.7657842	best: 13.7657842 (2334)	total: 1m 21s	remaining: 58.1s
    2335:	learn: 9.0338033	test: 13.7662422	best: 13.7657842 (2334)	total: 1m 21s	remaining: 58.1s
    2336:	learn: 9.0322548	test: 13.7650326	best: 13.7650326 (2336)	total: 1m 21s	remaining: 58s
    2337:	learn: 9.0312799	test: 13.7649305	best: 13.7649305 (2337)	total: 1m 21s	remaining: 58s
    2338:	learn: 9.0296448	test: 13.7652645	best: 13.7649305 (2337)	total: 1m 21s	remaining: 58s
    2339:	learn: 9.0286371	test: 13.7662389	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.9s
    2340:	learn: 9.0267718	test: 13.7662451	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.9s
    2341:	learn: 9.0263189	test: 13.7657598	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.8s
    2342:	learn: 9.0253633	test: 13.7654933	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.8s
    2343:	learn: 9.0244341	test: 13.7670536	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.8s
    2344:	learn: 9.0227817	test: 13.7662688	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.7s
    2345:	learn: 9.0218482	test: 13.7670499	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.7s
    2346:	learn: 9.0209793	test: 13.7685359	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.7s
    2347:	learn: 9.0203064	test: 13.7687865	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.6s
    2348:	learn: 9.0184244	test: 13.7702289	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.6s
    2349:	learn: 9.0183405	test: 13.7708528	best: 13.7649305 (2337)	total: 1m 21s	remaining: 57.6s
    2350:	learn: 9.0181825	test: 13.7702145	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.5s
    2351:	learn: 9.0173983	test: 13.7714237	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.5s
    2352:	learn: 9.0163836	test: 13.7711956	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.5s
    2353:	learn: 9.0148295	test: 13.7720904	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.4s
    2354:	learn: 9.0136567	test: 13.7721328	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.4s
    2355:	learn: 9.0134077	test: 13.7717197	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.4s
    2356:	learn: 9.0119088	test: 13.7701176	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.3s
    2357:	learn: 9.0111728	test: 13.7692560	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.3s
    2358:	learn: 9.0089785	test: 13.7683906	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.3s
    2359:	learn: 9.0077729	test: 13.7674060	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.2s
    2360:	learn: 9.0063378	test: 13.7679447	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.2s
    2361:	learn: 9.0054217	test: 13.7678798	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.1s
    2362:	learn: 9.0051341	test: 13.7679492	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.1s
    2363:	learn: 9.0034350	test: 13.7671506	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57.1s
    2364:	learn: 9.0025387	test: 13.7688040	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57s
    2365:	learn: 9.0022362	test: 13.7680127	best: 13.7649305 (2337)	total: 1m 22s	remaining: 57s
    2366:	learn: 9.0001614	test: 13.7687993	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.9s
    2367:	learn: 8.9964454	test: 13.7687927	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.9s
    2368:	learn: 8.9951341	test: 13.7683619	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.9s
    2369:	learn: 8.9946331	test: 13.7689158	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.8s
    2370:	learn: 8.9936444	test: 13.7687497	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.8s
    2371:	learn: 8.9910541	test: 13.7693094	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.8s
    2372:	learn: 8.9902875	test: 13.7698810	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.7s
    2373:	learn: 8.9888553	test: 13.7694235	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.7s
    2374:	learn: 8.9871258	test: 13.7695298	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.6s
    2375:	learn: 8.9858860	test: 13.7701302	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.6s
    2376:	learn: 8.9856289	test: 13.7708343	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.6s
    2377:	learn: 8.9849182	test: 13.7695649	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.5s
    2378:	learn: 8.9845775	test: 13.7705132	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.5s
    2379:	learn: 8.9826217	test: 13.7704520	best: 13.7649305 (2337)	total: 1m 22s	remaining: 56.5s
    2380:	learn: 8.9818309	test: 13.7702956	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.4s
    2381:	learn: 8.9812377	test: 13.7709777	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.4s
    2382:	learn: 8.9810386	test: 13.7704791	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.4s
    2383:	learn: 8.9785876	test: 13.7681654	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.3s
    2384:	learn: 8.9772475	test: 13.7687370	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.3s
    2385:	learn: 8.9752156	test: 13.7690443	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.3s
    2386:	learn: 8.9744976	test: 13.7705045	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.2s
    2387:	learn: 8.9731807	test: 13.7711677	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.2s
    2388:	learn: 8.9727573	test: 13.7706979	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.1s
    2389:	learn: 8.9725181	test: 13.7693544	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.1s
    2390:	learn: 8.9713056	test: 13.7693266	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56.1s
    2391:	learn: 8.9701377	test: 13.7696042	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56s
    2392:	learn: 8.9688579	test: 13.7686249	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56s
    2393:	learn: 8.9685844	test: 13.7683448	best: 13.7649305 (2337)	total: 1m 23s	remaining: 56s
    2394:	learn: 8.9672157	test: 13.7680989	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.9s
    2395:	learn: 8.9670861	test: 13.7676515	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.9s
    2396:	learn: 8.9666552	test: 13.7678164	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.9s
    2397:	learn: 8.9651471	test: 13.7677599	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.8s
    2398:	learn: 8.9638161	test: 13.7684946	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.8s
    2399:	learn: 8.9631271	test: 13.7690797	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.7s
    2400:	learn: 8.9623515	test: 13.7691444	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.7s
    2401:	learn: 8.9595068	test: 13.7700438	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.7s
    2402:	learn: 8.9592190	test: 13.7695096	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.6s
    2403:	learn: 8.9555251	test: 13.7673926	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.6s
    2404:	learn: 8.9547326	test: 13.7669400	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.6s
    2405:	learn: 8.9535187	test: 13.7676459	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.5s
    2406:	learn: 8.9529559	test: 13.7672720	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.5s
    2407:	learn: 8.9518677	test: 13.7675593	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.5s
    2408:	learn: 8.9511442	test: 13.7677930	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.4s
    2409:	learn: 8.9503077	test: 13.7682065	best: 13.7649305 (2337)	total: 1m 23s	remaining: 55.4s
    2410:	learn: 8.9488773	test: 13.7677008	best: 13.7649305 (2337)	total: 1m 24s	remaining: 55.4s
    2411:	learn: 8.9478307	test: 13.7668849	best: 13.7649305 (2337)	total: 1m 24s	remaining: 55.3s
    2412:	learn: 8.9458019	test: 13.7644034	best: 13.7644034 (2412)	total: 1m 24s	remaining: 55.3s
    2413:	learn: 8.9449141	test: 13.7641742	best: 13.7641742 (2413)	total: 1m 24s	remaining: 55.2s
    2414:	learn: 8.9437252	test: 13.7643904	best: 13.7641742 (2413)	total: 1m 24s	remaining: 55.2s
    2415:	learn: 8.9432649	test: 13.7645310	best: 13.7641742 (2413)	total: 1m 24s	remaining: 55.2s
    2416:	learn: 8.9426064	test: 13.7636582	best: 13.7636582 (2416)	total: 1m 24s	remaining: 55.1s
    2417:	learn: 8.9418412	test: 13.7628864	best: 13.7628864 (2417)	total: 1m 24s	remaining: 55.1s
    2418:	learn: 8.9412901	test: 13.7632841	best: 13.7628864 (2417)	total: 1m 24s	remaining: 55s
    2419:	learn: 8.9392016	test: 13.7612711	best: 13.7612711 (2419)	total: 1m 24s	remaining: 55s
    2420:	learn: 8.9374576	test: 13.7601252	best: 13.7601252 (2420)	total: 1m 24s	remaining: 55s
    2421:	learn: 8.9363476	test: 13.7604688	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.9s
    2422:	learn: 8.9343671	test: 13.7621931	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.9s
    2423:	learn: 8.9336205	test: 13.7622270	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.9s
    2424:	learn: 8.9326530	test: 13.7625292	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.8s
    2425:	learn: 8.9318691	test: 13.7633588	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.8s
    2426:	learn: 8.9307797	test: 13.7622507	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.7s
    2427:	learn: 8.9276277	test: 13.7622230	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.7s
    2428:	learn: 8.9269162	test: 13.7614809	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.7s
    2429:	learn: 8.9263567	test: 13.7619455	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.6s
    2430:	learn: 8.9259378	test: 13.7614867	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.6s
    2431:	learn: 8.9242651	test: 13.7622349	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.6s
    2432:	learn: 8.9234021	test: 13.7622074	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.5s
    2433:	learn: 8.9227672	test: 13.7626332	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.5s
    2434:	learn: 8.9221082	test: 13.7620083	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.5s
    2435:	learn: 8.9200883	test: 13.7634845	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.4s
    2436:	learn: 8.9191334	test: 13.7624056	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.4s
    2437:	learn: 8.9181490	test: 13.7622849	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.3s
    2438:	learn: 8.9170896	test: 13.7628704	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.3s
    2439:	learn: 8.9163285	test: 13.7616320	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.3s
    2440:	learn: 8.9155250	test: 13.7610202	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.2s
    2441:	learn: 8.9152183	test: 13.7617186	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.2s
    2442:	learn: 8.9135973	test: 13.7615864	best: 13.7601252 (2420)	total: 1m 24s	remaining: 54.2s
    2443:	learn: 8.9123271	test: 13.7600244	best: 13.7600244 (2443)	total: 1m 25s	remaining: 54.1s
    2444:	learn: 8.9109054	test: 13.7600241	best: 13.7600241 (2444)	total: 1m 25s	remaining: 54.1s
    2445:	learn: 8.9105611	test: 13.7608053	best: 13.7600241 (2444)	total: 1m 25s	remaining: 54s
    2446:	learn: 8.9100168	test: 13.7609760	best: 13.7600241 (2444)	total: 1m 25s	remaining: 54s
    2447:	learn: 8.9092063	test: 13.7635241	best: 13.7600241 (2444)	total: 1m 25s	remaining: 54s
    2448:	learn: 8.9085256	test: 13.7627318	best: 13.7600241 (2444)	total: 1m 25s	remaining: 53.9s
    2449:	learn: 8.9073772	test: 13.7611050	best: 13.7600241 (2444)	total: 1m 25s	remaining: 53.9s
    2450:	learn: 8.9061662	test: 13.7602689	best: 13.7600241 (2444)	total: 1m 25s	remaining: 53.8s
    2451:	learn: 8.9048882	test: 13.7594109	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.8s
    2452:	learn: 8.9040749	test: 13.7601675	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.8s
    2453:	learn: 8.9037988	test: 13.7603896	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.7s
    2454:	learn: 8.9026382	test: 13.7607722	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.7s
    2455:	learn: 8.9022295	test: 13.7616379	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.7s
    2456:	learn: 8.8996258	test: 13.7620064	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.6s
    2457:	learn: 8.8985589	test: 13.7637073	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.6s
    2458:	learn: 8.8979322	test: 13.7641221	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.5s
    2459:	learn: 8.8978566	test: 13.7635615	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.5s
    2460:	learn: 8.8971123	test: 13.7639020	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.5s
    2461:	learn: 8.8966391	test: 13.7623581	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.4s
    2462:	learn: 8.8959279	test: 13.7628865	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.4s
    2463:	learn: 8.8951205	test: 13.7617516	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.4s
    2464:	learn: 8.8931575	test: 13.7614379	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.3s
    2465:	learn: 8.8918373	test: 13.7612025	best: 13.7594109 (2451)	total: 1m 25s	remaining: 53.3s
    2466:	learn: 8.8900412	test: 13.7589503	best: 13.7589503 (2466)	total: 1m 25s	remaining: 53.3s
    2467:	learn: 8.8896892	test: 13.7587384	best: 13.7587384 (2467)	total: 1m 25s	remaining: 53.2s
    2468:	learn: 8.8890774	test: 13.7575053	best: 13.7575053 (2468)	total: 1m 25s	remaining: 53.2s
    2469:	learn: 8.8874148	test: 13.7561358	best: 13.7561358 (2469)	total: 1m 25s	remaining: 53.1s
    2470:	learn: 8.8865487	test: 13.7546548	best: 13.7546548 (2470)	total: 1m 25s	remaining: 53.1s
    2471:	learn: 8.8855750	test: 13.7539441	best: 13.7539441 (2471)	total: 1m 25s	remaining: 53.1s
    2472:	learn: 8.8845849	test: 13.7530833	best: 13.7530833 (2472)	total: 1m 25s	remaining: 53s
    2473:	learn: 8.8839499	test: 13.7523595	best: 13.7523595 (2473)	total: 1m 25s	remaining: 53s
    2474:	learn: 8.8808835	test: 13.7511460	best: 13.7511460 (2474)	total: 1m 25s	remaining: 53s
    2475:	learn: 8.8802986	test: 13.7526791	best: 13.7511460 (2474)	total: 1m 25s	remaining: 52.9s
    2476:	learn: 8.8797690	test: 13.7536618	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.9s
    2477:	learn: 8.8793822	test: 13.7542065	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.9s
    2478:	learn: 8.8783513	test: 13.7538588	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.8s
    2479:	learn: 8.8775664	test: 13.7540522	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.8s
    2480:	learn: 8.8769061	test: 13.7542690	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.7s
    2481:	learn: 8.8755519	test: 13.7531578	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.7s
    2482:	learn: 8.8745495	test: 13.7533319	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.7s
    2483:	learn: 8.8741451	test: 13.7548772	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.6s
    2484:	learn: 8.8737661	test: 13.7537566	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.6s
    2485:	learn: 8.8721033	test: 13.7545837	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.6s
    2486:	learn: 8.8719072	test: 13.7541296	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.5s
    2487:	learn: 8.8713123	test: 13.7550195	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.5s
    2488:	learn: 8.8705964	test: 13.7561026	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.4s
    2489:	learn: 8.8696306	test: 13.7560295	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.4s
    2490:	learn: 8.8691750	test: 13.7574458	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.4s
    2491:	learn: 8.8678662	test: 13.7575955	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.3s
    2492:	learn: 8.8667781	test: 13.7564506	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.3s
    2493:	learn: 8.8661355	test: 13.7559294	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.3s
    2494:	learn: 8.8656270	test: 13.7574303	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.2s
    2495:	learn: 8.8649233	test: 13.7568636	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.2s
    2496:	learn: 8.8630369	test: 13.7566537	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.1s
    2497:	learn: 8.8622503	test: 13.7574188	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.1s
    2498:	learn: 8.8614099	test: 13.7559963	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52.1s
    2499:	learn: 8.8605981	test: 13.7565469	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52s
    2500:	learn: 8.8586492	test: 13.7574887	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52s
    2501:	learn: 8.8572554	test: 13.7566196	best: 13.7511460 (2474)	total: 1m 26s	remaining: 52s
    2502:	learn: 8.8559252	test: 13.7588657	best: 13.7511460 (2474)	total: 1m 26s	remaining: 51.9s
    2503:	learn: 8.8553729	test: 13.7580346	best: 13.7511460 (2474)	total: 1m 26s	remaining: 51.9s
    2504:	learn: 8.8551481	test: 13.7579643	best: 13.7511460 (2474)	total: 1m 26s	remaining: 51.9s
    2505:	learn: 8.8543241	test: 13.7576514	best: 13.7511460 (2474)	total: 1m 26s	remaining: 51.8s
    2506:	learn: 8.8531499	test: 13.7564868	best: 13.7511460 (2474)	total: 1m 26s	remaining: 51.8s
    2507:	learn: 8.8528657	test: 13.7567802	best: 13.7511460 (2474)	total: 1m 26s	remaining: 51.7s
    2508:	learn: 8.8524673	test: 13.7567031	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.7s
    2509:	learn: 8.8504422	test: 13.7583843	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.7s
    2510:	learn: 8.8495309	test: 13.7586220	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.6s
    2511:	learn: 8.8488860	test: 13.7590284	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.6s
    2512:	learn: 8.8470889	test: 13.7589404	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.6s
    2513:	learn: 8.8462081	test: 13.7594695	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.5s
    2514:	learn: 8.8452940	test: 13.7618740	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.5s
    2515:	learn: 8.8451415	test: 13.7616810	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.4s
    2516:	learn: 8.8444079	test: 13.7613322	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.4s
    2517:	learn: 8.8431969	test: 13.7618321	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.4s
    2518:	learn: 8.8415276	test: 13.7624391	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.3s
    2519:	learn: 8.8413102	test: 13.7620391	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.3s
    2520:	learn: 8.8405777	test: 13.7616908	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.3s
    2521:	learn: 8.8382828	test: 13.7646215	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.2s
    2522:	learn: 8.8370476	test: 13.7632985	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.2s
    2523:	learn: 8.8358496	test: 13.7653475	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.1s
    2524:	learn: 8.8343604	test: 13.7651249	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.1s
    2525:	learn: 8.8337376	test: 13.7648825	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51.1s
    2526:	learn: 8.8334094	test: 13.7650090	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51s
    2527:	learn: 8.8329230	test: 13.7652726	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51s
    2528:	learn: 8.8316452	test: 13.7653555	best: 13.7511460 (2474)	total: 1m 27s	remaining: 51s
    2529:	learn: 8.8314808	test: 13.7661359	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.9s
    2530:	learn: 8.8304817	test: 13.7664682	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.9s
    2531:	learn: 8.8300117	test: 13.7660062	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.8s
    2532:	learn: 8.8292452	test: 13.7661774	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.8s
    2533:	learn: 8.8288631	test: 13.7660853	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.8s
    2534:	learn: 8.8277583	test: 13.7668586	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.7s
    2535:	learn: 8.8264790	test: 13.7668857	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.7s
    2536:	learn: 8.8259206	test: 13.7659524	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.7s
    2537:	learn: 8.8239371	test: 13.7669190	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.6s
    2538:	learn: 8.8236678	test: 13.7672362	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.6s
    2539:	learn: 8.8214709	test: 13.7662713	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.6s
    2540:	learn: 8.8199032	test: 13.7650444	best: 13.7511460 (2474)	total: 1m 27s	remaining: 50.5s
    2541:	learn: 8.8187924	test: 13.7649122	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.5s
    2542:	learn: 8.8180346	test: 13.7651618	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.5s
    2543:	learn: 8.8167878	test: 13.7666266	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.4s
    2544:	learn: 8.8160191	test: 13.7661337	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.4s
    2545:	learn: 8.8142820	test: 13.7641276	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.4s
    2546:	learn: 8.8136319	test: 13.7663898	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.3s
    2547:	learn: 8.8120276	test: 13.7657491	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.3s
    2548:	learn: 8.8117173	test: 13.7654046	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.2s
    2549:	learn: 8.8110244	test: 13.7657710	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.2s
    2550:	learn: 8.8099431	test: 13.7651253	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.2s
    2551:	learn: 8.8086445	test: 13.7671751	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.1s
    2552:	learn: 8.8083573	test: 13.7663006	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.1s
    2553:	learn: 8.8077156	test: 13.7656665	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50.1s
    2554:	learn: 8.8068632	test: 13.7651093	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50s
    2555:	learn: 8.8050209	test: 13.7639886	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50s
    2556:	learn: 8.8037579	test: 13.7644851	best: 13.7511460 (2474)	total: 1m 28s	remaining: 50s
    2557:	learn: 8.8030689	test: 13.7644069	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.9s
    2558:	learn: 8.8016803	test: 13.7654062	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.9s
    2559:	learn: 8.8009001	test: 13.7646275	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.8s
    2560:	learn: 8.8006750	test: 13.7636913	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.8s
    2561:	learn: 8.8004098	test: 13.7637540	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.8s
    2562:	learn: 8.7979912	test: 13.7650329	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.7s
    2563:	learn: 8.7950750	test: 13.7660520	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.7s
    2564:	learn: 8.7942244	test: 13.7672732	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.6s
    2565:	learn: 8.7903076	test: 13.7650098	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.6s
    2566:	learn: 8.7891431	test: 13.7649491	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.6s
    2567:	learn: 8.7880509	test: 13.7641368	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.5s
    2568:	learn: 8.7871675	test: 13.7646899	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.5s
    2569:	learn: 8.7859233	test: 13.7643445	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.5s
    2570:	learn: 8.7846631	test: 13.7643813	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.4s
    2571:	learn: 8.7837768	test: 13.7627747	best: 13.7511460 (2474)	total: 1m 28s	remaining: 49.4s
    2572:	learn: 8.7825154	test: 13.7618752	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49.4s
    2573:	learn: 8.7809265	test: 13.7608307	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49.3s
    2574:	learn: 8.7791152	test: 13.7623982	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49.3s
    2575:	learn: 8.7783635	test: 13.7628861	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49.2s
    2576:	learn: 8.7778821	test: 13.7634846	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49.2s
    2577:	learn: 8.7774473	test: 13.7633650	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49.2s
    2578:	learn: 8.7761186	test: 13.7638334	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49.1s
    2579:	learn: 8.7753626	test: 13.7634870	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49.1s
    2580:	learn: 8.7748824	test: 13.7637248	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49.1s
    2581:	learn: 8.7746540	test: 13.7639371	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49s
    2582:	learn: 8.7741192	test: 13.7637807	best: 13.7511460 (2474)	total: 1m 29s	remaining: 49s
    2583:	learn: 8.7735720	test: 13.7636390	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.9s
    2584:	learn: 8.7715323	test: 13.7650524	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.9s
    2585:	learn: 8.7711409	test: 13.7650688	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.9s
    2586:	learn: 8.7706024	test: 13.7637094	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.8s
    2587:	learn: 8.7688524	test: 13.7608857	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.8s
    2588:	learn: 8.7679939	test: 13.7626466	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.8s
    2589:	learn: 8.7668309	test: 13.7630625	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.7s
    2590:	learn: 8.7665185	test: 13.7631864	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.7s
    2591:	learn: 8.7649640	test: 13.7637002	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.6s
    2592:	learn: 8.7647552	test: 13.7653033	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.6s
    2593:	learn: 8.7636789	test: 13.7650330	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.6s
    2594:	learn: 8.7632172	test: 13.7645013	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.5s
    2595:	learn: 8.7622371	test: 13.7637264	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.5s
    2596:	learn: 8.7609450	test: 13.7630138	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.5s
    2597:	learn: 8.7602008	test: 13.7625660	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.4s
    2598:	learn: 8.7596712	test: 13.7620908	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.4s
    2599:	learn: 8.7575618	test: 13.7618990	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.4s
    2600:	learn: 8.7570191	test: 13.7612857	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.3s
    2601:	learn: 8.7552363	test: 13.7604963	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.3s
    2602:	learn: 8.7544522	test: 13.7612394	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.3s
    2603:	learn: 8.7536012	test: 13.7612031	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.2s
    2604:	learn: 8.7534569	test: 13.7614626	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.2s
    2605:	learn: 8.7527161	test: 13.7637321	best: 13.7511460 (2474)	total: 1m 29s	remaining: 48.1s
    2606:	learn: 8.7519397	test: 13.7631993	best: 13.7511460 (2474)	total: 1m 30s	remaining: 48.1s
    2607:	learn: 8.7508488	test: 13.7629595	best: 13.7511460 (2474)	total: 1m 30s	remaining: 48.1s
    2608:	learn: 8.7495310	test: 13.7624249	best: 13.7511460 (2474)	total: 1m 30s	remaining: 48s
    2609:	learn: 8.7486782	test: 13.7619170	best: 13.7511460 (2474)	total: 1m 30s	remaining: 48s
    2610:	learn: 8.7483444	test: 13.7610984	best: 13.7511460 (2474)	total: 1m 30s	remaining: 48s
    2611:	learn: 8.7466774	test: 13.7606700	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.9s
    2612:	learn: 8.7460886	test: 13.7598623	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.9s
    2613:	learn: 8.7449840	test: 13.7605729	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.9s
    2614:	learn: 8.7430711	test: 13.7623010	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.8s
    2615:	learn: 8.7427503	test: 13.7609372	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.8s
    2616:	learn: 8.7418701	test: 13.7617903	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.7s
    2617:	learn: 8.7409013	test: 13.7627560	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.7s
    2618:	learn: 8.7410609	test: 13.7631420	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.7s
    2619:	learn: 8.7403906	test: 13.7626756	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.6s
    2620:	learn: 8.7397514	test: 13.7631468	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.6s
    2621:	learn: 8.7378540	test: 13.7631680	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.6s
    2622:	learn: 8.7368351	test: 13.7655508	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.5s
    2623:	learn: 8.7364783	test: 13.7651682	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.5s
    2624:	learn: 8.7358277	test: 13.7655112	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.5s
    2625:	learn: 8.7349532	test: 13.7646234	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.4s
    2626:	learn: 8.7340274	test: 13.7642882	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.4s
    2627:	learn: 8.7315059	test: 13.7626828	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.4s
    2628:	learn: 8.7310664	test: 13.7634202	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.3s
    2629:	learn: 8.7297777	test: 13.7639924	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.3s
    2630:	learn: 8.7284116	test: 13.7640844	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.3s
    2631:	learn: 8.7285523	test: 13.7650500	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.2s
    2632:	learn: 8.7267871	test: 13.7663393	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.2s
    2633:	learn: 8.7262165	test: 13.7663663	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.2s
    2634:	learn: 8.7257155	test: 13.7662167	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.1s
    2635:	learn: 8.7251004	test: 13.7663774	best: 13.7511460 (2474)	total: 1m 30s	remaining: 47.1s
    2636:	learn: 8.7226624	test: 13.7663453	best: 13.7511460 (2474)	total: 1m 31s	remaining: 47s
    2637:	learn: 8.7216729	test: 13.7658345	best: 13.7511460 (2474)	total: 1m 31s	remaining: 47s
    2638:	learn: 8.7213177	test: 13.7666180	best: 13.7511460 (2474)	total: 1m 31s	remaining: 47s
    2639:	learn: 8.7195943	test: 13.7658557	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.9s
    2640:	learn: 8.7185843	test: 13.7657189	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.9s
    2641:	learn: 8.7180378	test: 13.7643480	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.9s
    2642:	learn: 8.7175585	test: 13.7654504	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.8s
    2643:	learn: 8.7163318	test: 13.7633642	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.8s
    2644:	learn: 8.7150942	test: 13.7626848	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.8s
    2645:	learn: 8.7136541	test: 13.7627413	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.7s
    2646:	learn: 8.7123228	test: 13.7624567	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.7s
    2647:	learn: 8.7120940	test: 13.7607996	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.7s
    2648:	learn: 8.7107563	test: 13.7634664	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.6s
    2649:	learn: 8.7100639	test: 13.7634343	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.6s
    2650:	learn: 8.7093640	test: 13.7630348	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.6s
    2651:	learn: 8.7085546	test: 13.7638897	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.5s
    2652:	learn: 8.7080368	test: 13.7655523	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.5s
    2653:	learn: 8.7067949	test: 13.7652492	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.5s
    2654:	learn: 8.7059459	test: 13.7651325	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.4s
    2655:	learn: 8.7052874	test: 13.7653988	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.4s
    2656:	learn: 8.7034341	test: 13.7630112	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.3s
    2657:	learn: 8.7029767	test: 13.7624529	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.3s
    2658:	learn: 8.7025930	test: 13.7630929	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.3s
    2659:	learn: 8.7016265	test: 13.7630315	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.2s
    2660:	learn: 8.6986711	test: 13.7628680	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.2s
    2661:	learn: 8.6963024	test: 13.7615292	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.2s
    2662:	learn: 8.6960372	test: 13.7613476	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.1s
    2663:	learn: 8.6946816	test: 13.7603313	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.1s
    2664:	learn: 8.6942773	test: 13.7598418	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46.1s
    2665:	learn: 8.6907870	test: 13.7579145	best: 13.7511460 (2474)	total: 1m 31s	remaining: 46s
    2666:	learn: 8.6890957	test: 13.7572254	best: 13.7511460 (2474)	total: 1m 32s	remaining: 46s
    2667:	learn: 8.6889067	test: 13.7578189	best: 13.7511460 (2474)	total: 1m 32s	remaining: 46s
    2668:	learn: 8.6877267	test: 13.7584415	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.9s
    2669:	learn: 8.6858718	test: 13.7577624	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.9s
    2670:	learn: 8.6847899	test: 13.7578790	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.8s
    2671:	learn: 8.6843171	test: 13.7573478	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.8s
    2672:	learn: 8.6838005	test: 13.7584250	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.8s
    2673:	learn: 8.6818887	test: 13.7585675	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.7s
    2674:	learn: 8.6808585	test: 13.7578528	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.7s
    2675:	learn: 8.6795263	test: 13.7570102	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.7s
    2676:	learn: 8.6782506	test: 13.7565570	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.6s
    2677:	learn: 8.6775647	test: 13.7567337	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.6s
    2678:	learn: 8.6764641	test: 13.7564555	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.6s
    2679:	learn: 8.6760951	test: 13.7563162	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.5s
    2680:	learn: 8.6757524	test: 13.7563742	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.5s
    2681:	learn: 8.6747296	test: 13.7539178	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.5s
    2682:	learn: 8.6739666	test: 13.7539932	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.4s
    2683:	learn: 8.6731593	test: 13.7544063	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.4s
    2684:	learn: 8.6728448	test: 13.7542205	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.4s
    2685:	learn: 8.6721274	test: 13.7533051	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.3s
    2686:	learn: 8.6710089	test: 13.7525384	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.3s
    2687:	learn: 8.6698074	test: 13.7523617	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.2s
    2688:	learn: 8.6688583	test: 13.7514440	best: 13.7511460 (2474)	total: 1m 32s	remaining: 45.2s
    2689:	learn: 8.6683667	test: 13.7504068	best: 13.7504068 (2689)	total: 1m 32s	remaining: 45.2s
    2690:	learn: 8.6665635	test: 13.7498812	best: 13.7498812 (2690)	total: 1m 32s	remaining: 45.1s
    2691:	learn: 8.6659702	test: 13.7503228	best: 13.7498812 (2690)	total: 1m 32s	remaining: 45.1s
    2692:	learn: 8.6646543	test: 13.7505883	best: 13.7498812 (2690)	total: 1m 32s	remaining: 45.1s
    2693:	learn: 8.6639253	test: 13.7518202	best: 13.7498812 (2690)	total: 1m 32s	remaining: 45s
    2694:	learn: 8.6629536	test: 13.7507416	best: 13.7498812 (2690)	total: 1m 32s	remaining: 45s
    2695:	learn: 8.6625755	test: 13.7512453	best: 13.7498812 (2690)	total: 1m 32s	remaining: 45s
    2696:	learn: 8.6625434	test: 13.7523728	best: 13.7498812 (2690)	total: 1m 33s	remaining: 44.9s
    2697:	learn: 8.6620714	test: 13.7522820	best: 13.7498812 (2690)	total: 1m 33s	remaining: 44.9s
    2698:	learn: 8.6616689	test: 13.7523601	best: 13.7498812 (2690)	total: 1m 33s	remaining: 44.9s
    2699:	learn: 8.6599596	test: 13.7536379	best: 13.7498812 (2690)	total: 1m 33s	remaining: 44.8s
    2700:	learn: 8.6595353	test: 13.7528003	best: 13.7498812 (2690)	total: 1m 33s	remaining: 44.8s
    2701:	learn: 8.6579052	test: 13.7523748	best: 13.7498812 (2690)	total: 1m 33s	remaining: 44.7s
    2702:	learn: 8.6574347	test: 13.7517314	best: 13.7498812 (2690)	total: 1m 33s	remaining: 44.7s
    2703:	learn: 8.6572279	test: 13.7522744	best: 13.7498812 (2690)	total: 1m 33s	remaining: 44.7s
    2704:	learn: 8.6565351	test: 13.7510541	best: 13.7498812 (2690)	total: 1m 33s	remaining: 44.6s
    2705:	learn: 8.6560680	test: 13.7496940	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.6s
    2706:	learn: 8.6556076	test: 13.7497731	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.6s
    2707:	learn: 8.6535656	test: 13.7504681	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.5s
    2708:	learn: 8.6527834	test: 13.7510363	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.5s
    2709:	learn: 8.6523782	test: 13.7509811	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.5s
    2710:	learn: 8.6518255	test: 13.7510253	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.4s
    2711:	learn: 8.6513168	test: 13.7503413	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.4s
    2712:	learn: 8.6499825	test: 13.7499752	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.4s
    2713:	learn: 8.6495422	test: 13.7506202	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.3s
    2714:	learn: 8.6490006	test: 13.7515359	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.3s
    2715:	learn: 8.6478733	test: 13.7513902	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.3s
    2716:	learn: 8.6462907	test: 13.7510463	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.2s
    2717:	learn: 8.6457699	test: 13.7516198	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.2s
    2718:	learn: 8.6441009	test: 13.7525786	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.2s
    2719:	learn: 8.6433151	test: 13.7533195	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.1s
    2720:	learn: 8.6427332	test: 13.7555899	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.1s
    2721:	learn: 8.6421311	test: 13.7553263	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44.1s
    2722:	learn: 8.6400252	test: 13.7541292	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44s
    2723:	learn: 8.6394954	test: 13.7548784	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44s
    2724:	learn: 8.6389791	test: 13.7556965	best: 13.7496940 (2705)	total: 1m 33s	remaining: 44s
    2725:	learn: 8.6381048	test: 13.7546813	best: 13.7496940 (2705)	total: 1m 33s	remaining: 43.9s
    2726:	learn: 8.6341202	test: 13.7544774	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.9s
    2727:	learn: 8.6333478	test: 13.7535936	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.9s
    2728:	learn: 8.6313008	test: 13.7536076	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.8s
    2729:	learn: 8.6298091	test: 13.7531437	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.8s
    2730:	learn: 8.6284830	test: 13.7534561	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.8s
    2731:	learn: 8.6275823	test: 13.7534993	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.7s
    2732:	learn: 8.6268565	test: 13.7534093	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.7s
    2733:	learn: 8.6255556	test: 13.7539018	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.7s
    2734:	learn: 8.6247186	test: 13.7540856	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.6s
    2735:	learn: 8.6241947	test: 13.7545775	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.6s
    2736:	learn: 8.6238088	test: 13.7542644	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.6s
    2737:	learn: 8.6230166	test: 13.7541965	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.5s
    2738:	learn: 8.6226528	test: 13.7545398	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.5s
    2739:	learn: 8.6206242	test: 13.7556661	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.5s
    2740:	learn: 8.6194340	test: 13.7549513	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.4s
    2741:	learn: 8.6174948	test: 13.7520628	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.4s
    2742:	learn: 8.6162059	test: 13.7524466	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.4s
    2743:	learn: 8.6142435	test: 13.7509867	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.3s
    2744:	learn: 8.6130440	test: 13.7509671	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.3s
    2745:	learn: 8.6120636	test: 13.7508782	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.3s
    2746:	learn: 8.6116641	test: 13.7504392	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.2s
    2747:	learn: 8.6106020	test: 13.7502241	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.2s
    2748:	learn: 8.6105493	test: 13.7510785	best: 13.7496940 (2705)	total: 1m 34s	remaining: 43.1s
    2749:	learn: 8.6080062	test: 13.7487736	best: 13.7487736 (2749)	total: 1m 34s	remaining: 43.1s
    2750:	learn: 8.6075390	test: 13.7502601	best: 13.7487736 (2749)	total: 1m 34s	remaining: 43.1s
    2751:	learn: 8.6072792	test: 13.7497619	best: 13.7487736 (2749)	total: 1m 34s	remaining: 43s
    2752:	learn: 8.6050870	test: 13.7480326	best: 13.7480326 (2752)	total: 1m 34s	remaining: 43s
    2753:	learn: 8.6041046	test: 13.7491036	best: 13.7480326 (2752)	total: 1m 34s	remaining: 43s
    2754:	learn: 8.6028293	test: 13.7493672	best: 13.7480326 (2752)	total: 1m 35s	remaining: 42.9s
    2755:	learn: 8.6023631	test: 13.7499759	best: 13.7480326 (2752)	total: 1m 35s	remaining: 42.9s
    2756:	learn: 8.6013174	test: 13.7488764	best: 13.7480326 (2752)	total: 1m 35s	remaining: 42.9s
    2757:	learn: 8.6012042	test: 13.7487289	best: 13.7480326 (2752)	total: 1m 35s	remaining: 42.9s
    2758:	learn: 8.6002926	test: 13.7489220	best: 13.7480326 (2752)	total: 1m 35s	remaining: 42.8s
    2759:	learn: 8.5995258	test: 13.7477528	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.8s
    2760:	learn: 8.5983622	test: 13.7489117	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.8s
    2761:	learn: 8.5975708	test: 13.7488047	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.7s
    2762:	learn: 8.5972102	test: 13.7483283	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.7s
    2763:	learn: 8.5966279	test: 13.7485251	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.7s
    2764:	learn: 8.5958293	test: 13.7492232	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.6s
    2765:	learn: 8.5937168	test: 13.7498058	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.6s
    2766:	learn: 8.5929160	test: 13.7497806	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.6s
    2767:	learn: 8.5921438	test: 13.7497602	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.5s
    2768:	learn: 8.5915365	test: 13.7501196	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.5s
    2769:	learn: 8.5910426	test: 13.7503680	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.5s
    2770:	learn: 8.5907078	test: 13.7517631	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.4s
    2771:	learn: 8.5890419	test: 13.7513412	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.4s
    2772:	learn: 8.5877567	test: 13.7519827	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.4s
    2773:	learn: 8.5870843	test: 13.7501204	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.3s
    2774:	learn: 8.5864397	test: 13.7502600	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.3s
    2775:	learn: 8.5849489	test: 13.7497769	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.3s
    2776:	learn: 8.5842857	test: 13.7510477	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.2s
    2777:	learn: 8.5843161	test: 13.7496469	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.2s
    2778:	learn: 8.5829976	test: 13.7498621	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.1s
    2779:	learn: 8.5825926	test: 13.7500002	best: 13.7477528 (2759)	total: 1m 35s	remaining: 42.1s
    2780:	learn: 8.5815160	test: 13.7497910	best: 13.7477528 (2759)	total: 1m 36s	remaining: 42.1s
    2781:	learn: 8.5794421	test: 13.7477556	best: 13.7477528 (2759)	total: 1m 36s	remaining: 42s
    2782:	learn: 8.5791121	test: 13.7482147	best: 13.7477528 (2759)	total: 1m 36s	remaining: 42s
    2783:	learn: 8.5773385	test: 13.7488068	best: 13.7477528 (2759)	total: 1m 36s	remaining: 42s
    2784:	learn: 8.5756879	test: 13.7488021	best: 13.7477528 (2759)	total: 1m 36s	remaining: 41.9s
    2785:	learn: 8.5745555	test: 13.7474151	best: 13.7474151 (2785)	total: 1m 36s	remaining: 41.9s
    2786:	learn: 8.5725586	test: 13.7457275	best: 13.7457275 (2786)	total: 1m 36s	remaining: 41.9s
    2787:	learn: 8.5720139	test: 13.7456046	best: 13.7456046 (2787)	total: 1m 36s	remaining: 41.9s
    2788:	learn: 8.5709055	test: 13.7451857	best: 13.7451857 (2788)	total: 1m 36s	remaining: 41.8s
    2789:	learn: 8.5698213	test: 13.7445853	best: 13.7445853 (2789)	total: 1m 36s	remaining: 41.8s
    2790:	learn: 8.5690154	test: 13.7454487	best: 13.7445853 (2789)	total: 1m 36s	remaining: 41.7s
    2791:	learn: 8.5666183	test: 13.7455385	best: 13.7445853 (2789)	total: 1m 36s	remaining: 41.7s
    2792:	learn: 8.5665973	test: 13.7477815	best: 13.7445853 (2789)	total: 1m 36s	remaining: 41.7s
    2793:	learn: 8.5647304	test: 13.7461305	best: 13.7445853 (2789)	total: 1m 36s	remaining: 41.6s
    2794:	learn: 8.5634520	test: 13.7448566	best: 13.7445853 (2789)	total: 1m 36s	remaining: 41.6s
    2795:	learn: 8.5619539	test: 13.7452263	best: 13.7445853 (2789)	total: 1m 36s	remaining: 41.6s
    2796:	learn: 8.5607581	test: 13.7455393	best: 13.7445853 (2789)	total: 1m 36s	remaining: 41.5s
    2797:	learn: 8.5598347	test: 13.7440645	best: 13.7440645 (2797)	total: 1m 36s	remaining: 41.5s
    2798:	learn: 8.5591769	test: 13.7435751	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.5s
    2799:	learn: 8.5581878	test: 13.7436212	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.4s
    2800:	learn: 8.5569437	test: 13.7452655	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.4s
    2801:	learn: 8.5554784	test: 13.7459295	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.4s
    2802:	learn: 8.5546536	test: 13.7453996	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.3s
    2803:	learn: 8.5542517	test: 13.7461222	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.3s
    2804:	learn: 8.5537784	test: 13.7457010	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.3s
    2805:	learn: 8.5526611	test: 13.7460750	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.2s
    2806:	learn: 8.5519092	test: 13.7459797	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.2s
    2807:	learn: 8.5511324	test: 13.7455244	best: 13.7435751 (2798)	total: 1m 36s	remaining: 41.2s
    2808:	learn: 8.5504776	test: 13.7454801	best: 13.7435751 (2798)	total: 1m 37s	remaining: 41.1s
    2809:	learn: 8.5493031	test: 13.7443914	best: 13.7435751 (2798)	total: 1m 37s	remaining: 41.1s
    2810:	learn: 8.5471511	test: 13.7473117	best: 13.7435751 (2798)	total: 1m 37s	remaining: 41.1s
    2811:	learn: 8.5465602	test: 13.7476164	best: 13.7435751 (2798)	total: 1m 37s	remaining: 41s
    2812:	learn: 8.5441687	test: 13.7498256	best: 13.7435751 (2798)	total: 1m 37s	remaining: 41s
    2813:	learn: 8.5436820	test: 13.7490550	best: 13.7435751 (2798)	total: 1m 37s	remaining: 41s
    2814:	learn: 8.5415571	test: 13.7495071	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.9s
    2815:	learn: 8.5401523	test: 13.7490955	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.9s
    2816:	learn: 8.5389431	test: 13.7498358	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.9s
    2817:	learn: 8.5374100	test: 13.7484957	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.8s
    2818:	learn: 8.5353996	test: 13.7454359	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.8s
    2819:	learn: 8.5351205	test: 13.7448863	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.7s
    2820:	learn: 8.5344504	test: 13.7459670	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.7s
    2821:	learn: 8.5333794	test: 13.7457393	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.7s
    2822:	learn: 8.5319870	test: 13.7457697	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.6s
    2823:	learn: 8.5317688	test: 13.7467315	best: 13.7435751 (2798)	total: 1m 37s	remaining: 40.6s
    2824:	learn: 8.5305339	test: 13.7432997	best: 13.7432997 (2824)	total: 1m 37s	remaining: 40.6s
    2825:	learn: 8.5296730	test: 13.7437265	best: 13.7432997 (2824)	total: 1m 37s	remaining: 40.5s
    2826:	learn: 8.5288931	test: 13.7437105	best: 13.7432997 (2824)	total: 1m 37s	remaining: 40.5s
    2827:	learn: 8.5283268	test: 13.7440951	best: 13.7432997 (2824)	total: 1m 37s	remaining: 40.5s
    2828:	learn: 8.5276757	test: 13.7433601	best: 13.7432997 (2824)	total: 1m 37s	remaining: 40.4s
    2829:	learn: 8.5275949	test: 13.7428325	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.4s
    2830:	learn: 8.5260953	test: 13.7434957	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.4s
    2831:	learn: 8.5257981	test: 13.7441456	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.3s
    2832:	learn: 8.5246933	test: 13.7461675	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.3s
    2833:	learn: 8.5241914	test: 13.7464860	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.2s
    2834:	learn: 8.5229877	test: 13.7471543	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.2s
    2835:	learn: 8.5226203	test: 13.7471196	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.2s
    2836:	learn: 8.5216184	test: 13.7483640	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.1s
    2837:	learn: 8.5213240	test: 13.7474339	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.1s
    2838:	learn: 8.5204227	test: 13.7469892	best: 13.7428325 (2829)	total: 1m 37s	remaining: 40.1s
    2839:	learn: 8.5196127	test: 13.7471560	best: 13.7428325 (2829)	total: 1m 38s	remaining: 40s
    2840:	learn: 8.5182466	test: 13.7461567	best: 13.7428325 (2829)	total: 1m 38s	remaining: 40s
    2841:	learn: 8.5163857	test: 13.7470005	best: 13.7428325 (2829)	total: 1m 38s	remaining: 40s
    2842:	learn: 8.5156949	test: 13.7480333	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.9s
    2843:	learn: 8.5144866	test: 13.7481343	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.9s
    2844:	learn: 8.5139490	test: 13.7488045	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.9s
    2845:	learn: 8.5129578	test: 13.7485661	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.8s
    2846:	learn: 8.5117960	test: 13.7470048	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.8s
    2847:	learn: 8.5104229	test: 13.7455520	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.7s
    2848:	learn: 8.5093406	test: 13.7461005	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.7s
    2849:	learn: 8.5084588	test: 13.7451083	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.7s
    2850:	learn: 8.5080808	test: 13.7444639	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.6s
    2851:	learn: 8.5074621	test: 13.7431912	best: 13.7428325 (2829)	total: 1m 38s	remaining: 39.6s
    2852:	learn: 8.5063970	test: 13.7421441	best: 13.7421441 (2852)	total: 1m 38s	remaining: 39.6s
    2853:	learn: 8.5055322	test: 13.7425931	best: 13.7421441 (2852)	total: 1m 38s	remaining: 39.5s
    2854:	learn: 8.5038425	test: 13.7427691	best: 13.7421441 (2852)	total: 1m 38s	remaining: 39.5s
    2855:	learn: 8.5033270	test: 13.7429776	best: 13.7421441 (2852)	total: 1m 38s	remaining: 39.5s
    2856:	learn: 8.5030161	test: 13.7429923	best: 13.7421441 (2852)	total: 1m 38s	remaining: 39.4s
    2857:	learn: 8.5025678	test: 13.7418110	best: 13.7418110 (2857)	total: 1m 38s	remaining: 39.4s
    2858:	learn: 8.5022438	test: 13.7419245	best: 13.7418110 (2857)	total: 1m 38s	remaining: 39.4s
    2859:	learn: 8.5020195	test: 13.7424856	best: 13.7418110 (2857)	total: 1m 38s	remaining: 39.3s
    2860:	learn: 8.5006990	test: 13.7429855	best: 13.7418110 (2857)	total: 1m 38s	remaining: 39.3s
    2861:	learn: 8.5002647	test: 13.7434812	best: 13.7418110 (2857)	total: 1m 38s	remaining: 39.3s
    2862:	learn: 8.4998864	test: 13.7448052	best: 13.7418110 (2857)	total: 1m 38s	remaining: 39.2s
    2863:	learn: 8.4997597	test: 13.7424681	best: 13.7418110 (2857)	total: 1m 38s	remaining: 39.2s
    2864:	learn: 8.4984912	test: 13.7425108	best: 13.7418110 (2857)	total: 1m 38s	remaining: 39.2s
    2865:	learn: 8.4967346	test: 13.7419059	best: 13.7418110 (2857)	total: 1m 38s	remaining: 39.1s
    2866:	learn: 8.4959359	test: 13.7415773	best: 13.7415773 (2866)	total: 1m 38s	remaining: 39.1s
    2867:	learn: 8.4947399	test: 13.7410470	best: 13.7410470 (2867)	total: 1m 38s	remaining: 39.1s
    2868:	learn: 8.4942656	test: 13.7399107	best: 13.7399107 (2868)	total: 1m 39s	remaining: 39s
    2869:	learn: 8.4935689	test: 13.7413443	best: 13.7399107 (2868)	total: 1m 39s	remaining: 39s
    2870:	learn: 8.4926476	test: 13.7409537	best: 13.7399107 (2868)	total: 1m 39s	remaining: 39s
    2871:	learn: 8.4917354	test: 13.7406283	best: 13.7399107 (2868)	total: 1m 39s	remaining: 38.9s
    2872:	learn: 8.4913932	test: 13.7401442	best: 13.7399107 (2868)	total: 1m 39s	remaining: 38.9s
    2873:	learn: 8.4909723	test: 13.7403883	best: 13.7399107 (2868)	total: 1m 39s	remaining: 38.9s
    2874:	learn: 8.4896659	test: 13.7393769	best: 13.7393769 (2874)	total: 1m 39s	remaining: 38.8s
    2875:	learn: 8.4895121	test: 13.7396771	best: 13.7393769 (2874)	total: 1m 39s	remaining: 38.8s
    2876:	learn: 8.4892903	test: 13.7392653	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.8s
    2877:	learn: 8.4888418	test: 13.7400216	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.7s
    2878:	learn: 8.4879776	test: 13.7399845	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.7s
    2879:	learn: 8.4871280	test: 13.7412716	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.7s
    2880:	learn: 8.4865283	test: 13.7427435	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.6s
    2881:	learn: 8.4860044	test: 13.7439941	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.6s
    2882:	learn: 8.4853838	test: 13.7443538	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.6s
    2883:	learn: 8.4845141	test: 13.7457664	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.5s
    2884:	learn: 8.4844277	test: 13.7458693	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.5s
    2885:	learn: 8.4836616	test: 13.7461767	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.5s
    2886:	learn: 8.4830749	test: 13.7471597	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.4s
    2887:	learn: 8.4822599	test: 13.7460802	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.4s
    2888:	learn: 8.4806205	test: 13.7458320	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.3s
    2889:	learn: 8.4798591	test: 13.7456451	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.3s
    2890:	learn: 8.4790268	test: 13.7465518	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.3s
    2891:	learn: 8.4785840	test: 13.7460176	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.2s
    2892:	learn: 8.4774203	test: 13.7473248	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.2s
    2893:	learn: 8.4756678	test: 13.7455429	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.2s
    2894:	learn: 8.4733769	test: 13.7435713	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.1s
    2895:	learn: 8.4723460	test: 13.7426677	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.1s
    2896:	learn: 8.4719862	test: 13.7439236	best: 13.7392653 (2876)	total: 1m 39s	remaining: 38.1s
    2897:	learn: 8.4714123	test: 13.7445763	best: 13.7392653 (2876)	total: 1m 40s	remaining: 38s
    2898:	learn: 8.4702950	test: 13.7457730	best: 13.7392653 (2876)	total: 1m 40s	remaining: 38s
    2899:	learn: 8.4702472	test: 13.7453030	best: 13.7392653 (2876)	total: 1m 40s	remaining: 38s
    2900:	learn: 8.4691580	test: 13.7464124	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.9s
    2901:	learn: 8.4653947	test: 13.7442618	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.9s
    2902:	learn: 8.4643409	test: 13.7445906	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.9s
    2903:	learn: 8.4626219	test: 13.7432249	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.8s
    2904:	learn: 8.4616153	test: 13.7427854	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.8s
    2905:	learn: 8.4609824	test: 13.7429507	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.7s
    2906:	learn: 8.4604446	test: 13.7429163	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.7s
    2907:	learn: 8.4591530	test: 13.7450520	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.7s
    2908:	learn: 8.4585753	test: 13.7446622	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.6s
    2909:	learn: 8.4564047	test: 13.7430891	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.6s
    2910:	learn: 8.4557779	test: 13.7440327	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.6s
    2911:	learn: 8.4555004	test: 13.7443218	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.5s
    2912:	learn: 8.4546689	test: 13.7441471	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.5s
    2913:	learn: 8.4539997	test: 13.7443345	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.5s
    2914:	learn: 8.4527691	test: 13.7458052	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.4s
    2915:	learn: 8.4507025	test: 13.7461365	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.4s
    2916:	learn: 8.4488105	test: 13.7464619	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.4s
    2917:	learn: 8.4472971	test: 13.7471456	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.3s
    2918:	learn: 8.4462367	test: 13.7463658	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.3s
    2919:	learn: 8.4442031	test: 13.7466135	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.2s
    2920:	learn: 8.4438506	test: 13.7467934	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.2s
    2921:	learn: 8.4422387	test: 13.7477603	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.2s
    2922:	learn: 8.4401440	test: 13.7471189	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.1s
    2923:	learn: 8.4377638	test: 13.7469699	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.1s
    2924:	learn: 8.4361702	test: 13.7454455	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37.1s
    2925:	learn: 8.4353968	test: 13.7453233	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37s
    2926:	learn: 8.4348218	test: 13.7451339	best: 13.7392653 (2876)	total: 1m 40s	remaining: 37s
    2927:	learn: 8.4337298	test: 13.7440947	best: 13.7392653 (2876)	total: 1m 40s	remaining: 36.9s
    2928:	learn: 8.4320851	test: 13.7453822	best: 13.7392653 (2876)	total: 1m 40s	remaining: 36.9s
    2929:	learn: 8.4314058	test: 13.7463314	best: 13.7392653 (2876)	total: 1m 40s	remaining: 36.9s
    2930:	learn: 8.4309058	test: 13.7463540	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.8s
    2931:	learn: 8.4297173	test: 13.7446048	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.8s
    2932:	learn: 8.4281811	test: 13.7459828	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.8s
    2933:	learn: 8.4264415	test: 13.7436670	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.7s
    2934:	learn: 8.4263455	test: 13.7432097	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.7s
    2935:	learn: 8.4258439	test: 13.7431918	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.7s
    2936:	learn: 8.4249355	test: 13.7427444	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.6s
    2937:	learn: 8.4239078	test: 13.7415708	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.6s
    2938:	learn: 8.4227981	test: 13.7428471	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.6s
    2939:	learn: 8.4220589	test: 13.7434619	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.5s
    2940:	learn: 8.4208824	test: 13.7437076	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.5s
    2941:	learn: 8.4202450	test: 13.7429403	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.4s
    2942:	learn: 8.4200364	test: 13.7417393	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.4s
    2943:	learn: 8.4200411	test: 13.7424882	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.4s
    2944:	learn: 8.4193120	test: 13.7427597	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.3s
    2945:	learn: 8.4179166	test: 13.7437180	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.3s
    2946:	learn: 8.4168216	test: 13.7455249	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.3s
    2947:	learn: 8.4155123	test: 13.7451367	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.2s
    2948:	learn: 8.4148431	test: 13.7459659	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.2s
    2949:	learn: 8.4142306	test: 13.7453742	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.2s
    2950:	learn: 8.4127243	test: 13.7458382	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.1s
    2951:	learn: 8.4118396	test: 13.7463825	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.1s
    2952:	learn: 8.4097320	test: 13.7465990	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36.1s
    2953:	learn: 8.4084630	test: 13.7465921	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36s
    2954:	learn: 8.4081700	test: 13.7456530	best: 13.7392653 (2876)	total: 1m 41s	remaining: 36s
    2955:	learn: 8.4055157	test: 13.7457359	best: 13.7392653 (2876)	total: 1m 41s	remaining: 35.9s
    2956:	learn: 8.4033246	test: 13.7449952	best: 13.7392653 (2876)	total: 1m 41s	remaining: 35.9s
    2957:	learn: 8.4024652	test: 13.7454641	best: 13.7392653 (2876)	total: 1m 41s	remaining: 35.9s
    2958:	learn: 8.4014693	test: 13.7462636	best: 13.7392653 (2876)	total: 1m 41s	remaining: 35.8s
    2959:	learn: 8.4001268	test: 13.7470196	best: 13.7392653 (2876)	total: 1m 41s	remaining: 35.8s
    2960:	learn: 8.3996928	test: 13.7473178	best: 13.7392653 (2876)	total: 1m 41s	remaining: 35.8s
    2961:	learn: 8.3987438	test: 13.7467134	best: 13.7392653 (2876)	total: 1m 41s	remaining: 35.7s
    2962:	learn: 8.3981296	test: 13.7473221	best: 13.7392653 (2876)	total: 1m 41s	remaining: 35.7s
    2963:	learn: 8.3974261	test: 13.7483198	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.7s
    2964:	learn: 8.3970037	test: 13.7479309	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.6s
    2965:	learn: 8.3966044	test: 13.7481935	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.6s
    2966:	learn: 8.3955045	test: 13.7466797	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.6s
    2967:	learn: 8.3955789	test: 13.7480825	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.5s
    2968:	learn: 8.3953731	test: 13.7485167	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.5s
    2969:	learn: 8.3951282	test: 13.7488170	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.4s
    2970:	learn: 8.3949145	test: 13.7485961	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.4s
    2971:	learn: 8.3944598	test: 13.7489321	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.4s
    2972:	learn: 8.3936067	test: 13.7493362	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.3s
    2973:	learn: 8.3917223	test: 13.7497548	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.3s
    2974:	learn: 8.3913194	test: 13.7495172	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.3s
    2975:	learn: 8.3909011	test: 13.7479611	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.2s
    2976:	learn: 8.3894189	test: 13.7464608	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.2s
    2977:	learn: 8.3884635	test: 13.7473562	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.2s
    2978:	learn: 8.3880303	test: 13.7473515	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.1s
    2979:	learn: 8.3872031	test: 13.7480839	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.1s
    2980:	learn: 8.3863988	test: 13.7489990	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35.1s
    2981:	learn: 8.3845607	test: 13.7480007	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35s
    2982:	learn: 8.3840187	test: 13.7491869	best: 13.7392653 (2876)	total: 1m 42s	remaining: 35s
    2983:	learn: 8.3821370	test: 13.7488340	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.9s
    2984:	learn: 8.3817058	test: 13.7497247	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.9s
    2985:	learn: 8.3815219	test: 13.7499575	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.9s
    2986:	learn: 8.3814067	test: 13.7502839	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.8s
    2987:	learn: 8.3800650	test: 13.7506753	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.8s
    2988:	learn: 8.3795053	test: 13.7515766	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.8s
    2989:	learn: 8.3791444	test: 13.7519029	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.7s
    2990:	learn: 8.3785064	test: 13.7507404	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.7s
    2991:	learn: 8.3774004	test: 13.7514729	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.7s
    2992:	learn: 8.3760729	test: 13.7504048	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.6s
    2993:	learn: 8.3749553	test: 13.7498851	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.6s
    2994:	learn: 8.3746885	test: 13.7501940	best: 13.7392653 (2876)	total: 1m 42s	remaining: 34.6s
    2995:	learn: 8.3736483	test: 13.7493509	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.5s
    2996:	learn: 8.3729498	test: 13.7493311	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.5s
    2997:	learn: 8.3716002	test: 13.7498425	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.5s
    2998:	learn: 8.3706780	test: 13.7484336	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.4s
    2999:	learn: 8.3693354	test: 13.7478769	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.4s
    3000:	learn: 8.3686764	test: 13.7478435	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.3s
    3001:	learn: 8.3681311	test: 13.7474928	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.3s
    3002:	learn: 8.3673815	test: 13.7470199	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.3s
    3003:	learn: 8.3669687	test: 13.7474787	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.2s
    3004:	learn: 8.3660317	test: 13.7476183	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.2s
    3005:	learn: 8.3653443	test: 13.7461866	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.2s
    3006:	learn: 8.3642888	test: 13.7477521	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.1s
    3007:	learn: 8.3636291	test: 13.7481746	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.1s
    3008:	learn: 8.3633091	test: 13.7489441	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34.1s
    3009:	learn: 8.3623259	test: 13.7494496	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34s
    3010:	learn: 8.3620569	test: 13.7497983	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34s
    3011:	learn: 8.3612111	test: 13.7493528	best: 13.7392653 (2876)	total: 1m 43s	remaining: 34s
    3012:	learn: 8.3599561	test: 13.7505195	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.9s
    3013:	learn: 8.3593189	test: 13.7496018	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.9s
    3014:	learn: 8.3584632	test: 13.7484775	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.9s
    3015:	learn: 8.3579729	test: 13.7491319	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.8s
    3016:	learn: 8.3575305	test: 13.7486483	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.8s
    3017:	learn: 8.3569358	test: 13.7471006	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.7s
    3018:	learn: 8.3550204	test: 13.7439066	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.7s
    3019:	learn: 8.3542787	test: 13.7454420	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.7s
    3020:	learn: 8.3533285	test: 13.7468966	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.6s
    3021:	learn: 8.3530492	test: 13.7475635	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.6s
    3022:	learn: 8.3527414	test: 13.7472048	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.6s
    3023:	learn: 8.3511796	test: 13.7466870	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.5s
    3024:	learn: 8.3503782	test: 13.7475436	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.5s
    3025:	learn: 8.3491053	test: 13.7464090	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.5s
    3026:	learn: 8.3465888	test: 13.7455044	best: 13.7392653 (2876)	total: 1m 43s	remaining: 33.4s
    3027:	learn: 8.3464435	test: 13.7462211	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.4s
    3028:	learn: 8.3462889	test: 13.7450402	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.4s
    3029:	learn: 8.3455835	test: 13.7438232	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.3s
    3030:	learn: 8.3446920	test: 13.7434265	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.3s
    3031:	learn: 8.3444303	test: 13.7443172	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.2s
    3032:	learn: 8.3428265	test: 13.7447172	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.2s
    3033:	learn: 8.3418166	test: 13.7448648	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.2s
    3034:	learn: 8.3412650	test: 13.7454168	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.1s
    3035:	learn: 8.3404655	test: 13.7440591	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.1s
    3036:	learn: 8.3402362	test: 13.7437865	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33.1s
    3037:	learn: 8.3391040	test: 13.7443940	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33s
    3038:	learn: 8.3382052	test: 13.7451921	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33s
    3039:	learn: 8.3367801	test: 13.7442832	best: 13.7392653 (2876)	total: 1m 44s	remaining: 33s
    3040:	learn: 8.3366179	test: 13.7453597	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.9s
    3041:	learn: 8.3359690	test: 13.7459508	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.9s
    3042:	learn: 8.3353530	test: 13.7463601	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.8s
    3043:	learn: 8.3346058	test: 13.7468425	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.8s
    3044:	learn: 8.3341112	test: 13.7469807	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.8s
    3045:	learn: 8.3336233	test: 13.7463806	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.7s
    3046:	learn: 8.3336713	test: 13.7465236	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.7s
    3047:	learn: 8.3317532	test: 13.7458969	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.7s
    3048:	learn: 8.3311228	test: 13.7449717	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.6s
    3049:	learn: 8.3303848	test: 13.7440843	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.6s
    3050:	learn: 8.3303069	test: 13.7435768	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.5s
    3051:	learn: 8.3296222	test: 13.7449866	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.5s
    3052:	learn: 8.3289600	test: 13.7453209	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.5s
    3053:	learn: 8.3284259	test: 13.7455996	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.4s
    3054:	learn: 8.3279582	test: 13.7456883	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.4s
    3055:	learn: 8.3250979	test: 13.7461753	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.4s
    3056:	learn: 8.3246574	test: 13.7452590	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.3s
    3057:	learn: 8.3226220	test: 13.7440645	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.3s
    3058:	learn: 8.3222852	test: 13.7441515	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.3s
    3059:	learn: 8.3214898	test: 13.7444114	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.2s
    3060:	learn: 8.3209883	test: 13.7433946	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.2s
    3061:	learn: 8.3207349	test: 13.7435953	best: 13.7392653 (2876)	total: 1m 44s	remaining: 32.2s
    3062:	learn: 8.3201464	test: 13.7434173	best: 13.7392653 (2876)	total: 1m 45s	remaining: 32.1s
    3063:	learn: 8.3200484	test: 13.7421870	best: 13.7392653 (2876)	total: 1m 45s	remaining: 32.1s
    3064:	learn: 8.3195432	test: 13.7425884	best: 13.7392653 (2876)	total: 1m 45s	remaining: 32.1s
    3065:	learn: 8.3180369	test: 13.7438321	best: 13.7392653 (2876)	total: 1m 45s	remaining: 32s
    3066:	learn: 8.3177984	test: 13.7418893	best: 13.7392653 (2876)	total: 1m 45s	remaining: 32s
    3067:	learn: 8.3171964	test: 13.7431762	best: 13.7392653 (2876)	total: 1m 45s	remaining: 32s
    3068:	learn: 8.3158239	test: 13.7423543	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.9s
    3069:	learn: 8.3148419	test: 13.7442267	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.9s
    3070:	learn: 8.3137838	test: 13.7438970	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.8s
    3071:	learn: 8.3132329	test: 13.7447428	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.8s
    3072:	learn: 8.3126068	test: 13.7442169	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.8s
    3073:	learn: 8.3119716	test: 13.7441681	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.7s
    3074:	learn: 8.3113988	test: 13.7435793	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.7s
    3075:	learn: 8.3107139	test: 13.7442713	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.7s
    3076:	learn: 8.3098749	test: 13.7448510	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.6s
    3077:	learn: 8.3101115	test: 13.7439410	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.6s
    3078:	learn: 8.3094336	test: 13.7444724	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.6s
    3079:	learn: 8.3075365	test: 13.7447503	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.5s
    3080:	learn: 8.3063126	test: 13.7454087	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.5s
    3081:	learn: 8.3056252	test: 13.7444787	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.5s
    3082:	learn: 8.3050960	test: 13.7445250	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.4s
    3083:	learn: 8.3038839	test: 13.7441007	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.4s
    3084:	learn: 8.3020762	test: 13.7443421	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.4s
    3085:	learn: 8.3014162	test: 13.7429353	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.3s
    3086:	learn: 8.3002014	test: 13.7428463	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.3s
    3087:	learn: 8.2998180	test: 13.7412093	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.2s
    3088:	learn: 8.2997100	test: 13.7404627	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.2s
    3089:	learn: 8.2994925	test: 13.7396868	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.2s
    3090:	learn: 8.2989818	test: 13.7397011	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.1s
    3091:	learn: 8.2986913	test: 13.7392873	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.1s
    3092:	learn: 8.2976132	test: 13.7400671	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31.1s
    3093:	learn: 8.2972553	test: 13.7403034	best: 13.7392653 (2876)	total: 1m 45s	remaining: 31s
    3094:	learn: 8.2957430	test: 13.7379133	best: 13.7379133 (3094)	total: 1m 45s	remaining: 31s
    3095:	learn: 8.2947355	test: 13.7388305	best: 13.7379133 (3094)	total: 1m 46s	remaining: 31s
    3096:	learn: 8.2942997	test: 13.7391771	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.9s
    3097:	learn: 8.2932115	test: 13.7391362	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.9s
    3098:	learn: 8.2919882	test: 13.7407125	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.8s
    3099:	learn: 8.2904722	test: 13.7416449	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.8s
    3100:	learn: 8.2892681	test: 13.7427463	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.8s
    3101:	learn: 8.2888603	test: 13.7420391	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.7s
    3102:	learn: 8.2883540	test: 13.7418988	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.7s
    3103:	learn: 8.2880457	test: 13.7426921	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.7s
    3104:	learn: 8.2877627	test: 13.7431653	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.6s
    3105:	learn: 8.2860071	test: 13.7415597	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.6s
    3106:	learn: 8.2855505	test: 13.7424262	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.6s
    3107:	learn: 8.2852940	test: 13.7426445	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.5s
    3108:	learn: 8.2844626	test: 13.7406448	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.5s
    3109:	learn: 8.2839662	test: 13.7416987	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.5s
    3110:	learn: 8.2834794	test: 13.7415278	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.4s
    3111:	learn: 8.2836224	test: 13.7428535	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.4s
    3112:	learn: 8.2828745	test: 13.7411356	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.3s
    3113:	learn: 8.2820756	test: 13.7422176	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.3s
    3114:	learn: 8.2815968	test: 13.7421242	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.3s
    3115:	learn: 8.2811996	test: 13.7440719	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.2s
    3116:	learn: 8.2811275	test: 13.7434568	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.2s
    3117:	learn: 8.2786743	test: 13.7420213	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.2s
    3118:	learn: 8.2783117	test: 13.7434026	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.1s
    3119:	learn: 8.2775493	test: 13.7447227	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.1s
    3120:	learn: 8.2763870	test: 13.7447946	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30.1s
    3121:	learn: 8.2755165	test: 13.7446303	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30s
    3122:	learn: 8.2748336	test: 13.7444337	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30s
    3123:	learn: 8.2734353	test: 13.7445977	best: 13.7379133 (3094)	total: 1m 46s	remaining: 30s
    3124:	learn: 8.2724120	test: 13.7426854	best: 13.7379133 (3094)	total: 1m 46s	remaining: 29.9s
    3125:	learn: 8.2714828	test: 13.7433096	best: 13.7379133 (3094)	total: 1m 46s	remaining: 29.9s
    3126:	learn: 8.2708310	test: 13.7413214	best: 13.7379133 (3094)	total: 1m 46s	remaining: 29.9s
    3127:	learn: 8.2700781	test: 13.7427862	best: 13.7379133 (3094)	total: 1m 46s	remaining: 29.8s
    3128:	learn: 8.2692989	test: 13.7427117	best: 13.7379133 (3094)	total: 1m 46s	remaining: 29.8s
    3129:	learn: 8.2680799	test: 13.7438737	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.7s
    3130:	learn: 8.2676714	test: 13.7443311	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.7s
    3131:	learn: 8.2666235	test: 13.7437918	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.7s
    3132:	learn: 8.2660326	test: 13.7432941	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.6s
    3133:	learn: 8.2656409	test: 13.7439202	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.6s
    3134:	learn: 8.2643616	test: 13.7440176	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.6s
    3135:	learn: 8.2636590	test: 13.7432230	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.5s
    3136:	learn: 8.2618920	test: 13.7421851	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.5s
    3137:	learn: 8.2620894	test: 13.7426148	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.5s
    3138:	learn: 8.2613481	test: 13.7428367	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.4s
    3139:	learn: 8.2611621	test: 13.7436222	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.4s
    3140:	learn: 8.2607631	test: 13.7440059	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.3s
    3141:	learn: 8.2599392	test: 13.7440358	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.3s
    3142:	learn: 8.2585944	test: 13.7441827	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.3s
    3143:	learn: 8.2573299	test: 13.7442530	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.2s
    3144:	learn: 8.2565027	test: 13.7437304	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.2s
    3145:	learn: 8.2560270	test: 13.7436757	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.2s
    3146:	learn: 8.2556195	test: 13.7438647	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.1s
    3147:	learn: 8.2553107	test: 13.7431380	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.1s
    3148:	learn: 8.2541744	test: 13.7450793	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29.1s
    3149:	learn: 8.2538528	test: 13.7450003	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29s
    3150:	learn: 8.2535936	test: 13.7450431	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29s
    3151:	learn: 8.2523906	test: 13.7461407	best: 13.7379133 (3094)	total: 1m 47s	remaining: 29s
    3152:	learn: 8.2519527	test: 13.7468411	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.9s
    3153:	learn: 8.2516967	test: 13.7470971	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.9s
    3154:	learn: 8.2505256	test: 13.7464015	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.9s
    3155:	learn: 8.2497956	test: 13.7459377	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.8s
    3156:	learn: 8.2481435	test: 13.7464732	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.8s
    3157:	learn: 8.2482125	test: 13.7463378	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.7s
    3158:	learn: 8.2476424	test: 13.7474045	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.7s
    3159:	learn: 8.2472872	test: 13.7463939	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.7s
    3160:	learn: 8.2471968	test: 13.7465567	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.6s
    3161:	learn: 8.2462304	test: 13.7471331	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.6s
    3162:	learn: 8.2452728	test: 13.7475925	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.6s
    3163:	learn: 8.2441876	test: 13.7475466	best: 13.7379133 (3094)	total: 1m 47s	remaining: 28.5s
    3164:	learn: 8.2439864	test: 13.7473504	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.5s
    3165:	learn: 8.2431537	test: 13.7482600	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.5s
    3166:	learn: 8.2417492	test: 13.7489249	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.4s
    3167:	learn: 8.2413637	test: 13.7494144	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.4s
    3168:	learn: 8.2407487	test: 13.7489262	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.4s
    3169:	learn: 8.2405154	test: 13.7486814	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.3s
    3170:	learn: 8.2391421	test: 13.7477548	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.3s
    3171:	learn: 8.2386408	test: 13.7472074	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.2s
    3172:	learn: 8.2374378	test: 13.7474584	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.2s
    3173:	learn: 8.2364338	test: 13.7472396	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.2s
    3174:	learn: 8.2353575	test: 13.7470904	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.1s
    3175:	learn: 8.2345284	test: 13.7478524	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.1s
    3176:	learn: 8.2339672	test: 13.7472013	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28.1s
    3177:	learn: 8.2332073	test: 13.7485712	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28s
    3178:	learn: 8.2323835	test: 13.7481548	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28s
    3179:	learn: 8.2306759	test: 13.7485272	best: 13.7379133 (3094)	total: 1m 48s	remaining: 28s
    3180:	learn: 8.2306900	test: 13.7489447	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.9s
    3181:	learn: 8.2300349	test: 13.7481691	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.9s
    3182:	learn: 8.2300350	test: 13.7470260	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.8s
    3183:	learn: 8.2291768	test: 13.7472765	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.8s
    3184:	learn: 8.2287819	test: 13.7484043	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.8s
    3185:	learn: 8.2278389	test: 13.7485466	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.7s
    3186:	learn: 8.2265560	test: 13.7466470	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.7s
    3187:	learn: 8.2251830	test: 13.7487556	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.7s
    3188:	learn: 8.2245909	test: 13.7478268	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.6s
    3189:	learn: 8.2241461	test: 13.7489340	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.6s
    3190:	learn: 8.2227138	test: 13.7479303	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.6s
    3191:	learn: 8.2220805	test: 13.7477056	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.5s
    3192:	learn: 8.2216445	test: 13.7473216	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.5s
    3193:	learn: 8.2209791	test: 13.7475532	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.5s
    3194:	learn: 8.2194980	test: 13.7464315	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.4s
    3195:	learn: 8.2176305	test: 13.7454162	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.4s
    3196:	learn: 8.2165881	test: 13.7449105	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.4s
    3197:	learn: 8.2159002	test: 13.7450656	best: 13.7379133 (3094)	total: 1m 48s	remaining: 27.3s
    3198:	learn: 8.2144281	test: 13.7437607	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27.3s
    3199:	learn: 8.2132112	test: 13.7427291	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27.3s
    3200:	learn: 8.2120594	test: 13.7415499	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27.2s
    3201:	learn: 8.2119964	test: 13.7421364	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27.2s
    3202:	learn: 8.2103978	test: 13.7420374	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27.2s
    3203:	learn: 8.2098406	test: 13.7417042	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27.1s
    3204:	learn: 8.2081644	test: 13.7404894	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27.1s
    3205:	learn: 8.2074502	test: 13.7409628	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27.1s
    3206:	learn: 8.2070934	test: 13.7395318	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27s
    3207:	learn: 8.2066902	test: 13.7398424	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27s
    3208:	learn: 8.2059753	test: 13.7392988	best: 13.7379133 (3094)	total: 1m 49s	remaining: 27s
    3209:	learn: 8.2049002	test: 13.7405387	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.9s
    3210:	learn: 8.2041168	test: 13.7411469	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.9s
    3211:	learn: 8.2038433	test: 13.7401991	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.9s
    3212:	learn: 8.2035787	test: 13.7406610	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.8s
    3213:	learn: 8.2021905	test: 13.7409327	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.8s
    3214:	learn: 8.2015508	test: 13.7415509	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.7s
    3215:	learn: 8.2005242	test: 13.7426482	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.7s
    3216:	learn: 8.2005572	test: 13.7425105	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.7s
    3217:	learn: 8.2003441	test: 13.7451006	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.6s
    3218:	learn: 8.1992909	test: 13.7424489	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.6s
    3219:	learn: 8.1987593	test: 13.7437911	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.6s
    3220:	learn: 8.1967234	test: 13.7449015	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.5s
    3221:	learn: 8.1954431	test: 13.7458224	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.5s
    3222:	learn: 8.1938456	test: 13.7465887	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.5s
    3223:	learn: 8.1935868	test: 13.7472562	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.4s
    3224:	learn: 8.1932709	test: 13.7477067	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.4s
    3225:	learn: 8.1922001	test: 13.7485007	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.4s
    3226:	learn: 8.1923046	test: 13.7483160	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.3s
    3227:	learn: 8.1914742	test: 13.7482892	best: 13.7379133 (3094)	total: 1m 49s	remaining: 26.3s
    3228:	learn: 8.1903668	test: 13.7490059	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26.3s
    3229:	learn: 8.1893376	test: 13.7481766	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26.2s
    3230:	learn: 8.1887443	test: 13.7468476	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26.2s
    3231:	learn: 8.1884439	test: 13.7467707	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26.2s
    3232:	learn: 8.1874918	test: 13.7459757	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26.1s
    3233:	learn: 8.1861833	test: 13.7462449	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26.1s
    3234:	learn: 8.1853451	test: 13.7444901	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26.1s
    3235:	learn: 8.1840564	test: 13.7462051	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26s
    3236:	learn: 8.1829710	test: 13.7463064	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26s
    3237:	learn: 8.1828660	test: 13.7478129	best: 13.7379133 (3094)	total: 1m 50s	remaining: 26s
    3238:	learn: 8.1813597	test: 13.7479443	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.9s
    3239:	learn: 8.1808869	test: 13.7487181	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.9s
    3240:	learn: 8.1804408	test: 13.7480536	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.9s
    3241:	learn: 8.1800268	test: 13.7485595	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.8s
    3242:	learn: 8.1796923	test: 13.7473795	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.8s
    3243:	learn: 8.1778774	test: 13.7469474	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.8s
    3244:	learn: 8.1772841	test: 13.7485221	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.7s
    3245:	learn: 8.1752582	test: 13.7484557	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.7s
    3246:	learn: 8.1753329	test: 13.7483246	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.7s
    3247:	learn: 8.1747122	test: 13.7481722	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.6s
    3248:	learn: 8.1739728	test: 13.7481140	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.6s
    3249:	learn: 8.1736841	test: 13.7487413	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.6s
    3250:	learn: 8.1730372	test: 13.7484003	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.5s
    3251:	learn: 8.1722561	test: 13.7487446	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.5s
    3252:	learn: 8.1718603	test: 13.7487246	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.5s
    3253:	learn: 8.1715982	test: 13.7485834	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.4s
    3254:	learn: 8.1706497	test: 13.7490856	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.4s
    3255:	learn: 8.1699945	test: 13.7502400	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.3s
    3256:	learn: 8.1689733	test: 13.7502558	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.3s
    3257:	learn: 8.1680822	test: 13.7508920	best: 13.7379133 (3094)	total: 1m 50s	remaining: 25.3s
    3258:	learn: 8.1672272	test: 13.7507971	best: 13.7379133 (3094)	total: 1m 51s	remaining: 25.2s
    3259:	learn: 8.1671891	test: 13.7506142	best: 13.7379133 (3094)	total: 1m 51s	remaining: 25.2s
    3260:	learn: 8.1664161	test: 13.7510243	best: 13.7379133 (3094)	total: 1m 51s	remaining: 25.2s
    3261:	learn: 8.1647063	test: 13.7513783	best: 13.7379133 (3094)	total: 1m 51s	remaining: 25.1s
    3262:	learn: 8.1641259	test: 13.7516064	best: 13.7379133 (3094)	total: 1m 51s	remaining: 25.1s
    3263:	learn: 8.1637473	test: 13.7511606	best: 13.7379133 (3094)	total: 1m 51s	remaining: 25.1s
    3264:	learn: 8.1636076	test: 13.7519365	best: 13.7379133 (3094)	total: 1m 51s	remaining: 25s
    3265:	learn: 8.1621232	test: 13.7528719	best: 13.7379133 (3094)	total: 1m 51s	remaining: 25s
    3266:	learn: 8.1615561	test: 13.7525478	best: 13.7379133 (3094)	total: 1m 51s	remaining: 25s
    3267:	learn: 8.1609726	test: 13.7522454	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.9s
    3268:	learn: 8.1601402	test: 13.7515430	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.9s
    3269:	learn: 8.1589574	test: 13.7531864	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.9s
    3270:	learn: 8.1580517	test: 13.7523014	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.8s
    3271:	learn: 8.1573476	test: 13.7522129	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.8s
    3272:	learn: 8.1570110	test: 13.7532267	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.8s
    3273:	learn: 8.1567653	test: 13.7527361	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.7s
    3274:	learn: 8.1558352	test: 13.7536183	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.7s
    3275:	learn: 8.1554951	test: 13.7533479	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.7s
    3276:	learn: 8.1550012	test: 13.7538326	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.6s
    3277:	learn: 8.1540962	test: 13.7546587	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.6s
    3278:	learn: 8.1529525	test: 13.7521438	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.6s
    3279:	learn: 8.1520442	test: 13.7521448	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.5s
    3280:	learn: 8.1516114	test: 13.7516166	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.5s
    3281:	learn: 8.1506679	test: 13.7520621	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.5s
    3282:	learn: 8.1500568	test: 13.7520702	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.4s
    3283:	learn: 8.1478677	test: 13.7518048	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.4s
    3284:	learn: 8.1480997	test: 13.7519515	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.4s
    3285:	learn: 8.1468253	test: 13.7522345	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.3s
    3286:	learn: 8.1462576	test: 13.7547085	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.3s
    3287:	learn: 8.1449835	test: 13.7556199	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.2s
    3288:	learn: 8.1443871	test: 13.7565441	best: 13.7379133 (3094)	total: 1m 51s	remaining: 24.2s
    3289:	learn: 8.1442425	test: 13.7559470	best: 13.7379133 (3094)	total: 1m 52s	remaining: 24.2s
    3290:	learn: 8.1441307	test: 13.7569450	best: 13.7379133 (3094)	total: 1m 52s	remaining: 24.1s
    3291:	learn: 8.1437642	test: 13.7551443	best: 13.7379133 (3094)	total: 1m 52s	remaining: 24.1s
    3292:	learn: 8.1433387	test: 13.7555679	best: 13.7379133 (3094)	total: 1m 52s	remaining: 24.1s
    3293:	learn: 8.1431213	test: 13.7561903	best: 13.7379133 (3094)	total: 1m 52s	remaining: 24s
    3294:	learn: 8.1419377	test: 13.7558548	best: 13.7379133 (3094)	total: 1m 52s	remaining: 24s
    3295:	learn: 8.1404174	test: 13.7566487	best: 13.7379133 (3094)	total: 1m 52s	remaining: 24s
    3296:	learn: 8.1397826	test: 13.7566825	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.9s
    3297:	learn: 8.1396248	test: 13.7575206	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.9s
    3298:	learn: 8.1376170	test: 13.7585469	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.9s
    3299:	learn: 8.1368033	test: 13.7582495	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.8s
    3300:	learn: 8.1367300	test: 13.7587982	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.8s
    3301:	learn: 8.1361855	test: 13.7577520	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.8s
    3302:	learn: 8.1352747	test: 13.7568276	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.7s
    3303:	learn: 8.1340175	test: 13.7584436	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.7s
    3304:	learn: 8.1334256	test: 13.7590520	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.6s
    3305:	learn: 8.1325625	test: 13.7604760	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.6s
    3306:	learn: 8.1319309	test: 13.7606973	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.6s
    3307:	learn: 8.1307774	test: 13.7589393	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.5s
    3308:	learn: 8.1297360	test: 13.7596794	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.5s
    3309:	learn: 8.1292311	test: 13.7590655	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.5s
    3310:	learn: 8.1281594	test: 13.7585037	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.4s
    3311:	learn: 8.1274158	test: 13.7581055	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.4s
    3312:	learn: 8.1266621	test: 13.7588788	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.4s
    3313:	learn: 8.1263792	test: 13.7578383	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.3s
    3314:	learn: 8.1253840	test: 13.7588947	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.3s
    3315:	learn: 8.1252557	test: 13.7593841	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.3s
    3316:	learn: 8.1235559	test: 13.7592215	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.2s
    3317:	learn: 8.1230812	test: 13.7588228	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.2s
    3318:	learn: 8.1232151	test: 13.7587958	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.2s
    3319:	learn: 8.1226560	test: 13.7581360	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.1s
    3320:	learn: 8.1223207	test: 13.7585695	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.1s
    3321:	learn: 8.1220410	test: 13.7587759	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23.1s
    3322:	learn: 8.1218671	test: 13.7588659	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23s
    3323:	learn: 8.1212909	test: 13.7596634	best: 13.7379133 (3094)	total: 1m 52s	remaining: 23s
    3324:	learn: 8.1206292	test: 13.7591071	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.9s
    3325:	learn: 8.1197060	test: 13.7592516	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.9s
    3326:	learn: 8.1197513	test: 13.7589384	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.9s
    3327:	learn: 8.1190084	test: 13.7581620	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.8s
    3328:	learn: 8.1179111	test: 13.7594101	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.8s
    3329:	learn: 8.1168988	test: 13.7603328	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.8s
    3330:	learn: 8.1164726	test: 13.7607083	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.7s
    3331:	learn: 8.1159152	test: 13.7618602	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.7s
    3332:	learn: 8.1154218	test: 13.7619053	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.7s
    3333:	learn: 8.1151413	test: 13.7622178	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.6s
    3334:	learn: 8.1133767	test: 13.7631860	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.6s
    3335:	learn: 8.1130170	test: 13.7632431	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.6s
    3336:	learn: 8.1125162	test: 13.7645278	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.5s
    3337:	learn: 8.1118477	test: 13.7646484	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.5s
    3338:	learn: 8.1113623	test: 13.7639686	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.5s
    3339:	learn: 8.1109049	test: 13.7647059	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.4s
    3340:	learn: 8.1099427	test: 13.7651129	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.4s
    3341:	learn: 8.1092889	test: 13.7661673	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.4s
    3342:	learn: 8.1086624	test: 13.7652158	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.3s
    3343:	learn: 8.1070454	test: 13.7646378	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.3s
    3344:	learn: 8.1069426	test: 13.7643350	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.2s
    3345:	learn: 8.1065179	test: 13.7643666	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.2s
    3346:	learn: 8.1059349	test: 13.7646972	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.2s
    3347:	learn: 8.1054663	test: 13.7649224	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.1s
    3348:	learn: 8.1044578	test: 13.7662116	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.1s
    3349:	learn: 8.1032970	test: 13.7676136	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22.1s
    3350:	learn: 8.1024323	test: 13.7683620	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22s
    3351:	learn: 8.1017984	test: 13.7696462	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22s
    3352:	learn: 8.1011877	test: 13.7681162	best: 13.7379133 (3094)	total: 1m 53s	remaining: 22s
    3353:	learn: 8.1009971	test: 13.7669731	best: 13.7379133 (3094)	total: 1m 53s	remaining: 21.9s
    3354:	learn: 8.1001883	test: 13.7658885	best: 13.7379133 (3094)	total: 1m 53s	remaining: 21.9s
    3355:	learn: 8.0988460	test: 13.7636801	best: 13.7379133 (3094)	total: 1m 53s	remaining: 21.9s
    3356:	learn: 8.0974953	test: 13.7628516	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.8s
    3357:	learn: 8.0962387	test: 13.7625502	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.8s
    3358:	learn: 8.0955365	test: 13.7637447	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.8s
    3359:	learn: 8.0949715	test: 13.7639217	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.7s
    3360:	learn: 8.0945000	test: 13.7631576	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.7s
    3361:	learn: 8.0938732	test: 13.7631875	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.7s
    3362:	learn: 8.0931046	test: 13.7624346	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.6s
    3363:	learn: 8.0925991	test: 13.7628084	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.6s
    3364:	learn: 8.0924044	test: 13.7636511	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.6s
    3365:	learn: 8.0916244	test: 13.7635304	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.5s
    3366:	learn: 8.0912084	test: 13.7645693	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.5s
    3367:	learn: 8.0904018	test: 13.7646697	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.5s
    3368:	learn: 8.0889053	test: 13.7641581	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.4s
    3369:	learn: 8.0877183	test: 13.7653316	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.4s
    3370:	learn: 8.0860409	test: 13.7653919	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.3s
    3371:	learn: 8.0848385	test: 13.7651926	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.3s
    3372:	learn: 8.0839141	test: 13.7669711	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.3s
    3373:	learn: 8.0833919	test: 13.7684699	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.2s
    3374:	learn: 8.0824404	test: 13.7692754	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.2s
    3375:	learn: 8.0809772	test: 13.7713976	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.2s
    3376:	learn: 8.0803849	test: 13.7708790	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.1s
    3377:	learn: 8.0798888	test: 13.7704040	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.1s
    3378:	learn: 8.0789813	test: 13.7693821	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21.1s
    3379:	learn: 8.0784269	test: 13.7692855	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21s
    3380:	learn: 8.0783003	test: 13.7717510	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21s
    3381:	learn: 8.0777692	test: 13.7723068	best: 13.7379133 (3094)	total: 1m 54s	remaining: 21s
    3382:	learn: 8.0762751	test: 13.7709566	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.9s
    3383:	learn: 8.0758058	test: 13.7701128	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.9s
    3384:	learn: 8.0749379	test: 13.7704070	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.9s
    3385:	learn: 8.0750878	test: 13.7706717	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.8s
    3386:	learn: 8.0731097	test: 13.7696531	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.8s
    3387:	learn: 8.0724325	test: 13.7698668	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.7s
    3388:	learn: 8.0716633	test: 13.7688997	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.7s
    3389:	learn: 8.0708492	test: 13.7667562	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.7s
    3390:	learn: 8.0704628	test: 13.7668224	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.6s
    3391:	learn: 8.0698011	test: 13.7666726	best: 13.7379133 (3094)	total: 1m 54s	remaining: 20.6s
    3392:	learn: 8.0687592	test: 13.7665526	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.6s
    3393:	learn: 8.0683933	test: 13.7663771	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.5s
    3394:	learn: 8.0664804	test: 13.7663830	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.5s
    3395:	learn: 8.0659749	test: 13.7657276	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.5s
    3396:	learn: 8.0657327	test: 13.7661558	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.4s
    3397:	learn: 8.0638788	test: 13.7673802	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.4s
    3398:	learn: 8.0630337	test: 13.7669171	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.4s
    3399:	learn: 8.0631219	test: 13.7667521	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.3s
    3400:	learn: 8.0627099	test: 13.7657403	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.3s
    3401:	learn: 8.0623714	test: 13.7651915	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.3s
    3402:	learn: 8.0621901	test: 13.7655663	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.2s
    3403:	learn: 8.0610042	test: 13.7648586	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.2s
    3404:	learn: 8.0603400	test: 13.7655637	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.2s
    3405:	learn: 8.0594788	test: 13.7667414	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.1s
    3406:	learn: 8.0593796	test: 13.7687245	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.1s
    3407:	learn: 8.0579553	test: 13.7689508	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20.1s
    3408:	learn: 8.0575977	test: 13.7675344	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20s
    3409:	learn: 8.0570461	test: 13.7670913	best: 13.7379133 (3094)	total: 1m 55s	remaining: 20s
    3410:	learn: 8.0562739	test: 13.7668631	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.9s
    3411:	learn: 8.0557774	test: 13.7672223	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.9s
    3412:	learn: 8.0558376	test: 13.7665519	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.9s
    3413:	learn: 8.0557180	test: 13.7669268	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.8s
    3414:	learn: 8.0549112	test: 13.7674026	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.8s
    3415:	learn: 8.0544462	test: 13.7655511	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.8s
    3416:	learn: 8.0526879	test: 13.7627232	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.7s
    3417:	learn: 8.0522116	test: 13.7627602	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.7s
    3418:	learn: 8.0516654	test: 13.7626801	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.7s
    3419:	learn: 8.0506902	test: 13.7642345	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.6s
    3420:	learn: 8.0492888	test: 13.7664894	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.6s
    3421:	learn: 8.0487431	test: 13.7669640	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.6s
    3422:	learn: 8.0473895	test: 13.7684931	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.5s
    3423:	learn: 8.0461456	test: 13.7686835	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.5s
    3424:	learn: 8.0453467	test: 13.7692028	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.5s
    3425:	learn: 8.0448695	test: 13.7682202	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.4s
    3426:	learn: 8.0440700	test: 13.7690235	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.4s
    3427:	learn: 8.0431910	test: 13.7686865	best: 13.7379133 (3094)	total: 1m 55s	remaining: 19.4s
    3428:	learn: 8.0420974	test: 13.7702669	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19.3s
    3429:	learn: 8.0418522	test: 13.7715345	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19.3s
    3430:	learn: 8.0414876	test: 13.7718786	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19.3s
    3431:	learn: 8.0411331	test: 13.7724822	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19.2s
    3432:	learn: 8.0402886	test: 13.7724107	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19.2s
    3433:	learn: 8.0394253	test: 13.7734184	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19.1s
    3434:	learn: 8.0377306	test: 13.7752768	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19.1s
    3435:	learn: 8.0361342	test: 13.7767815	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19.1s
    3436:	learn: 8.0355534	test: 13.7769722	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19s
    3437:	learn: 8.0350532	test: 13.7771763	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19s
    3438:	learn: 8.0344003	test: 13.7774177	best: 13.7379133 (3094)	total: 1m 56s	remaining: 19s
    3439:	learn: 8.0335108	test: 13.7778164	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.9s
    3440:	learn: 8.0326204	test: 13.7795008	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.9s
    3441:	learn: 8.0315015	test: 13.7778338	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.9s
    3442:	learn: 8.0305445	test: 13.7778483	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.8s
    3443:	learn: 8.0300395	test: 13.7768842	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.8s
    3444:	learn: 8.0295717	test: 13.7766301	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.8s
    3445:	learn: 8.0291395	test: 13.7758858	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.7s
    3446:	learn: 8.0287467	test: 13.7764851	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.7s
    3447:	learn: 8.0278234	test: 13.7747119	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.7s
    3448:	learn: 8.0264356	test: 13.7743845	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.6s
    3449:	learn: 8.0259058	test: 13.7736149	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.6s
    3450:	learn: 8.0252263	test: 13.7743016	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.6s
    3451:	learn: 8.0247216	test: 13.7733621	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.5s
    3452:	learn: 8.0241926	test: 13.7738727	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.5s
    3453:	learn: 8.0233371	test: 13.7726909	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.5s
    3454:	learn: 8.0224562	test: 13.7742463	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.4s
    3455:	learn: 8.0219172	test: 13.7747538	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.4s
    3456:	learn: 8.0213628	test: 13.7751923	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.4s
    3457:	learn: 8.0200573	test: 13.7760663	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.3s
    3458:	learn: 8.0198676	test: 13.7752884	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.3s
    3459:	learn: 8.0189079	test: 13.7759039	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.2s
    3460:	learn: 8.0174892	test: 13.7759544	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.2s
    3461:	learn: 8.0172518	test: 13.7766447	best: 13.7379133 (3094)	total: 1m 56s	remaining: 18.2s
    3462:	learn: 8.0162275	test: 13.7760644	best: 13.7379133 (3094)	total: 1m 57s	remaining: 18.1s
    3463:	learn: 8.0160859	test: 13.7760241	best: 13.7379133 (3094)	total: 1m 57s	remaining: 18.1s
    3464:	learn: 8.0148065	test: 13.7768837	best: 13.7379133 (3094)	total: 1m 57s	remaining: 18.1s
    3465:	learn: 8.0140742	test: 13.7768189	best: 13.7379133 (3094)	total: 1m 57s	remaining: 18s
    3466:	learn: 8.0133245	test: 13.7777249	best: 13.7379133 (3094)	total: 1m 57s	remaining: 18s
    3467:	learn: 8.0129299	test: 13.7770216	best: 13.7379133 (3094)	total: 1m 57s	remaining: 18s
    3468:	learn: 8.0112055	test: 13.7759015	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.9s
    3469:	learn: 8.0101786	test: 13.7762638	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.9s
    3470:	learn: 8.0098897	test: 13.7755342	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.9s
    3471:	learn: 8.0086499	test: 13.7759749	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.8s
    3472:	learn: 8.0078258	test: 13.7776608	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.8s
    3473:	learn: 8.0069006	test: 13.7783471	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.8s
    3474:	learn: 8.0064250	test: 13.7791212	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.7s
    3475:	learn: 8.0055523	test: 13.7795772	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.7s
    3476:	learn: 8.0047512	test: 13.7801558	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.7s
    3477:	learn: 8.0039657	test: 13.7794227	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.6s
    3478:	learn: 8.0035302	test: 13.7789318	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.6s
    3479:	learn: 8.0033932	test: 13.7788210	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.6s
    3480:	learn: 8.0033966	test: 13.7782694	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.5s
    3481:	learn: 8.0032148	test: 13.7787416	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.5s
    3482:	learn: 8.0023159	test: 13.7782580	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.5s
    3483:	learn: 8.0016821	test: 13.7784116	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.4s
    3484:	learn: 8.0011314	test: 13.7791591	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.4s
    3485:	learn: 8.0007300	test: 13.7784598	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.4s
    3486:	learn: 8.0001832	test: 13.7792394	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.3s
    3487:	learn: 7.9997108	test: 13.7806137	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.3s
    3488:	learn: 7.9993997	test: 13.7801315	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.3s
    3489:	learn: 7.9986937	test: 13.7803134	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.2s
    3490:	learn: 7.9982676	test: 13.7798877	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.2s
    3491:	learn: 7.9980717	test: 13.7784839	best: 13.7379133 (3094)	total: 1m 57s	remaining: 17.2s
    3492:	learn: 7.9979837	test: 13.7790845	best: 13.7379133 (3094)	total: 1m 58s	remaining: 17.1s
    3493:	learn: 7.9974986	test: 13.7792910	best: 13.7379133 (3094)	total: 1m 58s	remaining: 17.1s
    3494:	learn: 7.9966364	test: 13.7774089	best: 13.7379133 (3094)	total: 1m 58s	remaining: 17.1s
    3495:	learn: 7.9950877	test: 13.7760507	best: 13.7379133 (3094)	total: 1m 58s	remaining: 17s
    3496:	learn: 7.9947003	test: 13.7769058	best: 13.7379133 (3094)	total: 1m 58s	remaining: 17s
    3497:	learn: 7.9939319	test: 13.7772469	best: 13.7379133 (3094)	total: 1m 58s	remaining: 17s
    3498:	learn: 7.9929950	test: 13.7759458	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.9s
    3499:	learn: 7.9925371	test: 13.7758759	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.9s
    3500:	learn: 7.9917811	test: 13.7762588	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.9s
    3501:	learn: 7.9912023	test: 13.7752814	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.8s
    3502:	learn: 7.9901583	test: 13.7760898	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.8s
    3503:	learn: 7.9887197	test: 13.7758702	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.8s
    3504:	learn: 7.9883599	test: 13.7766351	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.7s
    3505:	learn: 7.9878790	test: 13.7761944	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.7s
    3506:	learn: 7.9870577	test: 13.7752169	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.7s
    3507:	learn: 7.9860692	test: 13.7746384	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.6s
    3508:	learn: 7.9859996	test: 13.7750604	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.6s
    3509:	learn: 7.9859691	test: 13.7755801	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.6s
    3510:	learn: 7.9851052	test: 13.7755394	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.5s
    3511:	learn: 7.9844390	test: 13.7750401	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.5s
    3512:	learn: 7.9837265	test: 13.7755192	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.4s
    3513:	learn: 7.9833716	test: 13.7751704	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.4s
    3514:	learn: 7.9830494	test: 13.7742214	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.4s
    3515:	learn: 7.9825374	test: 13.7732452	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.3s
    3516:	learn: 7.9798704	test: 13.7721643	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.3s
    3517:	learn: 7.9793727	test: 13.7743629	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.3s
    3518:	learn: 7.9792108	test: 13.7743981	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.2s
    3519:	learn: 7.9788626	test: 13.7740908	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.2s
    3520:	learn: 7.9784095	test: 13.7744523	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.2s
    3521:	learn: 7.9782054	test: 13.7740871	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.1s
    3522:	learn: 7.9768910	test: 13.7720621	best: 13.7379133 (3094)	total: 1m 58s	remaining: 16.1s
    3523:	learn: 7.9764794	test: 13.7713860	best: 13.7379133 (3094)	total: 1m 59s	remaining: 16.1s
    3524:	learn: 7.9763528	test: 13.7706432	best: 13.7379133 (3094)	total: 1m 59s	remaining: 16s
    3525:	learn: 7.9755778	test: 13.7720530	best: 13.7379133 (3094)	total: 1m 59s	remaining: 16s
    3526:	learn: 7.9752881	test: 13.7725583	best: 13.7379133 (3094)	total: 1m 59s	remaining: 16s
    3527:	learn: 7.9749053	test: 13.7734130	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.9s
    3528:	learn: 7.9728496	test: 13.7742428	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.9s
    3529:	learn: 7.9725932	test: 13.7745667	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.9s
    3530:	learn: 7.9715283	test: 13.7730470	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.8s
    3531:	learn: 7.9702365	test: 13.7711945	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.8s
    3532:	learn: 7.9696144	test: 13.7717776	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.8s
    3533:	learn: 7.9689552	test: 13.7717111	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.7s
    3534:	learn: 7.9685109	test: 13.7718083	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.7s
    3535:	learn: 7.9679368	test: 13.7719760	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.7s
    3536:	learn: 7.9674262	test: 13.7719305	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.6s
    3537:	learn: 7.9666717	test: 13.7703212	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.6s
    3538:	learn: 7.9661508	test: 13.7703904	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.6s
    3539:	learn: 7.9654555	test: 13.7708167	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.5s
    3540:	learn: 7.9646804	test: 13.7707491	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.5s
    3541:	learn: 7.9641287	test: 13.7711167	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.5s
    3542:	learn: 7.9638164	test: 13.7715789	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.4s
    3543:	learn: 7.9632940	test: 13.7709220	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.4s
    3544:	learn: 7.9625664	test: 13.7711972	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.4s
    3545:	learn: 7.9620087	test: 13.7699297	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.3s
    3546:	learn: 7.9616306	test: 13.7699655	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.3s
    3547:	learn: 7.9619009	test: 13.7701682	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.3s
    3548:	learn: 7.9611702	test: 13.7709968	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.2s
    3549:	learn: 7.9588727	test: 13.7733257	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.2s
    3550:	learn: 7.9584896	test: 13.7726607	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.2s
    3551:	learn: 7.9582430	test: 13.7719362	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.1s
    3552:	learn: 7.9562874	test: 13.7703332	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.1s
    3553:	learn: 7.9555538	test: 13.7689374	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15.1s
    3554:	learn: 7.9547533	test: 13.7694146	best: 13.7379133 (3094)	total: 1m 59s	remaining: 15s
    3555:	learn: 7.9542805	test: 13.7684456	best: 13.7379133 (3094)	total: 2m	remaining: 15s
    3556:	learn: 7.9538494	test: 13.7701384	best: 13.7379133 (3094)	total: 2m	remaining: 15s
    3557:	learn: 7.9531346	test: 13.7703935	best: 13.7379133 (3094)	total: 2m	remaining: 14.9s
    3558:	learn: 7.9528147	test: 13.7696028	best: 13.7379133 (3094)	total: 2m	remaining: 14.9s
    3559:	learn: 7.9520316	test: 13.7693457	best: 13.7379133 (3094)	total: 2m	remaining: 14.9s
    3560:	learn: 7.9515184	test: 13.7694324	best: 13.7379133 (3094)	total: 2m	remaining: 14.8s
    3561:	learn: 7.9508354	test: 13.7694028	best: 13.7379133 (3094)	total: 2m	remaining: 14.8s
    3562:	learn: 7.9501098	test: 13.7690816	best: 13.7379133 (3094)	total: 2m	remaining: 14.8s
    3563:	learn: 7.9494188	test: 13.7686815	best: 13.7379133 (3094)	total: 2m	remaining: 14.7s
    3564:	learn: 7.9493266	test: 13.7689647	best: 13.7379133 (3094)	total: 2m	remaining: 14.7s
    3565:	learn: 7.9488388	test: 13.7686315	best: 13.7379133 (3094)	total: 2m	remaining: 14.7s
    3566:	learn: 7.9475266	test: 13.7689503	best: 13.7379133 (3094)	total: 2m	remaining: 14.6s
    3567:	learn: 7.9465722	test: 13.7674389	best: 13.7379133 (3094)	total: 2m	remaining: 14.6s
    3568:	learn: 7.9459359	test: 13.7684898	best: 13.7379133 (3094)	total: 2m	remaining: 14.5s
    3569:	learn: 7.9453096	test: 13.7694155	best: 13.7379133 (3094)	total: 2m	remaining: 14.5s
    3570:	learn: 7.9448468	test: 13.7689862	best: 13.7379133 (3094)	total: 2m	remaining: 14.5s
    3571:	learn: 7.9441150	test: 13.7691762	best: 13.7379133 (3094)	total: 2m	remaining: 14.4s
    3572:	learn: 7.9435198	test: 13.7688480	best: 13.7379133 (3094)	total: 2m	remaining: 14.4s
    3573:	learn: 7.9425122	test: 13.7699570	best: 13.7379133 (3094)	total: 2m	remaining: 14.4s
    3574:	learn: 7.9419271	test: 13.7698530	best: 13.7379133 (3094)	total: 2m	remaining: 14.3s
    3575:	learn: 7.9416894	test: 13.7686125	best: 13.7379133 (3094)	total: 2m	remaining: 14.3s
    3576:	learn: 7.9415368	test: 13.7682256	best: 13.7379133 (3094)	total: 2m	remaining: 14.3s
    3577:	learn: 7.9413143	test: 13.7685303	best: 13.7379133 (3094)	total: 2m	remaining: 14.2s
    3578:	learn: 7.9410136	test: 13.7698206	best: 13.7379133 (3094)	total: 2m	remaining: 14.2s
    3579:	learn: 7.9408908	test: 13.7698205	best: 13.7379133 (3094)	total: 2m	remaining: 14.2s
    3580:	learn: 7.9407215	test: 13.7697399	best: 13.7379133 (3094)	total: 2m	remaining: 14.1s
    3581:	learn: 7.9409979	test: 13.7707667	best: 13.7379133 (3094)	total: 2m	remaining: 14.1s
    3582:	learn: 7.9384297	test: 13.7697891	best: 13.7379133 (3094)	total: 2m	remaining: 14.1s
    3583:	learn: 7.9368108	test: 13.7718427	best: 13.7379133 (3094)	total: 2m	remaining: 14s
    3584:	learn: 7.9359451	test: 13.7725703	best: 13.7379133 (3094)	total: 2m 1s	remaining: 14s
    3585:	learn: 7.9348782	test: 13.7732309	best: 13.7379133 (3094)	total: 2m 1s	remaining: 14s
    3586:	learn: 7.9341960	test: 13.7737193	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.9s
    3587:	learn: 7.9337374	test: 13.7746114	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.9s
    3588:	learn: 7.9329943	test: 13.7760898	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.9s
    3589:	learn: 7.9332716	test: 13.7764638	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.8s
    3590:	learn: 7.9327399	test: 13.7745582	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.8s
    3591:	learn: 7.9326373	test: 13.7763436	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.8s
    3592:	learn: 7.9321678	test: 13.7768265	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.7s
    3593:	learn: 7.9313976	test: 13.7755984	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.7s
    3594:	learn: 7.9306931	test: 13.7741564	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.7s
    3595:	learn: 7.9299398	test: 13.7739653	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.6s
    3596:	learn: 7.9290047	test: 13.7731913	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.6s
    3597:	learn: 7.9283057	test: 13.7714564	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.6s
    3598:	learn: 7.9279729	test: 13.7710426	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.5s
    3599:	learn: 7.9268578	test: 13.7716518	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.5s
    3600:	learn: 7.9262308	test: 13.7717724	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.5s
    3601:	learn: 7.9242849	test: 13.7713840	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.4s
    3602:	learn: 7.9235111	test: 13.7716359	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.4s
    3603:	learn: 7.9231960	test: 13.7722182	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.4s
    3604:	learn: 7.9224895	test: 13.7715644	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.3s
    3605:	learn: 7.9220373	test: 13.7712570	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.3s
    3606:	learn: 7.9215281	test: 13.7718409	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.3s
    3607:	learn: 7.9212436	test: 13.7724474	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.2s
    3608:	learn: 7.9207358	test: 13.7718527	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.2s
    3609:	learn: 7.9198241	test: 13.7734925	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.2s
    3610:	learn: 7.9191519	test: 13.7731140	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.1s
    3611:	learn: 7.9184389	test: 13.7746423	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.1s
    3612:	learn: 7.9175451	test: 13.7744824	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13.1s
    3613:	learn: 7.9170526	test: 13.7741834	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13s
    3614:	learn: 7.9163593	test: 13.7762271	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13s
    3615:	learn: 7.9156757	test: 13.7760385	best: 13.7379133 (3094)	total: 2m 1s	remaining: 13s
    3616:	learn: 7.9151654	test: 13.7759104	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.9s
    3617:	learn: 7.9148912	test: 13.7757566	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.9s
    3618:	learn: 7.9131152	test: 13.7753732	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.9s
    3619:	learn: 7.9126016	test: 13.7737582	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.8s
    3620:	learn: 7.9111394	test: 13.7728104	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.8s
    3621:	learn: 7.9107694	test: 13.7728104	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.7s
    3622:	learn: 7.9097345	test: 13.7726233	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.7s
    3623:	learn: 7.9085332	test: 13.7727625	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.7s
    3624:	learn: 7.9083854	test: 13.7727522	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.6s
    3625:	learn: 7.9065860	test: 13.7722677	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.6s
    3626:	learn: 7.9060872	test: 13.7722377	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.6s
    3627:	learn: 7.9060147	test: 13.7714189	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.5s
    3628:	learn: 7.9053757	test: 13.7725862	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.5s
    3629:	learn: 7.9047662	test: 13.7716943	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.5s
    3630:	learn: 7.9045672	test: 13.7721119	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.4s
    3631:	learn: 7.9031941	test: 13.7721165	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.4s
    3632:	learn: 7.9014181	test: 13.7724654	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.4s
    3633:	learn: 7.9010086	test: 13.7741246	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.3s
    3634:	learn: 7.9007754	test: 13.7738382	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.3s
    3635:	learn: 7.8989117	test: 13.7693830	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.3s
    3636:	learn: 7.8985628	test: 13.7691761	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.2s
    3637:	learn: 7.8964427	test: 13.7700923	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.2s
    3638:	learn: 7.8966015	test: 13.7704440	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.2s
    3639:	learn: 7.8959520	test: 13.7707746	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.1s
    3640:	learn: 7.8954448	test: 13.7703548	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.1s
    3641:	learn: 7.8948891	test: 13.7704506	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12.1s
    3642:	learn: 7.8942859	test: 13.7702491	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12s
    3643:	learn: 7.8938732	test: 13.7697155	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12s
    3644:	learn: 7.8930117	test: 13.7690441	best: 13.7379133 (3094)	total: 2m 2s	remaining: 12s
    3645:	learn: 7.8924247	test: 13.7699059	best: 13.7379133 (3094)	total: 2m 2s	remaining: 11.9s
    3646:	learn: 7.8915874	test: 13.7689997	best: 13.7379133 (3094)	total: 2m 2s	remaining: 11.9s
    3647:	learn: 7.8908295	test: 13.7701794	best: 13.7379133 (3094)	total: 2m 2s	remaining: 11.9s
    3648:	learn: 7.8896427	test: 13.7694604	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.8s
    3649:	learn: 7.8891198	test: 13.7698781	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.8s
    3650:	learn: 7.8881330	test: 13.7694381	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.8s
    3651:	learn: 7.8871812	test: 13.7697543	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.7s
    3652:	learn: 7.8870471	test: 13.7684296	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.7s
    3653:	learn: 7.8862684	test: 13.7671371	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.7s
    3654:	learn: 7.8857866	test: 13.7666499	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.6s
    3655:	learn: 7.8847676	test: 13.7678244	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.6s
    3656:	learn: 7.8842513	test: 13.7686935	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.6s
    3657:	learn: 7.8841775	test: 13.7697427	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.5s
    3658:	learn: 7.8832100	test: 13.7724181	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.5s
    3659:	learn: 7.8829021	test: 13.7730078	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.5s
    3660:	learn: 7.8826921	test: 13.7731482	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.4s
    3661:	learn: 7.8812826	test: 13.7732308	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.4s
    3662:	learn: 7.8801219	test: 13.7733995	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.4s
    3663:	learn: 7.8791240	test: 13.7751656	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.3s
    3664:	learn: 7.8782488	test: 13.7736163	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.3s
    3665:	learn: 7.8781604	test: 13.7730874	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.3s
    3666:	learn: 7.8769584	test: 13.7728315	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.2s
    3667:	learn: 7.8762538	test: 13.7730609	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.2s
    3668:	learn: 7.8758077	test: 13.7727060	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.1s
    3669:	learn: 7.8756084	test: 13.7727282	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.1s
    3670:	learn: 7.8745580	test: 13.7734057	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11.1s
    3671:	learn: 7.8741402	test: 13.7726647	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11s
    3672:	learn: 7.8740184	test: 13.7725333	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11s
    3673:	learn: 7.8734302	test: 13.7720868	best: 13.7379133 (3094)	total: 2m 3s	remaining: 11s
    3674:	learn: 7.8732474	test: 13.7740194	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.9s
    3675:	learn: 7.8719892	test: 13.7742287	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.9s
    3676:	learn: 7.8715519	test: 13.7735343	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.9s
    3677:	learn: 7.8711568	test: 13.7757179	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.8s
    3678:	learn: 7.8705901	test: 13.7746224	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.8s
    3679:	learn: 7.8699410	test: 13.7751094	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.8s
    3680:	learn: 7.8698628	test: 13.7754852	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.7s
    3681:	learn: 7.8696873	test: 13.7759928	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.7s
    3682:	learn: 7.8676999	test: 13.7754481	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.7s
    3683:	learn: 7.8670396	test: 13.7752383	best: 13.7379133 (3094)	total: 2m 3s	remaining: 10.6s
    3684:	learn: 7.8667456	test: 13.7752434	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.6s
    3685:	learn: 7.8657241	test: 13.7766568	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.6s
    3686:	learn: 7.8653505	test: 13.7767625	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.5s
    3687:	learn: 7.8649795	test: 13.7770211	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.5s
    3688:	learn: 7.8632602	test: 13.7770474	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.5s
    3689:	learn: 7.8629645	test: 13.7766207	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.4s
    3690:	learn: 7.8626081	test: 13.7764904	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.4s
    3691:	learn: 7.8622602	test: 13.7761525	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.4s
    3692:	learn: 7.8622639	test: 13.7758488	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.3s
    3693:	learn: 7.8619019	test: 13.7759048	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.3s
    3694:	learn: 7.8614890	test: 13.7761260	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.3s
    3695:	learn: 7.8592107	test: 13.7740982	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.2s
    3696:	learn: 7.8590430	test: 13.7733838	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.2s
    3697:	learn: 7.8575602	test: 13.7741146	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.2s
    3698:	learn: 7.8576582	test: 13.7738423	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.1s
    3699:	learn: 7.8571369	test: 13.7742212	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.1s
    3700:	learn: 7.8564262	test: 13.7752333	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10.1s
    3701:	learn: 7.8552958	test: 13.7752211	best: 13.7379133 (3094)	total: 2m 4s	remaining: 10s
    3702:	learn: 7.8546153	test: 13.7757870	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.99s
    3703:	learn: 7.8527747	test: 13.7743028	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.95s
    3704:	learn: 7.8510730	test: 13.7712311	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.92s
    3705:	learn: 7.8508300	test: 13.7700714	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.88s
    3706:	learn: 7.8505587	test: 13.7697392	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.85s
    3707:	learn: 7.8496642	test: 13.7691939	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.82s
    3708:	learn: 7.8486100	test: 13.7690977	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.78s
    3709:	learn: 7.8478515	test: 13.7689600	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.75s
    3710:	learn: 7.8474535	test: 13.7700802	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.72s
    3711:	learn: 7.8464354	test: 13.7706093	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.68s
    3712:	learn: 7.8464110	test: 13.7703747	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.65s
    3713:	learn: 7.8460804	test: 13.7735368	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.62s
    3714:	learn: 7.8452263	test: 13.7725828	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.58s
    3715:	learn: 7.8436804	test: 13.7733158	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.55s
    3716:	learn: 7.8433855	test: 13.7738900	best: 13.7379133 (3094)	total: 2m 4s	remaining: 9.52s
    3717:	learn: 7.8427893	test: 13.7740564	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.48s
    3718:	learn: 7.8422262	test: 13.7736641	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.45s
    3719:	learn: 7.8415364	test: 13.7730501	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.41s
    3720:	learn: 7.8414463	test: 13.7723770	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.38s
    3721:	learn: 7.8406904	test: 13.7729944	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.35s
    3722:	learn: 7.8403149	test: 13.7715151	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.31s
    3723:	learn: 7.8395095	test: 13.7714993	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.28s
    3724:	learn: 7.8386425	test: 13.7720604	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.24s
    3725:	learn: 7.8381223	test: 13.7717347	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.21s
    3726:	learn: 7.8377832	test: 13.7727988	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.18s
    3727:	learn: 7.8377006	test: 13.7725512	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.14s
    3728:	learn: 7.8370495	test: 13.7734047	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.11s
    3729:	learn: 7.8368754	test: 13.7729765	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.08s
    3730:	learn: 7.8364403	test: 13.7721661	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.04s
    3731:	learn: 7.8361353	test: 13.7715001	best: 13.7379133 (3094)	total: 2m 5s	remaining: 9.01s
    3732:	learn: 7.8352373	test: 13.7719806	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.97s
    3733:	learn: 7.8345386	test: 13.7721853	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.94s
    3734:	learn: 7.8337891	test: 13.7731840	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.91s
    3735:	learn: 7.8336097	test: 13.7721024	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.87s
    3736:	learn: 7.8331133	test: 13.7721266	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.84s
    3737:	learn: 7.8321738	test: 13.7717266	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.81s
    3738:	learn: 7.8322375	test: 13.7711300	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.77s
    3739:	learn: 7.8318224	test: 13.7720108	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.74s
    3740:	learn: 7.8308043	test: 13.7726361	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.71s
    3741:	learn: 7.8300682	test: 13.7716398	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.67s
    3742:	learn: 7.8295912	test: 13.7705818	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.64s
    3743:	learn: 7.8291694	test: 13.7705391	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.6s
    3744:	learn: 7.8287073	test: 13.7703740	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.57s
    3745:	learn: 7.8277332	test: 13.7685761	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.54s
    3746:	learn: 7.8273827	test: 13.7688487	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.5s
    3747:	learn: 7.8264937	test: 13.7682483	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.47s
    3748:	learn: 7.8256040	test: 13.7692296	best: 13.7379133 (3094)	total: 2m 5s	remaining: 8.44s
    3749:	learn: 7.8252925	test: 13.7688006	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.4s
    3750:	learn: 7.8244982	test: 13.7689071	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.37s
    3751:	learn: 7.8236762	test: 13.7674246	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.33s
    3752:	learn: 7.8234373	test: 13.7680651	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.3s
    3753:	learn: 7.8229644	test: 13.7671278	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.27s
    3754:	learn: 7.8227418	test: 13.7668840	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.23s
    3755:	learn: 7.8216479	test: 13.7693064	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.2s
    3756:	learn: 7.8216161	test: 13.7683973	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.17s
    3757:	learn: 7.8209973	test: 13.7684481	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.13s
    3758:	learn: 7.8206940	test: 13.7682021	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.1s
    3759:	learn: 7.8201928	test: 13.7690466	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.06s
    3760:	learn: 7.8197578	test: 13.7688923	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8.03s
    3761:	learn: 7.8191548	test: 13.7669657	best: 13.7379133 (3094)	total: 2m 6s	remaining: 8s
    3762:	learn: 7.8180496	test: 13.7683183	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.96s
    3763:	learn: 7.8169802	test: 13.7676402	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.93s
    3764:	learn: 7.8152732	test: 13.7687960	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.9s
    3765:	learn: 7.8151016	test: 13.7688199	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.86s
    3766:	learn: 7.8139712	test: 13.7672331	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.83s
    3767:	learn: 7.8136766	test: 13.7669871	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.79s
    3768:	learn: 7.8130775	test: 13.7665451	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.76s
    3769:	learn: 7.8128591	test: 13.7665318	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.73s
    3770:	learn: 7.8126299	test: 13.7659015	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.69s
    3771:	learn: 7.8125930	test: 13.7666316	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.66s
    3772:	learn: 7.8122986	test: 13.7667422	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.63s
    3773:	learn: 7.8112190	test: 13.7668549	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.59s
    3774:	learn: 7.8102177	test: 13.7673235	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.56s
    3775:	learn: 7.8098000	test: 13.7672608	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.53s
    3776:	learn: 7.8092456	test: 13.7661976	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.49s
    3777:	learn: 7.8086691	test: 13.7663039	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.46s
    3778:	learn: 7.8083422	test: 13.7670888	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.42s
    3779:	learn: 7.8079347	test: 13.7652214	best: 13.7379133 (3094)	total: 2m 6s	remaining: 7.39s
    3780:	learn: 7.8073255	test: 13.7653573	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.36s
    3781:	learn: 7.8071430	test: 13.7656357	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.32s
    3782:	learn: 7.8047126	test: 13.7642640	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.29s
    3783:	learn: 7.8040735	test: 13.7658748	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.26s
    3784:	learn: 7.8032149	test: 13.7664852	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.22s
    3785:	learn: 7.8030273	test: 13.7657939	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.19s
    3786:	learn: 7.8024113	test: 13.7656551	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.16s
    3787:	learn: 7.8015996	test: 13.7661916	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.12s
    3788:	learn: 7.8011392	test: 13.7652841	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.09s
    3789:	learn: 7.8009249	test: 13.7658463	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.05s
    3790:	learn: 7.7984734	test: 13.7628725	best: 13.7379133 (3094)	total: 2m 7s	remaining: 7.02s
    3791:	learn: 7.7981162	test: 13.7639152	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.99s
    3792:	learn: 7.7978391	test: 13.7641111	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.95s
    3793:	learn: 7.7963588	test: 13.7638463	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.92s
    3794:	learn: 7.7953758	test: 13.7630597	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.89s
    3795:	learn: 7.7935897	test: 13.7635571	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.85s
    3796:	learn: 7.7916184	test: 13.7631322	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.82s
    3797:	learn: 7.7911666	test: 13.7624893	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.79s
    3798:	learn: 7.7908919	test: 13.7624732	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.75s
    3799:	learn: 7.7897223	test: 13.7626048	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.72s
    3800:	learn: 7.7893022	test: 13.7617566	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.68s
    3801:	learn: 7.7888661	test: 13.7614590	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.65s
    3802:	learn: 7.7886217	test: 13.7610236	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.62s
    3803:	learn: 7.7880149	test: 13.7628994	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.58s
    3804:	learn: 7.7874789	test: 13.7628268	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.55s
    3805:	learn: 7.7866132	test: 13.7649056	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.51s
    3806:	learn: 7.7863657	test: 13.7659589	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.48s
    3807:	learn: 7.7858368	test: 13.7664795	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.45s
    3808:	learn: 7.7838798	test: 13.7671770	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.41s
    3809:	learn: 7.7831928	test: 13.7661346	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.38s
    3810:	learn: 7.7827329	test: 13.7667177	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.34s
    3811:	learn: 7.7812916	test: 13.7661608	best: 13.7379133 (3094)	total: 2m 7s	remaining: 6.31s
    3812:	learn: 7.7802190	test: 13.7649548	best: 13.7379133 (3094)	total: 2m 8s	remaining: 6.28s
    3813:	learn: 7.7797423	test: 13.7648633	best: 13.7379133 (3094)	total: 2m 8s	remaining: 6.24s
    3814:	learn: 7.7793561	test: 13.7663363	best: 13.7379133 (3094)	total: 2m 8s	remaining: 6.21s
    3815:	learn: 7.7781602	test: 13.7673843	best: 13.7379133 (3094)	total: 2m 8s	remaining: 6.18s
    3816:	learn: 7.7767002	test: 13.7686080	best: 13.7379133 (3094)	total: 2m 8s	remaining: 6.14s
    3817:	learn: 7.7759838	test: 13.7682164	best: 13.7379133 (3094)	total: 2m 8s	remaining: 6.11s
    3818:	learn: 7.7761580	test: 13.7689222	best: 13.7379133 (3094)	total: 2m 8s	remaining: 6.08s
    3819:	learn: 7.7759710	test: 13.7702417	best: 13.7379133 (3094)	total: 2m 8s	remaining: 6.04s
    3820:	learn: 7.7735835	test: 13.7705468	best: 13.7379133 (3094)	total: 2m 8s	remaining: 6.01s
    3821:	learn: 7.7716544	test: 13.7710476	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.97s
    3822:	learn: 7.7706520	test: 13.7704552	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.94s
    3823:	learn: 7.7704657	test: 13.7706356	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.91s
    3824:	learn: 7.7703742	test: 13.7715186	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.87s
    3825:	learn: 7.7703740	test: 13.7714009	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.84s
    3826:	learn: 7.7693985	test: 13.7714015	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.81s
    3827:	learn: 7.7686706	test: 13.7720774	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.77s
    3828:	learn: 7.7672600	test: 13.7724301	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.74s
    3829:	learn: 7.7667121	test: 13.7721338	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.7s
    3830:	learn: 7.7663062	test: 13.7736960	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.67s
    3831:	learn: 7.7653720	test: 13.7733899	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.64s
    3832:	learn: 7.7653838	test: 13.7738250	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.6s
    3833:	learn: 7.7648609	test: 13.7746719	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.57s
    3834:	learn: 7.7640633	test: 13.7742738	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.54s
    3835:	learn: 7.7624413	test: 13.7722429	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.5s
    3836:	learn: 7.7615612	test: 13.7734769	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.47s
    3837:	learn: 7.7610172	test: 13.7721171	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.43s
    3838:	learn: 7.7606962	test: 13.7712574	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.4s
    3839:	learn: 7.7602735	test: 13.7714871	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.37s
    3840:	learn: 7.7594880	test: 13.7715371	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.33s
    3841:	learn: 7.7592020	test: 13.7713032	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.3s
    3842:	learn: 7.7584225	test: 13.7727552	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.27s
    3843:	learn: 7.7567572	test: 13.7726487	best: 13.7379133 (3094)	total: 2m 8s	remaining: 5.23s
    3844:	learn: 7.7558935	test: 13.7732427	best: 13.7379133 (3094)	total: 2m 9s	remaining: 5.2s
    3845:	learn: 7.7550467	test: 13.7734520	best: 13.7379133 (3094)	total: 2m 9s	remaining: 5.17s
    3846:	learn: 7.7548317	test: 13.7735595	best: 13.7379133 (3094)	total: 2m 9s	remaining: 5.13s
    3847:	learn: 7.7546128	test: 13.7727518	best: 13.7379133 (3094)	total: 2m 9s	remaining: 5.1s
    3848:	learn: 7.7547832	test: 13.7740336	best: 13.7379133 (3094)	total: 2m 9s	remaining: 5.07s
    3849:	learn: 7.7546727	test: 13.7725260	best: 13.7379133 (3094)	total: 2m 9s	remaining: 5.03s
    3850:	learn: 7.7543030	test: 13.7719105	best: 13.7379133 (3094)	total: 2m 9s	remaining: 5s
    3851:	learn: 7.7536690	test: 13.7714862	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.96s
    3852:	learn: 7.7518654	test: 13.7718163	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.93s
    3853:	learn: 7.7520271	test: 13.7724656	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.9s
    3854:	learn: 7.7511024	test: 13.7725060	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.86s
    3855:	learn: 7.7501846	test: 13.7712623	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.83s
    3856:	learn: 7.7497471	test: 13.7717771	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.8s
    3857:	learn: 7.7496754	test: 13.7722386	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.76s
    3858:	learn: 7.7489986	test: 13.7730675	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.73s
    3859:	learn: 7.7490056	test: 13.7728939	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.7s
    3860:	learn: 7.7486866	test: 13.7720445	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.66s
    3861:	learn: 7.7479590	test: 13.7718489	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.63s
    3862:	learn: 7.7463490	test: 13.7717980	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.59s
    3863:	learn: 7.7461131	test: 13.7725568	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.56s
    3864:	learn: 7.7455490	test: 13.7726226	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.53s
    3865:	learn: 7.7453219	test: 13.7734305	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.49s
    3866:	learn: 7.7449353	test: 13.7727098	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.46s
    3867:	learn: 7.7442083	test: 13.7735211	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.43s
    3868:	learn: 7.7437479	test: 13.7743311	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.39s
    3869:	learn: 7.7433871	test: 13.7733963	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.36s
    3870:	learn: 7.7435538	test: 13.7735462	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.33s
    3871:	learn: 7.7423028	test: 13.7740623	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.29s
    3872:	learn: 7.7420482	test: 13.7743466	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.26s
    3873:	learn: 7.7405967	test: 13.7738162	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.23s
    3874:	learn: 7.7401061	test: 13.7740258	best: 13.7379133 (3094)	total: 2m 9s	remaining: 4.19s
    3875:	learn: 7.7396417	test: 13.7747366	best: 13.7379133 (3094)	total: 2m 10s	remaining: 4.16s
    3876:	learn: 7.7391772	test: 13.7748528	best: 13.7379133 (3094)	total: 2m 10s	remaining: 4.13s
    3877:	learn: 7.7388600	test: 13.7750427	best: 13.7379133 (3094)	total: 2m 10s	remaining: 4.09s
    3878:	learn: 7.7382540	test: 13.7746202	best: 13.7379133 (3094)	total: 2m 10s	remaining: 4.06s
    3879:	learn: 7.7371145	test: 13.7748423	best: 13.7379133 (3094)	total: 2m 10s	remaining: 4.02s
    3880:	learn: 7.7365190	test: 13.7752307	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.99s
    3881:	learn: 7.7360816	test: 13.7748842	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.96s
    3882:	learn: 7.7356453	test: 13.7761817	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.92s
    3883:	learn: 7.7358475	test: 13.7769801	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.89s
    3884:	learn: 7.7354123	test: 13.7767332	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.86s
    3885:	learn: 7.7348012	test: 13.7762240	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.82s
    3886:	learn: 7.7343086	test: 13.7760117	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.79s
    3887:	learn: 7.7338393	test: 13.7769409	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.76s
    3888:	learn: 7.7327659	test: 13.7759002	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.72s
    3889:	learn: 7.7321512	test: 13.7754977	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.69s
    3890:	learn: 7.7320112	test: 13.7749389	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.65s
    3891:	learn: 7.7316512	test: 13.7752706	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.62s
    3892:	learn: 7.7312696	test: 13.7744967	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.59s
    3893:	learn: 7.7305535	test: 13.7740615	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.55s
    3894:	learn: 7.7288093	test: 13.7734558	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.52s
    3895:	learn: 7.7282523	test: 13.7731456	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.49s
    3896:	learn: 7.7275166	test: 13.7716467	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.45s
    3897:	learn: 7.7267065	test: 13.7714555	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.42s
    3898:	learn: 7.7266935	test: 13.7710633	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.39s
    3899:	learn: 7.7255281	test: 13.7709648	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.35s
    3900:	learn: 7.7246714	test: 13.7711562	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.32s
    3901:	learn: 7.7248144	test: 13.7719779	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.29s
    3902:	learn: 7.7242684	test: 13.7729507	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.25s
    3903:	learn: 7.7241526	test: 13.7733690	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.22s
    3904:	learn: 7.7242535	test: 13.7731659	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.19s
    3905:	learn: 7.7235349	test: 13.7745637	best: 13.7379133 (3094)	total: 2m 10s	remaining: 3.15s
    3906:	learn: 7.7233523	test: 13.7735865	best: 13.7379133 (3094)	total: 2m 11s	remaining: 3.12s
    3907:	learn: 7.7224305	test: 13.7746798	best: 13.7379133 (3094)	total: 2m 11s	remaining: 3.08s
    3908:	learn: 7.7219273	test: 13.7741851	best: 13.7379133 (3094)	total: 2m 11s	remaining: 3.05s
    3909:	learn: 7.7218214	test: 13.7757947	best: 13.7379133 (3094)	total: 2m 11s	remaining: 3.02s
    3910:	learn: 7.7216479	test: 13.7756340	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.98s
    3911:	learn: 7.7212876	test: 13.7750484	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.95s
    3912:	learn: 7.7205136	test: 13.7750141	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.92s
    3913:	learn: 7.7200098	test: 13.7745909	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.88s
    3914:	learn: 7.7186292	test: 13.7757168	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.85s
    3915:	learn: 7.7179041	test: 13.7768167	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.82s
    3916:	learn: 7.7167322	test: 13.7779938	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.78s
    3917:	learn: 7.7156710	test: 13.7785003	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.75s
    3918:	learn: 7.7152627	test: 13.7763828	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.71s
    3919:	learn: 7.7143772	test: 13.7758230	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.68s
    3920:	learn: 7.7138922	test: 13.7774018	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.65s
    3921:	learn: 7.7132398	test: 13.7782098	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.61s
    3922:	learn: 7.7101656	test: 13.7734465	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.58s
    3923:	learn: 7.7094225	test: 13.7730297	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.55s
    3924:	learn: 7.7085836	test: 13.7741884	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.51s
    3925:	learn: 7.7077526	test: 13.7746060	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.48s
    3926:	learn: 7.7070573	test: 13.7750893	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.45s
    3927:	learn: 7.7051621	test: 13.7726011	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.41s
    3928:	learn: 7.7048375	test: 13.7724758	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.38s
    3929:	learn: 7.7041675	test: 13.7712993	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.35s
    3930:	learn: 7.7034887	test: 13.7726922	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.31s
    3931:	learn: 7.7030348	test: 13.7725188	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.28s
    3932:	learn: 7.7028391	test: 13.7715843	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.25s
    3933:	learn: 7.7022128	test: 13.7721745	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.21s
    3934:	learn: 7.7016068	test: 13.7723587	best: 13.7379133 (3094)	total: 2m 11s	remaining: 2.18s
    3935:	learn: 7.7003468	test: 13.7730720	best: 13.7379133 (3094)	total: 2m 12s	remaining: 2.15s
    3936:	learn: 7.6999367	test: 13.7751167	best: 13.7379133 (3094)	total: 2m 12s	remaining: 2.11s
    3937:	learn: 7.6997428	test: 13.7758670	best: 13.7379133 (3094)	total: 2m 12s	remaining: 2.08s
    3938:	learn: 7.6981605	test: 13.7765558	best: 13.7379133 (3094)	total: 2m 12s	remaining: 2.05s
    3939:	learn: 7.6978126	test: 13.7776608	best: 13.7379133 (3094)	total: 2m 12s	remaining: 2.01s
    3940:	learn: 7.6973802	test: 13.7769121	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.98s
    3941:	learn: 7.6966589	test: 13.7774846	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.95s
    3942:	learn: 7.6961416	test: 13.7777584	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.91s
    3943:	learn: 7.6957588	test: 13.7782872	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.88s
    3944:	learn: 7.6952940	test: 13.7785203	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.84s
    3945:	learn: 7.6954191	test: 13.7767623	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.81s
    3946:	learn: 7.6943807	test: 13.7750529	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.78s
    3947:	learn: 7.6940851	test: 13.7754514	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.74s
    3948:	learn: 7.6935149	test: 13.7747414	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.71s
    3949:	learn: 7.6931955	test: 13.7753930	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.68s
    3950:	learn: 7.6925564	test: 13.7753263	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.64s
    3951:	learn: 7.6918223	test: 13.7760479	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.61s
    3952:	learn: 7.6913812	test: 13.7756433	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.58s
    3953:	learn: 7.6904003	test: 13.7760273	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.54s
    3954:	learn: 7.6900122	test: 13.7773150	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.51s
    3955:	learn: 7.6893091	test: 13.7776651	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.48s
    3956:	learn: 7.6888733	test: 13.7783086	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.44s
    3957:	learn: 7.6885144	test: 13.7777866	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.41s
    3958:	learn: 7.6874619	test: 13.7786583	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.38s
    3959:	learn: 7.6872067	test: 13.7795755	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.34s
    3960:	learn: 7.6871561	test: 13.7798346	best: 13.7379133 (3094)	total: 2m 12s	remaining: 1.31s
    3961:	learn: 7.6869391	test: 13.7797710	best: 13.7379133 (3094)	total: 2m 13s	remaining: 1.27s
    3962:	learn: 7.6855668	test: 13.7786127	best: 13.7379133 (3094)	total: 2m 13s	remaining: 1.24s
    3963:	learn: 7.6849511	test: 13.7785805	best: 13.7379133 (3094)	total: 2m 13s	remaining: 1.21s
    3964:	learn: 7.6845605	test: 13.7789109	best: 13.7379133 (3094)	total: 2m 13s	remaining: 1.18s
    3965:	learn: 7.6840565	test: 13.7790349	best: 13.7379133 (3094)	total: 2m 13s	remaining: 1.14s
    3966:	learn: 7.6840242	test: 13.7793205	best: 13.7379133 (3094)	total: 2m 13s	remaining: 1.11s
    3967:	learn: 7.6836493	test: 13.7787336	best: 13.7379133 (3094)	total: 2m 13s	remaining: 1.07s
    3968:	learn: 7.6832532	test: 13.7782400	best: 13.7379133 (3094)	total: 2m 13s	remaining: 1.04s
    3969:	learn: 7.6819185	test: 13.7778325	best: 13.7379133 (3094)	total: 2m 13s	remaining: 1.01s
    3970:	learn: 7.6814767	test: 13.7787357	best: 13.7379133 (3094)	total: 2m 13s	remaining: 974ms
    3971:	learn: 7.6812217	test: 13.7787549	best: 13.7379133 (3094)	total: 2m 13s	remaining: 940ms
    3972:	learn: 7.6808399	test: 13.7795716	best: 13.7379133 (3094)	total: 2m 13s	remaining: 907ms
    3973:	learn: 7.6802419	test: 13.7793262	best: 13.7379133 (3094)	total: 2m 13s	remaining: 873ms
    3974:	learn: 7.6801544	test: 13.7780966	best: 13.7379133 (3094)	total: 2m 13s	remaining: 839ms
    3975:	learn: 7.6797393	test: 13.7777078	best: 13.7379133 (3094)	total: 2m 13s	remaining: 806ms
    3976:	learn: 7.6796751	test: 13.7772718	best: 13.7379133 (3094)	total: 2m 13s	remaining: 772ms
    3977:	learn: 7.6775546	test: 13.7775797	best: 13.7379133 (3094)	total: 2m 13s	remaining: 739ms
    3978:	learn: 7.6755002	test: 13.7759341	best: 13.7379133 (3094)	total: 2m 13s	remaining: 705ms
    3979:	learn: 7.6752320	test: 13.7752332	best: 13.7379133 (3094)	total: 2m 13s	remaining: 672ms
    3980:	learn: 7.6747445	test: 13.7760218	best: 13.7379133 (3094)	total: 2m 13s	remaining: 638ms
    3981:	learn: 7.6739415	test: 13.7754860	best: 13.7379133 (3094)	total: 2m 13s	remaining: 604ms
    3982:	learn: 7.6723299	test: 13.7751334	best: 13.7379133 (3094)	total: 2m 13s	remaining: 571ms
    3983:	learn: 7.6720451	test: 13.7772243	best: 13.7379133 (3094)	total: 2m 13s	remaining: 537ms
    3984:	learn: 7.6710632	test: 13.7758349	best: 13.7379133 (3094)	total: 2m 13s	remaining: 504ms
    3985:	learn: 7.6705191	test: 13.7756060	best: 13.7379133 (3094)	total: 2m 13s	remaining: 470ms
    3986:	learn: 7.6702440	test: 13.7751846	best: 13.7379133 (3094)	total: 2m 13s	remaining: 436ms
    3987:	learn: 7.6693427	test: 13.7761297	best: 13.7379133 (3094)	total: 2m 13s	remaining: 403ms
    3988:	learn: 7.6680891	test: 13.7755862	best: 13.7379133 (3094)	total: 2m 13s	remaining: 369ms
    3989:	learn: 7.6671181	test: 13.7751486	best: 13.7379133 (3094)	total: 2m 13s	remaining: 336ms
    3990:	learn: 7.6669199	test: 13.7740173	best: 13.7379133 (3094)	total: 2m 14s	remaining: 302ms
    3991:	learn: 7.6660430	test: 13.7733066	best: 13.7379133 (3094)	total: 2m 14s	remaining: 269ms
    3992:	learn: 7.6657965	test: 13.7730604	best: 13.7379133 (3094)	total: 2m 14s	remaining: 235ms
    3993:	learn: 7.6654740	test: 13.7745018	best: 13.7379133 (3094)	total: 2m 14s	remaining: 202ms
    3994:	learn: 7.6648474	test: 13.7734049	best: 13.7379133 (3094)	total: 2m 14s	remaining: 168ms
    3995:	learn: 7.6641926	test: 13.7725158	best: 13.7379133 (3094)	total: 2m 14s	remaining: 134ms
    3996:	learn: 7.6620207	test: 13.7732069	best: 13.7379133 (3094)	total: 2m 14s	remaining: 101ms
    3997:	learn: 7.6620473	test: 13.7737919	best: 13.7379133 (3094)	total: 2m 14s	remaining: 67.2ms
    3998:	learn: 7.6615257	test: 13.7723114	best: 13.7379133 (3094)	total: 2m 14s	remaining: 33.6ms
    3999:	learn: 7.6609669	test: 13.7726341	best: 13.7379133 (3094)	total: 2m 14s	remaining: 0us
    
    bestTest = 13.73791335
    bestIteration = 3094
    
    Shrink model to first 3095 iterations.
    




    <catboost.core.CatBoostRegressor at 0x24bfe999ba8>




```python
# do prediction on the 'submit' data and save as csv for final submission
submit_Y.total_cases = model.predict(submit.drop(['total_cases', 'city'], axis=1)).astype(int)
submit_Y.to_csv('submission.csv', index=False)
```
