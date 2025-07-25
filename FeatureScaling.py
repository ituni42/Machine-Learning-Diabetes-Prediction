import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)

dataset = pd.read_csv("diabetes_prediction_datasetFiltered.csv")
original_columns = dataset.columns.tolist()

dataset = dataset.drop(columns=["gender","smoking_history","hypertension","heart_disease","diabetes"])

X = dataset.iloc[:, 0:4]

scaleMinMax = MinMaxScaler(feature_range=(0,1))
X = scaleMinMax.fit_transform(X)
X = pd.DataFrame(X, columns=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'])


dataset = pd.read_csv("diabetes_prediction_datasetFiltered.csv")
dataset = dataset.drop( columns=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
dataset = pd.concat([dataset, X], axis=1)


dataset = dataset[original_columns]
dataset.to_csv("diabetes_prediction_datasetNormalized.csv",index= False)

print("Normalized Dataset:")
print(dataset.head())