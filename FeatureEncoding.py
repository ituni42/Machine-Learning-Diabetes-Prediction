import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#Performing OneHotEncoding mmmm hot
dataset = pd.read_csv("diabetes_prediction_datasetNormalized.csv")

print("\nTipovi Pre i Posle FeatureEncoding-a:")
def tipovi(dataset):
    print("------------------------------------")
    print("Prisutni Tipovi Podataka:")
    print(dataset.dtypes)
    return

tipovi(dataset)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output = False).set_output(transform="pandas")

#mapiranje brojeva umesto stringova
ohetransformgender = ohe.fit_transform(dataset[["gender"]])
ohetransformsmoking = ohe.fit_transform(dataset[["smoking_history"]])

dataset = pd.concat([dataset,ohetransformgender],axis=1).drop(columns=["gender"])
dataset = pd.concat([dataset,ohetransformsmoking],axis=1).drop(columns=["smoking_history"])
dataset = dataset.reset_index(drop=True)


pd.set_option('display.max_columns', None)
print(tipovi(dataset))
dataset.to_csv("diabetes_prediction_datasetEncoded.csv",index=False)