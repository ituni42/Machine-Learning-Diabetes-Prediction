import time

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('diabetes_prediction_dataset.csv')


print("Provera koliko ima Other vrednosti za Gender:",dataset['gender'].value_counts()['Other'])
print("Uklanjanje tih vrednosti...")
dataset = dataset.drop(dataset[dataset.gender == 'Other'].index)

print("Broj ćelija u kojem fale podaci za svaki atribut:")
print(dataset.isnull().sum())

# Izdvojite samo redove sa BMI većim od 45
#dataset_bmi_gt_45 = dataset[dataset['bmi'] > 45]

# Provjerite koliko od tih redova ima vrijednost dijabetesa 1 i koliko ima vrijednost dijabetesa 0
#diabetes_counts = dataset_bmi_gt_45['diabetes'].value_counts()

#print("Broj redova sa dijabetesom 1:", diabetes_counts[1])
#print("Broj redova sa dijabetesom 0:", diabetes_counts[0])

#Pregled Anomalija:
def plot_boxplot(df,ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()
def PlotData():
    plot_boxplot(dataset, "age")
    plot_boxplot(dataset, "bmi")
    plot_boxplot(dataset, "HbA1c_level")
    plot_boxplot(dataset, "blood_glucose_level")
PlotData()

#Uklanjanje Anomalija:
def outliers(df,ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    ls = df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]

    return ls

index_list = []
for feature in ['age','bmi','HbA1c_level',"blood_glucose_level"]:
    index_list.extend(outliers(dataset,feature))

def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df

print("Broj redova pre brisanja anomalija:",len(dataset))
dataset = remove(dataset, index_list)
print("Broj redova nakon brisanja anomalija:",len(dataset))
PlotData()

dataset.to_csv('diabetes_prediction_datasetFiltered.csv', index=False)