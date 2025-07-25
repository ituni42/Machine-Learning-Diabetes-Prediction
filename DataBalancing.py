import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv('diabetes_prediction_datasetEncoded.csv')

#Provera da li su podaci balansirani
print('Nema Dijabetes: ', data['diabetes'].value_counts()[0])
print('Ima Dijabetes: ', data['diabetes'].value_counts()[1])
#Nisu


X = data.drop(["diabetes"], axis=1)
y = data["diabetes"]

#Plotujemo odnos ljudi koji nemaju i imaju dijabetes pre undersampling-a
l = ["Nema dijabetes","Ima dijabetes"]
fig1,ax1 = plt.subplots()
plt.title("Dijabetes")
ax1.pie(y.value_counts(),autopct='%.2f',labels=l)
plt.show()

rus = RandomUnderSampler(sampling_strategy=1)
X_res, y_res = rus.fit_resample(X,y)

#Sada Plotujemo isto to, posle undersampling-
ax = y_res.value_counts().plot.pie(autopct='%.2f',labels=l)
_ = ax.set_title("Diabetes Under-sampling")
plt.show()

#Izbacujemo balansirani skup u fajl, pa cemo ga koristiti za treniranje modela.
X_res['diabetes'] = y_res
X_res.to_csv("diabetes_prediction_datasetBalanced.csv",index=False)