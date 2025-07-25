import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv("diabetes_prediction_datasetFiltered.csv")

#histogram - da vidimo distribuciju
dataset.hist(bins=60, figsize=(20,10))
plt.show()


# Cat plots - uvid u postojeće korelacije
for i in ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'HbA1c_level', 'blood_glucose_level']:
    g = sns.catplot(data=dataset, x=i, hue="diabetes", kind="count", height=8, aspect=2)
    if i == 'age':
        g.ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
    plt.subplots_adjust(top=0.9)
    plt.title(f'Count plot for {i}')
    plt.show()

# Za BMI gledamo imamo poseban slučaj
# Nameštamo "Okvirne" Vrednosti za BMI, Kako na X osi ne bi imali beskonacno vrednosti, jer BMI pripada skupu Realnih brojeva
bins = [15, 20, 25, 30, 35, 40]

dataset['bmi_grouped'] = pd.cut(dataset['bmi'], bins=bins, labels=['15-20', '20-25', '25-30', '30-35', '35-40'])

g = sns.catplot(data=dataset, x='bmi_grouped', hue="diabetes", kind="count", height=6, aspect=2)
plt.subplots_adjust(top=0.9)
plt.title('Count plot for BMI')
plt.show()