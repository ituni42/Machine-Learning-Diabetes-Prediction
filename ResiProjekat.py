from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# Učitavanje podataka
data = pd.read_csv('diabetes_prediction_datasetBalanced.csv')
X = data.drop(["diabetes"], axis=1)
y = data["diabetes"]

# Podela podataka na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

def OdradiSve():

    #Algoritam za nalazenje K:
    k_values = list(range(1, 10))

    k_scores = {}

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        k_scores[k] = np.mean(scores)
    best_k = max(k_scores, key=k_scores.get)
    best_score = k_scores[best_k]

    # Stacking, Bagging i Boosting
    classifiers = [
        ('lr', LogisticRegressionCV(cv=5, random_state=42, max_iter=10000)),
        ('knn', KNeighborsClassifier(n_neighbors=best_k,metric="euclidean",weights="distance")),
        ('dt', DecisionTreeClassifier(criterion="gini",max_depth=7,min_samples_split=2))
    ]

    stacking_clf = StackingClassifier(estimators=classifiers, final_estimator=LogisticRegressionCV(cv=5))
    bagging_clf = RandomForestClassifier()
    boosting_clf = AdaBoostClassifier()

    models = {
        'Stacking': stacking_clf,
        'Bagging': bagging_clf,
        'Boosting': boosting_clf
    }

    # Podešavanje hiperparametara i unakrsna validacija
    for name, model in models.items():
        if name == 'Stacking':
            params = {}
        elif name == 'Bagging':
            params = {'n_estimators': [50, 100, 200]}
        elif name == 'Boosting':
            params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

        grid_search = GridSearchCV(model, params, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"Najbolji Parametri za {name}: {grid_search.best_params_}")
        print(f"Cross-validation score za {name}: {grid_search.best_score_}")

        # Fitovanje modela sa najboljim parametrima
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Analiza rezultata predikcije
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        print(f"Metrics for {name}:")
        print("Accuracy:",  "%.2f" % (accuracy * 100))
        print("Precision:", "%.2f" % (precision * 100))
        print("Recall:",  "%.2f" % (recall * 100))
        print("F1 Score:", "%.2f" % (f1 * 100))
        print("Matthews Korelacioni Koeficijent:","%.2f" % (mcc))
        print("Confusion Matrix:\n", confusion)

        # ROC kriva
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.show()

        print("\n")
    return

OdradiSve()

# Odabir najbitnijih atributa
# Za RandomForestSSSClassifier možemo dobiti važnost atributa
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

top_features = feature_importances.head(4).index.tolist()
X_top = X[top_features]
print("Najbitnija 4 atributa:")
print(top_features)

# Podela podataka na trening i test skup koristeći samo top 4 atributa
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.3, stratify=y)

print("---------------------------------------------------------------------------------------------")
print("REZULTATI MODELA KOJI IMA SAMO 4 ATRIBUTA:")
OdradiSve()