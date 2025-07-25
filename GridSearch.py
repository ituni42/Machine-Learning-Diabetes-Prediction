import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# Ucitavanje Podataka
data = pd.read_csv('diabetes_prediction_datasetBalanced.csv')
X = data.drop(["diabetes"], axis=1)
y = data["diabetes"]

# Podela na podatke za trening i testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Decision Tree Hyperparametri
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
dt = DecisionTreeClassifier(random_state=42)

# Logistic Regression Hyperparameters
lr_params = [
    {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']},
    {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'sag', 'newton-cg']},
    {'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['saga'], 'l1_ratio': [0.1, 0.5, 0.9]}
]
lr = LogisticRegression(random_state=42, max_iter=10000)

# K-Nearest Neighbors hyperparameters
knn_params = {
    'n_neighbors':list(range(1, 11)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()



classifiers = {
    "Decision Tree": (dt, dt_params),
    "Logistic Regression": (lr, lr_params),
    "K-Nearest Neighbors": (knn, knn_params)
}

results = {}

for clf_name, (clf, clf_params) in classifiers.items():
    grid_search = GridSearchCV(clf, clf_params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    best_params = grid_search.best_params_
    cross_val_score = grid_search.best_score_

    print(f"Best {clf_name} parameters: {best_params}")
    print(f"Best {clf_name} cross-validation score: {cross_val_score}")
    print(f"{clf_name} test accuracy: {test_accuracy}\n")

    results[clf_name] = {
        "best_params": best_params,
        "cross_val_score": cross_val_score,
        "test_accuracy": test_accuracy
    }

