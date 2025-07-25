Diabetes Prediction - Machine Learning Project
📝 Project Description

This project focuses on predicting the onset of diabetes based on crucial health-related data from patients. The primary goal is to develop, train, and evaluate various machine learning models to accurately classify whether a patient has diabetes. The dataset includes key medical indicators such as gender, age, BMI, HbA1c levels, and blood glucose levels, making it a comprehensive basis for predictive modeling.
📊 Dataset

The dataset used for this project contains the following attributes for each patient:

    gender: The patient's gender (Male, Female, Other).

    age: The patient's age in years.

    hypertension: Whether the patient has hypertension (0 for No, 1 for Yes).

    heart_disease: Whether the patient has a heart disease (0 for No, 1 for Yes).

    smoking_history: The patient's smoking status (e.g., 'never', 'current', 'former').

    bmi: Body Mass Index.

    HbA1c_level: Hemoglobin A1c level, a key indicator of long-term blood sugar control.

    blood_glucose_level: The patient's current blood glucose concentration.

    diabetes: The target variable (0 for No, 1 for Yes).

⚙️ Project Structure

The project is organized into a modular pipeline, with each stage handled by a dedicated Python script:

    DataFiltering.py: Cleans the dataset by removing outliers and irrelevant entries.

    ExploratoryDataAnalysis.py: Performs data visualization and analysis to understand distributions and correlations.

    FeatureScaling.py: Normalizes numerical features to a common scale.

    FeatureEncoding.py: Converts categorical features into a numerical format using One-Hot Encoding.

    DataBalancing.py: Addresses the class imbalance in the target variable using undersampling.

    GridSearch.py: Optimizes hyperparameters for baseline machine learning models.

    ResiProjekat.py: Implements the final model training, evaluation, and comparison of advanced ensemble techniques (Stacking, Bagging, and Boosting).

🛠️ Installation and Execution

To run this project, you need Python and the following libraries installed:

    pandas

    matplotlib

    seaborn

    scikit-learn

    imblearn

You can install them using pip:
Generated bash

      
pip install pandas matplotlib seaborn scikit-learn imblearn

    

IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

To execute the full pipeline, run the scripts in the following order:

    python DataFiltering.py

    python ExploratoryDataAnalysis.py

    python FeatureScaling.py

    python FeatureEncoding.py

    python DataBalancing.py

    python GridSearch.py

    python ResiProjekat.py

🧪 Methodology
1. Data Preprocessing

    Filtering: Rows with 'Other' as the gender were removed to maintain data consistency.

    Outlier Removal: The Interquartile Range (IQR) method was used to detect and remove outliers from the age, bmi, HbA1c_level, and blood_glucose_level features.

    Feature Scaling: Numerical features were scaled to a range using MinMaxScaler.

    Feature Encoding: Categorical features (gender, smoking_history) were converted into a numerical format using OneHotEncoder.

    Data Balancing: To handle the imbalanced nature of the diabetes class, RandomUnderSampler was applied to create a balanced dataset for model training.

2. Exploratory Data Analysis (EDA)

    Histograms were plotted to visualize the distribution of each feature.

    Categorical plots (catplots) were used to analyze the relationship between various features and the target variable, diabetes.

3. Modeling and Evaluation

The following models and techniques were implemented and evaluated:

    Baseline Models (with GridSearchCV for hyperparameter tuning):

        Decision Tree

        Logistic Regression

        K-Nearest Neighbors (KNN)

    Ensemble Methods:

        Stacking: Combines predictions from multiple base models (Logistic Regression, KNN, Decision Tree) with a final logistic regression estimator.

        Bagging: Implemented using RandomForestClassifier.

        Boosting: Implemented using AdaBoostClassifier.

Models were evaluated using a comprehensive set of metrics:

    Accuracy

    Precision

    Recall

    F1-Score

    Matthews Correlation Coefficient (MCC)

    Confusion Matrix

    ROC Curve and AUC (Area Under the Curve)

A feature importance analysis was also conducted to identify the most influential predictors for diabetes.
📈 Results

The ResiProjekat.py script provides a detailed evaluation of each model's performance. It compares models trained on the full feature set against models trained on only the top 4 most important features. ROC curves are generated to provide a clear visual comparison of classifier performance, helping to identify the most robust model for this prediction task.
🤝 Contributing

Contributions are welcome! If you would like to improve this project, please fork the repository and submit a pull request with your changes.
📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.