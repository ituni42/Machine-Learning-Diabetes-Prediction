<h1 align="center">Diabetes Prediction Pipeline</h1>

<p align="center">
  An end-to-end machine learning project focused on predicting diabetes from health data. This repository walks through the complete data science workflow, from initial data cleaning to advanced model evaluation.
  <br />
  <a href="https://github.com/ituni42/Machine-Learning-Diabetes-Prediction/issues">Report Bug</a>
  ·
  <a href="https://github.com/ituni42/Machine-Learning-Diabetes-Prediction/issues">Request Feature</a>
</p>

---

### ➤ About The Project

This project demonstrates a practical approach to solving a real-world classification problem. The primary goal is to build a robust model capable of accurately predicting whether a patient has diabetes based on key medical indicators.

The pipeline addresses several critical data science challenges, including:
*   **Data Cleaning**: Tackling outliers and inconsistent data entries.
*   **Exploratory Data Analysis**: Using visualizations to uncover insights and correlations.
*   **Feature Engineering**: Preparing data for modeling through scaling and encoding.
*   **Handling Class Imbalance**: Using undersampling to build a fair and unbiased model.
*   **Advanced Modeling**: Implementing and comparing powerful ensemble techniques like **Stacking**, **Bagging**, and **Boosting**.

### ✨ Built With
This project relies on a standard stack of powerful data science libraries:
*   [Python](https://www.python.org/)
*   [Pandas](https://pandas.pydata.org/)
*   [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)
*   [Scikit-Learn](https://scikit-learn.org/)
*   [Imbalanced-learn](https://imbalanced-learn.org/stable/)

---

### ⚙️ Project Workflow
The repository is structured as a sequence of Python scripts, each performing a specific step. They are designed to be run in the following order to reproduce the full pipeline.

| # | Script Name | Purpose |
| :-- | :--- | :--- |
| 1 | `DataFiltering.py` | Cleans the raw dataset by removing outliers and invalid entries. |
| 2 | `ExploratoryDataAnalysis.py` | Generates visualizations to understand data distributions and relationships. |
| 3 | `FeatureScaling.py` & `FeatureEncoding.py` | Normalizes numerical data and converts categorical data into a machine-readable format. |
| 4 | `DataBalancing.py` | Addresses the class imbalance in the target variable using random undersampling. |
| 5 | `GridSearch.py` | Performs hyperparameter tuning on baseline models to find the best configurations. |
| 6 | `ResiProjekat.py` | Trains, tests, and evaluates the final ensemble models, reporting key performance metrics. |

---

### 🚀 Getting Started
To get a local copy up and running, follow these simple steps.

#### Prerequisites
Make sure you have Python (3.8+) and Pip installed on your system.

#### Installation & Execution

1.  **Clone the repository to your local machine.**
    ```sh
    git clone https://github.com/ituni42/Machine-Learning-Diabetes-Prediction.git
    ```

2.  **Navigate into the project directory.**
    ```sh
    cd Machine-Learning-Diabetes-Prediction
    ```

3.  **Install all the necessary libraries.**
    ```sh
    pip install pandas matplotlib seaborn scikit-learn imblearn
    ```

4.  **Run the Python scripts** in the order described in the workflow table above to see the results.

---

### 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request.
1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

### 📜 License
Distributed under the MIT License. See the `LICENSE` file for more information.
