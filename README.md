<br/>
<p align="center">
<h1 align="center">🩺 Diabetes Prediction Engine</h1>
<p align="center">
From messy data to a predictive model. An end-to-end machine learning project exploring how to forecast diabetes with Python & Scikit-Learn.
<br />
<br />
<a href="https://github.com/ituni42/Machine-Learning-Diabetes-Prediction/issues">Report a Bug</a>
·
<a href="https://github.com/ituni42/Machine-Learning-Diabetes-Prediction/issues">Request a Feature</a>
</p>
</p>

![alt text](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

The Mission

This project is more than just a model.fit() call. It's a deep dive into the practical data science workflow, designed to build a robust and reliable diabetes classifier from the ground up.

We start with raw, real-world health data and guide it through a rigorous pipeline of cleaning, exploration, feature engineering, and modeling. The final goal? To not only predict outcomes but also to uncover the most influential health markers that point to a diabetes diagnosis using advanced ensemble techniques and feature importance analysis.
Tech Stack

This project leverages a powerful stack of open-source libraries:

    [Python] - The core language for everything.

    [Pandas] - For data manipulation, cleaning, and wrangling.

    [Matplotlib & Seaborn] - For creating insightful and rich data visualizations.

    [Scikit-Learn] - The heart of our machine learning pipeline, used for modeling, scaling, and evaluation.

    [Imbalanced-learn] - The key to solving class imbalance and building a fair model.

![alt text](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

The Workflow ⚙️

The project is broken down into a logical sequence of scripts, each performing a critical task. To reproduce the results, run them in this order:

    DataFiltering.py

        Taming the Beast: Wrestles with the raw dataset, removes statistical outliers, and creates a clean foundation for analysis.

    ExploratoryDataAnalysis.py

        Visual Storytelling: Uses plots and charts to uncover the data's hidden patterns and initial correlations.

    FeatureScaling.py & FeatureEncoding.py

        Prep School: Scales numerical features and one-hot encodes categorical data, making it ready for the machine learning models.

    DataBalancing.py

        Leveling the Playing Field: Tackles the dataset's significant class imbalance with undersampling to prevent a biased model.

    GridSearch.py

        The Tune-Up: Runs an exhaustive hyperparameter search to find the optimal settings for our baseline classifiers.

    ResiProjekat.py

        The Grand Finale: Deploys advanced ensemble models (Stacking, Bagging, Boosting), evaluates their performance with metrics like AUC-ROC, and reveals which features matter most.

![alt text](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

How to Run It

Ready to see it in action? Get a local copy up and running with these steps.

    Clone the repository
    Generated sh

      
git clone https://github.com/ituni42/Machine-Learning-Diabetes-Prediction.git

    

IGNORE_WHEN_COPYING_START
Use code with caution. Sh
IGNORE_WHEN_COPYING_END

Navigate into the directory
Generated sh

      
cd Machine-Learning-Diabetes-Prediction

    

IGNORE_WHEN_COPYING_START
Use code with caution. Sh
IGNORE_WHEN_COPYING_END

Install the dependencies
Generated sh

      
pip install pandas matplotlib seaborn scikit-learn imblearn

    

IGNORE_WHEN_COPYING_START

    Use code with caution. Sh
    IGNORE_WHEN_COPYING_END

    Run the scripts in the order described above!

Contribute & Connect

Found a bug, have an idea, or want to improve the models? I'd love to hear from you. Contributions are always welcome!

    Fork the Project

    Create your Feature Branch (git checkout -b feature/MyCoolFeature)

    Commit your Changes (git commit -m 'Add some MyCoolFeature')

    Push to the Branch (git push origin feature/MyCoolFeature)

    Open a Pull Request

![alt text](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

License

Distributed under the MIT License. See LICENSE for more information.
