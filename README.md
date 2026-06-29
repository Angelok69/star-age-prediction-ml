# Star Age Prediction using Machine Learning

A data science project focused on predicting the age of cool solar-type stars based on their physical and atmospheric characteristics. Developed as a final graduation project for the Professional Retraining Program in Data Analytics & Machine Learning at Saratov State University (2023).

## 📌 Project Overview
Determining stellar age is a fundamental yet challenging task in astrophysics. This project formalizes the problem as a **regression task**, utilizing physical parameters extracted from stellar spectra to predict a star's age in gigayears (Gyr).

## 📊 Dataset & Features
The model is trained on a specialized dataset containing physical parameters of stars (originally sourced from Kaggle):
* **Teff** – Effective temperature (K)
* **logg** – Surface gravity
* **Vt** – Microturbulent velocity (km/s)
* **[Fe/H]** – Metallicity
* **Mass** – Estimated stellar mass

## 🛠️ Data Preprocessing & EDA
* **Data Cleaning:** Removed metadata (star names) and measurement error attributes to reduce noise; handled and dropped missing values (NaNs).
* **Exploratory Data Analysis (EDA):** Visualized feature distributions and relationships using `seaborn.pairplot`.
* **Feature Importance:** Evaluated feature weights using a `RandomForestRegressor`, identifying **Mass** and **Surface Gravity (logg)** as the most critical predictors.
* **Normalization:** Scaled all continuous features using `MinMaxScaler` for optimal model performance.

## 🤖 Machine Learning Models Implemented
The dataset was split into training and testing sets (70:30 ratio). Four different regression algorithms were implemented and evaluated using `scikit-learn`:
1. **Linear Regression**
2. **Ridge Regression**
3. **K-Nearest Neighbors (KNN) Regressor** (Optimized at $k=6$)
4. **Decision Tree Regressor**

## 📈 Key Results & Evaluation
Models were compared using key regression metrics: Mean Absolute Error (**MAE**), Mean Squared Error (**MSE**), and Root Mean Squared Error (**RMSE**).

* **Top Performer:** The **K-Nearest Neighbors (KNN)** model achieved the highest accuracy, yielding the lowest error rates among all tested approaches.
* **Limitation:** The overall prediction error remains relatively high due to the limited size of the sample dataset (441 records after cleaning), highlighting the need for larger astronomical datasets in future iterations.

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Statsmodels
* **Environment:** VS Code / Jupyter Notebook
