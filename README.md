# 📞 Telecom Churn Prediction

An end-to-end machine learning project designed to predict customer churn for telecom operators. This project explores the complete machine learning workflow—from data exploration and complex preprocessing to training advanced gradient boosting models and outputting final risk predictions.

## 📌 Project Overview

The objective is to predict whether a telecom customer will cancel their service (**churn**) based on usage metrics, billing history, and profile data.

With a massive dataset of over **590,000 records**, this project focuses on robust feature handling and building an advanced pipeline to maximize precision.

In the business world, minimizing false positives ensures that expensive customer retention offers are spent only on users who are truly at risk of leaving.

## 🎯 Objectives

- Perform comprehensive Exploratory Data Analysis (EDA) on a large-scale dataset.
- Handle missing value structures and clean data for tree-based algorithms.
- Implement custom, manual Ordinal Encoding for categorical features.
- Train, tune, and benchmark three powerful gradient boosting engines.
- Optimize decision thresholds to minimize false-positive predictions.
- Generate a final prediction array against holdout test data.

## ⚙️ Machine Learning Workflow

### 1. Exploratory Data Analysis (EDA)

- **Dataset Inspection:** Analyzing distribution balance across a 590k+ row matrix.
- **Feature Distribution:** Spotting drops in data consumption or spikes in customer service calls.
- **Imbalance Analysis:** Evaluating the natural ratio of loyal versus churned accounts.

### 2. Data Preprocessing & Encoding

- **Ordinal Mapping:** Hand-crafting ordinal dictionaries to translate categorical text strings into numeric scales. This preserves memory and matches how decision trees split data.
- **Null Strategy:** Preparing implicit missing value tracks compatible with gradient boosters.

### 3. Model Building & Ensembling

Multiple state-of-the-art classification frameworks were trained side-by-side within a unified workflow:

- **LightGBM:** Leveraged for fast, leaf-wise tree growth on large datasets.
- **XGBoost:** Utilized for its advanced $L_1$ and $L_2$ regularization to prevent overfitting.
- **CatBoost:** Implemented to cross-examine categorical structures safely.

### 4. Evaluation & Precision Tuning

Models were evaluated using **Precision-Recall curves** instead of standard accuracy to make sure predictions are highly reliable before triggering marketing spend.

## 🛠️ Technologies Used

- Python
- Pandas & NumPy *(Data manipulation)*
- Matplotlib & Seaborn *(Data visualization)*
- Scikit-Learn *(Evaluation metrics & splitting)*
- LightGBM, XGBoost, CatBoost *(Gradient boosting engines)*
- Jupyter Notebook *(Development environment)*

## 📁 Project Structure

```text
telecom-churn-prediction/
├── train (1).csv       # Training dataset with 590,000+ historical rows and churn labels
├── test.csv            # Holdout test dataset used to evaluate final model deployment
├── churn.ipynb         # Complete end-to-end Jupyter Notebook (EDA, Preprocessing, Modeling)
└── README.md           # Project documentation and workflow map

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/buildwitharyaman/telecom-churn-prediction.git
cd telecom-churn-prediction
```

### 2. Install Dependencies

Make sure you have the core machine learning libraries installed:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn notebook
```

### 3. Open the Notebook

Launch the Jupyter interface from your terminal:

```bash
jupyter notebook
```

Open **`churn.ipynb`** and run all cells sequentially to execute the pipeline from raw data to final predictions.

## 📚 What I Learned

This project provided hands-on experience with advanced data science concepts:

- Managing and scaling pipelines for large datasets (590k+ rows).
- Why tree-based models prefer Ordinal Encoding over One-Hot Encoding.
- Working with modern gradient boosting frameworks (LightGBM, XGBoost, CatBoost).
- Optimizing for business-critical metrics (Precision) rather than just basic accuracy.

## 👨‍💻 Author

**Aryaman**

Middle school student (8th grade) passionate about Artificial Intelligence, Machine Learning, and Data Science.

**GitHub:** https://github.com/buildwitharyaman
GitHub: github.com/yourusername
