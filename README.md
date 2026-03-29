# 🏦 Loan Conversion Prediction — ML Classification Project

A machine learning project to predict whether a bank customer will take a **loan on their credit card**, using classification models with hyperparameter tuning.

---

## 📌 Project Overview

This project was built as part of a Machine Learning course (Semester 4). The goal is to help a bank identify customers who are likely to take a loan on their credit card, based on spending patterns and demographic data.

Two datasets are merged, cleaned, and used to train and evaluate three classification models. The best model is selected based on accuracy and tuned using GridSearchCV.

---

## 📂 Repository Structure

```
├── Assignment2.ipynb       # Main Jupyter Notebook with full analysis
├── Part2 - Data1.csv       # First dataset (customer demographics)
├── Part2 - Data2.csv       # Second dataset (spending behaviour)
└── README.md               # Project documentation
```

> ⚠️ **Note:** Update the file paths in the notebook from local paths (e.g., `D:\SEM4\...`) to relative paths like `./Part2 - Data1.csv` before running on any other machine.

---

## 🧰 Tech Stack & Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data loading, merging, cleaning |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization (EDA) |
| `scikit-learn` | Model training, evaluation, tuning |
| `imbalanced-learn (SMOTE)` | Handling class imbalance |

---

## 🔄 Project Workflow

### 1. Data Importing & Merging
- Loaded two CSV datasets: customer demographics (`Data1`) and spending behaviour (`Data2`)
- Merged on the `ID` column to create a unified dataset

### 2. Data Cleaning
- Checked for null values; dropped rows with missing `LoanOnCard` entries
- Converted `LoanOnCard` column to integer type

### 3. Exploratory Data Analysis (EDA)
- Distribution of `Age`
- Box plot for `HighestSpend`
- Distribution of `MonthlyAverageSpend`
- Count plot for `CreditCard` usage

### 4. Data Preprocessing
- Separated features (`X`) and target (`y` = `LoanOnCard`)
- Applied **SMOTE** to fix class imbalance
- Train-test split: **70% train / 30% test**
- Feature scaling using `StandardScaler`

### 5. Model Training
Three classifiers were trained:
- Logistic Regression
- Gaussian Naive Bayes
- Random Forest Classifier

### 6. Model Evaluation & Tuning
All models were evaluated using accuracy score and classification report. GridSearchCV was used to tune hyperparameters for all three models.

---

## 📊 Results

| Model | Accuracy (Before Tuning) | Best Hyperparameters |
|---|---|---|
| Logistic Regression | 89.29% | `C=1, penalty='l2', solver='liblinear'` |
| Naive Bayes | 88.81% | `var_smoothing=1e-9` |
| **Random Forest** | **97.93%** | `max_depth=None, min_samples_split=2, n_estimators=100` |

✅ **Winner: Random Forest** — selected as the final model due to its superior accuracy.

---

## 💡 Key Insights

- **MonthlyAverageSpend**, **HighestSpend**, and **HiddenScore** showed strong correlation with the target variable.
- SMOTE significantly improved model fairness by balancing the class distribution.
- Random Forest outperformed the other models by a large margin, making it the best choice for deployment.

---

## 🚀 Suggestions for Improvement

1. Collect more data on customer behavior and demographics.
2. Include additional features like income level, employment status, and credit score.
3. Use real-time data to improve model accuracy.
4. Regularly retrain the model with new data to adapt to changing customer behavior.

---

## 🧑‍💻 Author

**Ajmal Patel**
---

## 📋 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AjmalPatel/Loan-Conversion-Prediction.git
   cd Loan-Conversion-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

3. Update the data file paths in the notebook to relative paths:
   ```python
   df1 = pd.read_csv("Part2 - Data1.csv")
   df2 = pd.read_csv("Part2 - Data2.csv")
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook Assignment2.ipynb
   ```

---

## 📄 License

This project is for educational purposes only.
