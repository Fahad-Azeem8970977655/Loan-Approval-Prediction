Loan Approval Prediction

This project builds a machine learning model to predict whether a loan application will be approved based on applicant and loan-related features.
The dataset is taken from Loan Approval Prediction Dataset (Kaggle)
.

📊 Dataset

The dataset contains features such as applicant income, education, employment status, loan amount, credit history, and property area. The target variable is Loan_Status (Approved / Not Approved).

🛠️ Steps Performed

Data cleaning and handling missing values

Encoding categorical features

Splitting into training and testing sets

Handling imbalanced data using SMOTE

Training models: Logistic Regression and Decision Tree

Evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix

Comparison of models

⚙️ Tools & Libraries

Python

Pandas

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib & Seaborn

🎯 Results

The project highlights model comparison for loan approval prediction, with focus on recall and F1-score due to imbalanced data. Logistic Regression provides baseline performance, while Decision Tree captures complex relationships.
