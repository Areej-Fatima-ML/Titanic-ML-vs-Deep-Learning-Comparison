# Titanic-ML-vs-Deep-Learning-Comparison
# Machine Learning vs Deep Learning – Titanic Survival Prediction

A comparative study applying classical Machine Learning algorithms and a Dense Neural Network to the Titanic dataset, evaluating and comparing their performance on a binary classification task.

This project was completed as part of a Deep Learning & Neural Networks course assignment (Atomcamp Arabia – AI Division).

## Dataset

- **Source:** `seaborn.load_dataset('titanic')`
- **Task:** Binary classification – predict passenger survival (`survived`: 0 or 1)
- **Samples:** 891 passengers
- **Features:** pclass, sex, age, sibsp, parch, fare, embarked, class, who, adult_male, alone

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Dataset overview (shape, data types, missing values)
- Summary statistics for numerical features
- Target class distribution plot
- Feature distributions split by target class
- Correlation heatmap
- Key insights documented

### 2. Data Preprocessing
- Dropped irrelevant/derived columns (`deck`, `embark_town`, `alive`, `who`, `class`)
- Handled missing values (median for age, mode for embarked, dropped remaining NaNs)
- Encoded categorical features (`sex`, `embarked`)
- Train/test split (80/20, stratified, `random_state=42`)
- Feature scaling with `StandardScaler` (fit on train only, to avoid data leakage)

### 3. Machine Learning Models
Trained and evaluated 4 classifiers using scikit-learn:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

For each model: training/test accuracy, classification report, confusion matrix heatmap, and interpretation.

### 4. Dense Neural Network
Built with TensorFlow/Keras:
- 2+ hidden Dense layers with ReLU activation
- Sigmoid output activation
- Dropout regularization (0.2–0.4)
- Binary crossentropy loss, Adam optimizer
- EarlyStopping callback (monitor `val_loss`, patience=10)
- Evaluated on Accuracy, AUC, Precision, Recall
- Training/validation accuracy & loss curves, ROC curve with AUC score

### 5. Comparison & Analysis
- Summary table comparing all 5 models (Accuracy, Precision, Recall, F1, ROC-AUC)
- Reflection on best-performing model, ML vs Deep Learning trade-offs on small datasets, and metric prioritization for medical-style diagnosis tasks

## Notebook Structure

1. Setup & Imports
2. Data Loading
3. EDA
4. Preprocessing
5. ML Models
6. Neural Network
7. Comparison
8. Conclusion

## Tools & Libraries

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- TensorFlow / Keras

## How to run

1. Open the notebook in Google Colab.
2. Run all cells in order (no dataset download required — loaded via seaborn).

## Author

Completed as part of the Deep Learning & Neural Networks course, Atomcamp Arabia – AI Division.
