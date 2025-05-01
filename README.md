# Credit Card Fraud Detection and Titanic Survival Prediction

This repository contains two end-to-end machine learning projects developed using Python and Jupyter Notebooks:

- `Credit_Card_Detection.ipynb`: Detects fraudulent credit card transactions.
- `Titanic_Survival.ipynb`: Predicts survival of passengers aboard the Titanic.

## 🧠 Projects Overview

### 1. Credit Card Fraud Detection
- **File**: `Credit_Card_Detection.ipynb`
- **Description**: Builds a machine learning model using `RandomForestClassifier` to identify fraudulent credit card transactions. The dataset is imbalanced, so **SMOTE** is used for oversampling.
- **Dataset**: `creditcard.csv`
- **Features**: Time, V1–V28 (anonymized), Amount, and Class (0 = genuine, 1 = fraud)
- **Workflow**:
  - Data preprocessing and scaling
  - SMOTE oversampling
  - Model training and testing
  - Evaluation using precision, recall, F1-score

### 2. Titanic Survival Prediction
- **File**: `Titanic_Survival.ipynb`
- **Description**: Predicts passenger survival based on dataset features like Age, Sex, Fare, and Pclass using a `RandomForestClassifier`. Uses **cross-validation** to avoid overfitting.
- **Dataset**: `tested.csv`
- **Workflow**:
  - Handling missing values and encoding categorical variables
  - Feature scaling
  - Model training with k-fold cross-validation
  - Evaluation and feature importance analysis
## Usage
1. Clone the repository:
git clone <repository-url>
2. Install required dependencies:
pip install pandas numpy scikit-learn imbalanced-learn
3. Upload the respective datasets (`creditcard.csv` for credit card detection, `tested.csv` for Titanic survival) to your Colab environment.
4. Open and run the Jupyter notebooks in Google Colab or a local Jupyter environment.

## Notes
- Ensure datasets are properly uploaded before running the notebooks.
- The Titanic model includes cross-validation to address potential overfitting observed with perfect accuracy (1.0) on the test set.
- For the credit card model, adjust SMOTE parameters or explore other classifiers if needed based on evaluation metrics.

## License
This project is for educational purposes only. Feel free to use and modify the code, but please credit the original source.

## Contact
For questions or contributions, please open an issue in the repository.
