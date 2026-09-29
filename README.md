# Heart Disease Prediction System — VS Code

A complete Flask + Machine Learning website for the topic **Heart Disease Prediction System**.

## Dataset

This project uses the **UCI Heart Disease dataset — Cleveland database**. It contains 303 records and 13 input features. The original target has values 0–4; for binary classification this project maps `0` to the negative class and `1–4` to the positive class.

Source: UCI Machine Learning Repository — Heart Disease, DOI 10.24432/C52P4X.

## ML Pipeline

1. Download/load the dataset.
2. Convert `?` values to missing values.
3. Convert columns to numeric.
4. Convert the original target to binary.
5. Split into 80% training and 20% testing using stratification.
6. Median-impute numeric features.
7. Most-frequent-impute categorical features.
8. Standardize numeric features.
9. One-hot encode categorical features.
10. Train a Random Forest classifier.
11. Evaluate accuracy, precision, recall, F1 and ROC-AUC.
12. Save the trained model with Joblib.
13. Flask loads the trained model and serves the prediction website.

## Run in VS Code — Windows

Open this folder in VS Code.

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate it

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

You will see the dataset size and test metrics in the terminal. The trained model is saved in `model/heart_disease_model.joblib`.

### 5. Start the website

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

The app will automatically train the model if the saved model does not exist.

## Project Structure

```text
Heart_Disease_Prediction_VSCode/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── heart_disease_cleveland.csv
│
├── model/
│   ├── heart_disease_model.joblib
│   └── metrics.json
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

## Important

The website output is a machine-learning classification result for an academic project. It is not a medical diagnosis and should not be used for clinical decisions.
