import os, json, joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from train_model import train, MODEL_PATH, METRICS_PATH

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Automatically train if the model is not present.
if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
    MODEL, METRICS = train()
else:
    MODEL = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        METRICS = json.load(f)

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

@app.route('/')
def home():
    return render_template('index.html', metrics=METRICS)

@app.route('/api/metrics')
def metrics():
    return jsonify(METRICS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        row = {
            'age': float(data['age']),
            'sex': int(data['sex']),
            'cp': int(data['cp']),
            'trestbps': float(data['trestbps']),
            'chol': float(data['chol']),
            'fbs': int(data['fbs']),
            'restecg': int(data['restecg']),
            'thalach': float(data['thalach']),
            'exang': int(data['exang']),
            'oldpeak': float(data['oldpeak']),
            'slope': int(data['slope']),
            'ca': int(data['ca']),
            'thal': int(data['thal'])
        }
        X = pd.DataFrame([row], columns=FEATURES)
        prediction = int(MODEL.predict(X)[0])
        probability = float(MODEL.predict_proba(X)[0][1] * 100)

        if prediction == 1:
            title = 'Higher Likelihood'
            description = 'The trained model classified this record as positive for the heart-disease target.'
        else:
            title = 'Lower Likelihood'
            description = 'The trained model classified this record as negative for the heart-disease target.'

        return jsonify({
            'success': True,
            'prediction': prediction,
            'title': title,
            'description': description,
            'probability': round(probability, 1)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print('\nHeart Disease Prediction System')
    print('Open http://127.0.0.1:5000 in your browser')
    app.run(host='127.0.0.1', port=5000, debug=True)
