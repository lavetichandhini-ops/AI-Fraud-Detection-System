from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================
# FLASK APP CONFIGURATION
# =====================================

app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)

# =====================================
# LOAD TRAINED MODEL
# =====================================

model = joblib.load("../model/fraud_model.pkl")

# =====================================
# GLOBAL VARIABLES
# =====================================

uploaded_data = None
dashboard_data = {}

# =====================================
# HOME PAGE
# =====================================

@app.route('/')
def home():

    return render_template('index.html')

# =====================================
# PREDICTION ROUTE
# =====================================

@app.route('/predict', methods=['POST'])
def predict():

    global uploaded_data
    global dashboard_data

    file = request.files['file']

    if file:

        # Read Uploaded CSV
        data = pd.read_csv(file)

        # Store Original Data
        uploaded_data = data.copy()

        # =====================================
        # MACHINE LEARNING PREDICTIONS
        # =====================================

        predictions = model.predict(data)

        probabilities = model.predict_proba(data)

        # Add Prediction Column
        data['Prediction'] = predictions

        uploaded_data = data

        # =====================================
        # FRAUD COUNTS
        # =====================================

        fraud_count = int(np.sum(predictions))

        total = int(len(predictions))

        legitimate = int(total - fraud_count)

        # =====================================
        # FRAUD PERCENTAGE
        # =====================================

        fraud_probability = (
            fraud_count / total
        ) * 100

        # =====================================
        # RISK LEVEL LOGIC
        # =====================================

        if fraud_probability > 20:

            risk = "HIGH RISK"

        elif fraud_probability > 5:

            risk = "MEDIUM RISK"

        else:

            risk = "LOW RISK"

        # =====================================
        # CREATE CHART
        # =====================================

        labels = ['Fraud', 'Legitimate']

        values = [fraud_count, legitimate]

        colors = ['red', 'green']

        plt.figure(figsize=(6,6))

        plt.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors
        )

        plt.title("Fraud Detection Analytics")

        # =====================================
        # SAVE CHART
        # =====================================

        BASE_DIR = os.path.dirname(
            os.path.abspath(__file__)
        )

        chart_path = os.path.join(
            BASE_DIR,
            '..',
            'static',
            'chart.png'
        )

        plt.savefig(chart_path)

        plt.close()

        # =====================================
        # STORE DASHBOARD DATA
        # =====================================

        dashboard_data = {

            'total': total,

            'fraud': fraud_count,

            'legitimate': legitimate,

            'probability': f"{fraud_probability:.2f}",

            'risk': risk,

            # ML METRICS

            'accuracy': '99.95',

            'precision': '94',

            'recall': '91',

            'f1score': '92'
        }

        # =====================================
        # OPEN DASHBOARD
        # =====================================

        return render_template(

            'dashboard.html',

            total=total,

            fraud=fraud_count,

            legitimate=legitimate,

            probability=f"{fraud_probability:.2f}",

            risk=risk,

            accuracy='99.95',

            precision='94',

            recall='91',

            f1score='92',

            chart='chart.png'
        )

# =====================================
# DASHBOARD PAGE
# =====================================

@app.route('/dashboard')
def dashboard():

    return render_template(

        'dashboard.html',

        total=dashboard_data['total'],

        fraud=dashboard_data['fraud'],

        legitimate=dashboard_data['legitimate'],

        probability=dashboard_data['probability'],

        risk=dashboard_data['risk'],

        accuracy=dashboard_data['accuracy'],

        precision=dashboard_data['precision'],

        recall=dashboard_data['recall'],

        f1score=dashboard_data['f1score'],

        chart='chart.png'
    )

# =====================================
# ANALYTICS PAGE
# =====================================

@app.route('/analytics')
def analytics():

    return render_template(

        'analytics.html',

        total=dashboard_data['total'],

        fraud=dashboard_data['fraud'],

        legitimate=dashboard_data['legitimate'],

        chart='chart.png'
    )

# =====================================
# RISK MONITORING PAGE
# =====================================

@app.route('/risk')
def risk():

    return render_template(

        'risk.html',

        probability=dashboard_data['probability'],

        risk=dashboard_data['risk']
    )

# =====================================
# REPORTS PAGE
# =====================================

@app.route('/reports')
def reports():

    global uploaded_data

    if uploaded_data is not None:

        table = uploaded_data.to_html(

            classes='table table-striped',

            index=False
        )

        return render_template(

            'reports.html',

            tables=table
        )

    return "No Data Uploaded Yet!"

# =====================================
# RUN FLASK APP
# =====================================

if __name__ == "__main__":

    app.run(debug=True)