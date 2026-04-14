from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from werkzeug.utils import secure_filename
import joblib
app = Flask(__name__)
## Load the Model 
model = joblib.load('models/Best_Arima_model.pkl')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the uploaded file and generate the forecast
            forecast_image = process_file(filepath)
            return render_template('index.html', forecast_image=forecast_image)

    return render_template('index.html', forecast_image=None)

def process_file(filepath):
    # Read CSV or Excel file
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    else:
        df = pd.read_excel(filepath, parse_dates=['Date'], index_col='Date')

    df = df.asfreq('D')  # Ensure daily frequency

    # Fit ARIMA model
    model = ARIMA(df['Sales'], order=(5,1,0))
    model_fit = model.fit()

    # Forecast next 30 days
    forecast_steps = 30
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Generate the plot
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df['Sales'], color='blue', label='Actual Sales')
    plt.plot(pd.date_range(df.index[-1], periods=forecast_steps+1, freq='D')[1:], forecast, 'r--', label='Forecasted Sales')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales Forecast Using ARIMA")
    plt.legend()
    plt.grid(True)

    # Save the plot
    forecast_image = os.path.join('static', 'forecast.png')
    plt.savefig(forecast_image)
    plt.close()

    return forecast_image

if __name__ == '__main__':
    app.run(debug=True)
