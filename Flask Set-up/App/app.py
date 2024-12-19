# Flask Dashboard Code

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import plotly.express as px
import os

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# Load dataset for visualizations
data_path = "C:/Users/enock/Downloads/data/new_train.csv"
df = pd.read_csv(data_path)

# Helper function for predictions
def predict_price(inputs):
    features = pd.DataFrame([inputs])
    rf_prediction = rf_model.predict(features)[0]
    xgb_prediction = xgb_model.predict(features)[0]
    ensemble_prediction = (rf_prediction + xgb_prediction) / 2
    return {
        "Random Forest Prediction": rf_prediction,
        "XGBoost Prediction": xgb_prediction,
        "Ensemble Prediction": ensemble_prediction
    }

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/visualizations')
def visualizations():
    # Price distribution
    fig = px.histogram(df, x='SalePrice', nbins=50, title='Sale Price Distribution')
    price_dist = fig.to_html(full_html=False)

    # Correlation heatmap
    correlation_fig = px.imshow(df.corr(), title='Correlation Heatmap', text_auto=True)
    correlation_heatmap = correlation_fig.to_html(full_html=False)

    return render_template('visualizations.html', price_dist=price_dist, correlation_heatmap=correlation_heatmap)

@app.route('/filters', methods=['GET', 'POST'])
def filters():
    if request.method == 'POST':
        filters = request.form.to_dict()
        filtered_df = df.copy()

        # Apply filters
        for key, value in filters.items():
            if value:
                filtered_df = filtered_df[filtered_df[key] == value]

        # Generate visualization for filtered data
        fig = px.scatter(filtered_df, x='LotArea', y='SalePrice', color='Neighborhood', title='Filtered Sale Prices')
        scatter_plot = fig.to_html(full_html=False)

        return render_template('filters.html', scatter_plot=scatter_plot, filters=filters)

    return render_template('filters.html', filters={})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        for key in input_data:
            try:
                input_data[key] = float(input_data[key])
            except ValueError:
                pass

        predictions = predict_price(input_data)
        return render_template('predict.html', predictions=predictions)

    return render_template('predict.html', predictions=None)

if __name__ == '__main__':
    # Ensure templates and static folders exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
