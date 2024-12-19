# Real-Estate-Price-Prediction-Using-Random-Forest-and-XGboost

A comprehensive project leveraging data analytics and machine learning to provide actionable insights into real estate trends. This repository demonstrates how predictive modeling, visualization, and interactive dashboards can transform real estate market analysis.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Dashboard](#dashboard)
- [Future Work](#future-work)

---

## Overview

### Purpose

This project aims to create a **Big Data Dashboard for Real-Time Real Estate Market Insights** by analyzing property characteristics, market dynamics, and customer preferences. Key goals include:

- Predicting house prices based on property features.
- Visualizing trends like regional price differences and market conditions.
- Providing insights into customer preferences and feature importance.

### Key Features

- **Predictive Modeling:** Uses Random Forest and XGBoost models for house price predictions.
- **Data Visualization:** Includes bar plots and heatmaps to highlight correlations and trends.
- **Interactive Dashboard:** A Flask-based web app to explore predictions and trends dynamically.

---

## Dataset

### Description

The dataset consists of real estate property details with features categorized as:

1. **Property Characteristics:**
   - `MSSubClass`, `MSZoning`, `LotFrontage`, `LotArea`, `Neighborhood`.
2. **Structural Details:**
   - `YearBuilt`, `YearRemodAdd`, `Foundation`, `BsmtQual`.
3. **Utility Features:**
   - `Heating`, `CentralAir`, `Electrical`, `GarageType`.
4. **Lifestyle Features:**
   - `Fireplaces`, `WoodDeckSF`, `PoolArea`.
5. **Market Features:**
   - `MoSold`, `YrSold`, `SaleType`, `SaleCondition`.

**Target Variable:**  
`SalePrice`: The final price of the property.

---

## Project Structure

```plaintext
real-estate-insights/
├── data/                       # Dataset storage
│   ├── raw/                    # Raw data
│   │   ├── train.csv
│   │   └── test.csv
│   ├── preprocessed/           # Processed datasets
│       ├── train_preprocessed.csv
│       └── test_preprocessed.csv
│
├── flask-setup/                # Flask app setup
│   ├── app/                    # Main Flask app
│   │   └── app.py
│   ├── static/                 # Static assets (CSS, JS)
│   │   └── styles.css
│   └── templates/              # HTML templates
│       ├── index.html
│       ├── predict.html
│       ├── virtualization.html
│
├── models/                     # Trained machine learning models
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── results/                    # Visualization outputs
│   ├── barplot.png
│   └── correlation_heatmap.png
│
├── src/                        # Source code and notebooks
│   ├── feature_preprocess.ipynb
│   ├── model_training.ipynb
│   └── new_feature_selected.ipynb
│
├── LICENSE                     # License file
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
```

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- Libraries: Pandas, NumPy, Matplotlib, XGBoost, Flask

### Installation Steps

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Kwiz-boss/Real-Estate-Price-Prediction-Using-Random-Forest-and-XGboost-.git
   cd Real-Estate-Price-Prediction-Using-Random-Forest-and-XGboost-
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv real-estate-env
   source real-estate-env/bin/activate  # Windows: real-estate-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the data:**
   - Place `train.csv` and `test.csv` in the `data/raw/` folder.
   - Run `src/feature_preprocess.ipynb` to preprocess the data.

---

## Model Training

To train the models, use the provided notebooks:

1. **Feature Preprocessing:**
   ```bash
   python src/feature_preprocess.ipynb
   ```

2. **Train the Models:**
   Open and run `src/model_training.ipynb` to train and save the Random Forest and XGBoost models.

---

## Evaluation

Evaluate the trained models using:
```bash
python src/model_training.ipynb
```

### Outputs:
- **Bar Plot:** Highlights important features (`results/barplot.png`).
- **Correlation Heatmap:** Shows feature relationships (`results/correlation_heatmap.png`).

---

## Dashboard

Run the Flask application to explore predictions and visualizations:
```bash
python flask-setup/app/app.py
```

Visit `http://127.0.0.1:5000` in your browser to interact with the dashboard.

---

## Future Work

- Incorporate real-time data updates via APIs.
- Expand feature engineering with interaction terms.
- Deploy the dashboard on Heroku or AWS for accessibility.
- Integrate more advanced visualizations using Plotly or Dash.

---

