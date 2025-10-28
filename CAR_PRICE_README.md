# Car Price Prediction Project

## Overview
This project implements a machine learning pipeline to predict pre-owned car prices using LightGBM with K-fold target encoding and Optuna hyperparameter tuning.

## Features
- **K-fold Target Encoding**: Leakage-safe encoding for high-cardinality features (brand, model)
- **Feature Engineering**: Creates derived features like car_age, km_per_year, is_automatic
- **Hyperparameter Tuning**: Uses Optuna for optimizing LightGBM parameters
- **Cross-Validation**: 5-fold CV for robust model evaluation
- **Interactive Prediction**: User-friendly CLI for price predictions with "what-if" scenarios

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project expects a CSV file named `pre-owned cars.csv` in the project root directory with the following columns:

### Required Columns:
- `brand` - Car manufacturer (e.g., Maruti, Hyundai, Honda)
- `model` - Car model name (e.g., Swift, i20, City)
- `make year` or `Make Year` - Year of manufacture
- `km driven` or `KM Driven` - Kilometers driven
- `price` or `Price` - Target variable (sale price)

### Optional Columns (improve predictions if available):
- `transmission` or `Transmission` - Manual/Automatic
- `fuel type` or `Fuel Type` - Petrol/Diesel/CNG/Electric
- `engine capacity(CC)` - Engine capacity in CC
- `ownership` or `Ownership` - First/Second/Third owner
- `reg number` - Registration number (for state extraction)
- `reg year` - Registration year
- `has insurance` - Yes/No
- `spare key` - Yes/No
- `overall cost` - Overall maintenance cost

### Sample Data Format:
```csv
brand,model,make year,km driven,price,transmission,fuel type,engine capacity(CC),ownership
Maruti,Swift,2015,50000,450000,Manual,Petrol,1200,First Owner
Hyundai,i20,2017,35000,550000,Manual,Diesel,1400,First Owner
Honda,City,2016,60000,650000,Automatic,Petrol,1500,Second Owner
```

### Where to Get Data:
You can obtain pre-owned car data from:
1. **Kaggle**: Search for "used cars dataset" or "pre-owned cars"
2. **Data.gov.in**: Indian government open data portal
3. **Web Scraping**: From sites like CarDekho, Cars24 (with permission)
4. **Manual Collection**: Create your own dataset from market research

### Creating a Sample Dataset:
If you don't have real data, you can create a small sample file for testing:

```python
import pandas as pd

sample_data = {
    'brand': ['Maruti', 'Hyundai', 'Honda', 'Maruti', 'Toyota'],
    'model': ['Swift', 'i20', 'City', 'Alto', 'Innova'],
    'make year': [2015, 2017, 2016, 2014, 2018],
    'km driven': [50000, 35000, 60000, 70000, 40000],
    'price': [450000, 550000, 650000, 250000, 1200000],
    'transmission': ['Manual', 'Manual', 'Automatic', 'Manual', 'Manual'],
    'fuel type': ['Petrol', 'Diesel', 'Petrol', 'Petrol', 'Diesel'],
    'engine capacity(CC)': [1200, 1400, 1500, 800, 2500],
    'ownership': ['First Owner', 'First Owner', 'Second Owner', 'Third Owner', 'First Owner']
}

df = pd.DataFrame(sample_data)
df.to_csv('pre-owned cars.csv', index=False)
```

## Usage

### 1. Train the Model

First, train the model using your dataset:

```bash
python improved_price_modeling.py
```

This will:
- Load and preprocess the data
- Apply K-fold target encoding (leakage-safe)
- Perform hyperparameter tuning with Optuna (40 trials by default)
- Train the final LightGBM model
- Evaluate on test set
- Save model artifacts: `model_joblib.pkl` and `encoders_joblib.pkl`

**Expected Output:**
```
Loading data...
Dataset after cleaning has 2806 rows.
Train rows: 2244, Test rows: 562
Computing K-fold target encoding for 'brand' (this is leakage-safe on training data).
Computing K-fold target encoding for 'model' (this is leakage-safe on training data).
Starting hyperparameter tuning with Optuna...
[Progress bar showing trials...]
Best trial:
  RMSE: 85420.32
  Params:
    learning_rate: 0.0523
    num_leaves: 128
    ...
Test RMSE = 82345.67, MAE = 65432.10, R2 = 0.8456
```

### 2. Predict Car Prices Interactively

After training, use the interactive predictor:

```bash
python predict_car_price.py
```

This script will:
- Ask you for minimal car details
- Show predicted prices for both Manual and Automatic transmissions
- Allow comparison across different fuel types
- Display price differences and savings

**Example Session:**
```
🚗 CAR PRICE PREDICTOR 🚗
Please enter the following car details:

Brand (e.g., Maruti, Hyundai, Honda): Maruti
Model (e.g., Swift, i20, City): Swift
Make Year (e.g., 2015): 2015
KM Driven (e.g., 50000): 50000
Engine Capacity in CC (e.g., 1200): 1200

Fuel Type:
1. Petrol
2. Diesel
...

💰 ESTIMATED PRICES:
   Manual Transmission:................. ₹4,50,000.00
   Automatic Transmission:.............. ₹4,75,000.00

💡 Choosing Automatic will cost you approximately ₹25,000.00 more
```

## Project Structure

```
Math-Project/
├── requirements.txt              # Project dependencies
├── improved_price_modeling.py    # Main training script
├── predict_car_price.py          # Interactive prediction CLI
├── CAR_PRICE_README.md          # This file
├── pre-owned cars.csv           # Dataset (you need to provide)
├── model_joblib.pkl             # Trained model (generated)
└── encoders_joblib.pkl          # Feature encoders (generated)
```

## Technical Details

### Preprocessing Pipeline
1. **Column normalization**: Lowercase, underscore-separated names
2. **Date/Year parsing**: Extract years from various formats
3. **Feature engineering**:
   - `car_age` = current_year - make_year
   - `km_per_year` = km_driven / car_age
   - `is_automatic` = 1 if transmission contains "auto"
   - `fuel_simple` = standardized fuel type categories
4. **State extraction**: From registration numbers

### Encoding Strategy
- **High-cardinality features** (brand, model): K-fold target encoding
  - Out-of-fold means prevent data leakage
  - Global mean fallback for unseen categories
- **Low-cardinality features** (fuel type): One-hot encoding
- **Frequency encoding**: For state codes

### Model Details
- **Algorithm**: LightGBM (Gradient Boosting Decision Trees)
- **Hyperparameter tuning**: Optuna with TPE sampler
- **Cross-validation**: 5-fold stratified
- **Early stopping**: Prevents overfitting
- **Evaluation metrics**: RMSE, MAE, R²

### Why K-fold Target Encoding?
Target encoding can cause data leakage if done naively. We use K-fold out-of-fold encoding:
1. Split data into K folds
2. For each fold:
   - Compute target mean on other K-1 folds
   - Apply to current fold
3. This ensures each sample is encoded using only information from other samples

## Customization

### Adjust Number of Optuna Trials
In `improved_price_modeling.py`, line ~450:
```python
study.optimize(lambda t: optuna_objective(t, X_train, y_train), n_trials=40)
# Change n_trials to increase/decrease tuning time
```

### Modify Features
Edit the `candidate_cols` list in `create_feature_matrix()` function to add/remove features.

### Change Test Split
In `main()` function:
```python
train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
# Adjust test_size as needed (default 20%)
```

## Troubleshooting

### Error: FileNotFoundError: Expected pre-owned cars.csv
- Make sure the CSV file is in the same directory as the script
- Check the filename matches exactly (including spaces and case)

### Error: Model file not found
- Run `python improved_price_modeling.py` first to train the model
- This creates `model_joblib.pkl` and `encoders_joblib.pkl`

### Low R² Score
- Need more data (minimum 500-1000 samples recommended)
- Check data quality (missing values, outliers)
- Increase Optuna trials for better hyperparameters

### Prediction seems off
- Ensure input features match training data distribution
- Check if brand/model exists in training data
- Verify units (KM not miles, INR not USD)

## Performance Expectations

With a typical pre-owned cars dataset (2000+ samples):
- **Training time**: 15-30 minutes (depends on n_trials)
- **R² score**: 0.80-0.90
- **RMSE**: Usually 10-15% of mean price
- **Prediction time**: < 1 second

## Future Enhancements

Potential improvements:
- [ ] Add more feature engineering (brand popularity, depreciation rate)
- [ ] Ensemble multiple models (LightGBM + XGBoost + CatBoost)
- [ ] Add time-series analysis for price trends
- [ ] Web interface using Flask/Streamlit
- [ ] API endpoint for batch predictions
- [ ] Confidence intervals for predictions
- [ ] Automated data quality checks

## Contributing

Feel free to:
- Add new features
- Improve encoding strategies
- Optimize hyperparameters
- Add visualizations
- Create web interface

## License

This project is part of the Math-Project repository and follows its licensing terms.

## Contact

For questions or issues, please open an issue in the GitHub repository.
