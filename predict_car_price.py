#!/usr/bin/env python3
"""
predict_car_price.py

Interactive script for predicting car prices using the trained model.
User can input minimal features and see how different options affect the price.

Run:
  python predict_car_price.py

Requirements:
- model_joblib.pkl and encoders_joblib.pkl must exist (run improved_price_modeling.py first)
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime

MODEL_FILE = "model_joblib.pkl"
ENCODERS_FILE = "encoders_joblib.pkl"


def load_model_and_encoders():
    """Load trained model and encoders."""
    try:
        artifact = joblib.load(MODEL_FILE)
        encoders = joblib.load(ENCODERS_FILE)
        return artifact["model"], encoders, artifact["features"]
    except FileNotFoundError:
        print(f"Error: {MODEL_FILE} or {ENCODERS_FILE} not found.")
        print("Please run 'python improved_price_modeling.py' first to train the model.")
        exit(1)


def get_user_input():
    """Get car details from user interactively."""
    print("\n" + "="*60)
    print("🚗 CAR PRICE PREDICTOR 🚗")
    print("="*60)
    print("\nPlease enter the following car details:")
    print("(Press Enter to use default values where applicable)\n")
    
    # Brand
    brand = input("Brand (e.g., Maruti, Hyundai, Honda): ").strip()
    if not brand:
        brand = "Maruti"
    
    # Model
    model = input("Model (e.g., Swift, i20, City): ").strip()
    if not model:
        model = "Swift"
    
    # Make year
    while True:
        make_year_str = input("Make Year (e.g., 2015): ").strip()
        if not make_year_str:
            make_year = 2015
            break
        try:
            make_year = int(make_year_str)
            if 1990 <= make_year <= datetime.now().year:
                break
            else:
                print(f"Please enter a year between 1990 and {datetime.now().year}")
        except ValueError:
            print("Please enter a valid year")
    
    # KM driven
    while True:
        km_str = input("KM Driven (e.g., 50000): ").strip()
        if not km_str:
            km_driven = 50000
            break
        try:
            km_driven = float(km_str)
            if km_driven >= 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Engine capacity
    while True:
        engine_str = input("Engine Capacity in CC (e.g., 1200): ").strip()
        if not engine_str:
            engine_cc = 1200
            break
        try:
            engine_cc = float(engine_str)
            if engine_cc > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Fuel type
    print("\nFuel Type:")
    print("1. Petrol")
    print("2. Diesel")
    print("3. CNG")
    print("4. Electric")
    print("5. Other")
    fuel_choice = input("Choose fuel type (1-5): ").strip()
    fuel_map = {"1": "petrol", "2": "diesel", "3": "cng", "4": "electric", "5": "other"}
    fuel_type = fuel_map.get(fuel_choice, "petrol")
    
    # Ownership
    print("\nOwnership:")
    print("1. First owner")
    print("2. Second owner")
    print("3. Third owner")
    own_choice = input("Choose ownership (1-3): ").strip()
    owner_map = {"1": 1, "2": 2, "3": 3}
    owner_count = owner_map.get(own_choice, 1)
    
    # State code (optional)
    state_code = input("\nState Code (e.g., DL, MH, KA) [Optional]: ").strip().upper()
    if not state_code:
        state_code = "missing"
    
    return {
        "brand": brand,
        "model": model,
        "make_year": make_year,
        "km_driven": km_driven,
        "engine_cc": engine_cc,
        "fuel_type": fuel_type,
        "owner_count": owner_count,
        "state_code": state_code,
    }


def create_input_df(user_data, is_automatic):
    """Create a dataframe from user input with computed features."""
    now_year = datetime.now().year
    car_age = now_year - user_data["make_year"]
    km_per_year = user_data["km_driven"] / car_age if car_age > 0 else 0
    
    data = {
        "brand": user_data["brand"],
        "model": user_data["model"],
        "make_year_int": user_data["make_year"],
        "car_age": car_age,
        "km_driven": user_data["km_driven"],
        "km_per_year": km_per_year,
        "engine_cc": user_data["engine_cc"],
        "has_engine_cc": 1,
        "fuel_type": user_data["fuel_type"],
        "fuel_simple": user_data["fuel_type"],
        "owner_count": user_data["owner_count"],
        "is_automatic": 1 if is_automatic else 0,
        "transmission": "Automatic" if is_automatic else "Manual",
        "state_code": user_data["state_code"],
        "has_insurance_flag": np.nan,
        "spare_key_flag": np.nan,
        "overall_cost": np.nan,
        "price": 100000,  # dummy value (not used for prediction)
    }
    
    return pd.DataFrame([data])


def prepare_features(df, encoders, feature_cols):
    """Prepare features using the same pipeline as training."""
    # Apply target encoding for brand/model
    for col in ["brand", "model"]:
        if col in df.columns and col in encoders:
            s = df[col].fillna("___missing___").astype(str)
            mapping = encoders[col]["mapping"]
            global_mean = encoders[col]["global_mean"]
            df[col + "_te"] = s.map(mapping).fillna(global_mean).astype(float)
    
    # State code frequency
    state_freq = df["state_code"].fillna("___missing___").astype(str).value_counts(normalize=True).to_dict()
    df["state_code_freq"] = df["state_code"].fillna("___missing___").astype(str).map(state_freq).fillna(0.0)
    
    # One-hot fuel type
    fuel_dummies = pd.get_dummies(df["fuel_simple"].fillna("other"), prefix="fuel", drop_first=True)
    df = pd.concat([df, fuel_dummies], axis=1)
    
    # Ensure all required features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Select and order features
    X = df[feature_cols].copy()
    
    # Impute with training medians
    medians = encoders.get("medians", {})
    for col in X.columns:
        if col in medians:
            X[col] = X[col].fillna(medians[col])
        else:
            X[col] = X[col].fillna(0)
    
    return X


def main():
    # Load model
    print("Loading trained model...")
    model, encoders, feature_cols = load_model_and_encoders()
    print("Model loaded successfully!\n")
    
    # Get user input
    user_data = get_user_input()
    
    print("\n" + "="*60)
    print("PRICE ESTIMATION")
    print("="*60)
    
    # Predict for manual transmission
    df_manual = create_input_df(user_data, is_automatic=False)
    X_manual = prepare_features(df_manual, encoders, feature_cols)
    price_manual = model.predict(X_manual)[0]
    
    # Predict for automatic transmission
    df_auto = create_input_df(user_data, is_automatic=True)
    X_auto = prepare_features(df_auto, encoders, feature_cols)
    price_auto = model.predict(X_auto)[0]
    
    # Display results
    print(f"\n📊 Based on your inputs:")
    print(f"   Brand: {user_data['brand']}")
    print(f"   Model: {user_data['model']}")
    print(f"   Year: {user_data['make_year']} (Age: {datetime.now().year - user_data['make_year']} years)")
    print(f"   KM Driven: {user_data['km_driven']:,.0f} km")
    print(f"   Engine: {user_data['engine_cc']:.0f} CC")
    print(f"   Fuel: {user_data['fuel_type'].title()}")
    print(f"   Ownership: {'First' if user_data['owner_count']==1 else 'Second' if user_data['owner_count']==2 else 'Third'} Owner")
    
    print(f"\n💰 ESTIMATED PRICES:")
    print(f"   {'Manual Transmission:':.<40} ₹{price_manual:,.2f}")
    print(f"   {'Automatic Transmission:':.<40} ₹{price_auto:,.2f}")
    
    price_diff = price_auto - price_manual
    if abs(price_diff) > 100:
        if price_diff > 0:
            print(f"\n💡 Choosing Automatic will cost you approximately ₹{price_diff:,.2f} more")
        else:
            print(f"\n💡 Choosing Manual will cost you approximately ₹{abs(price_diff):,.2f} more")
    else:
        print(f"\n💡 Both transmission types have similar pricing")
    
    # Ask if user wants to try different fuel type
    print("\n" + "="*60)
    compare = input("\nWould you like to see how fuel type affects the price? (y/n): ").strip().lower()
    
    if compare == 'y':
        print("\n💡 FUEL TYPE COMPARISON (with Manual transmission):")
        fuel_types = ["petrol", "diesel", "cng", "electric"]
        fuel_prices = {}
        
        for fuel in fuel_types:
            temp_data = user_data.copy()
            temp_data["fuel_type"] = fuel
            df_temp = create_input_df(temp_data, is_automatic=False)
            X_temp = prepare_features(df_temp, encoders, feature_cols)
            fuel_prices[fuel] = model.predict(X_temp)[0]
        
        # Sort by price
        sorted_fuels = sorted(fuel_prices.items(), key=lambda x: x[1])
        
        print()
        for fuel, price in sorted_fuels:
            marker = "✓" if fuel == user_data["fuel_type"] else " "
            print(f"   {marker} {fuel.title():.<20} ₹{price:,.2f}")
        
        cheapest_fuel = sorted_fuels[0][0]
        if cheapest_fuel != user_data["fuel_type"]:
            savings = fuel_prices[user_data["fuel_type"]] - fuel_prices[cheapest_fuel]
            print(f"\n   Switching to {cheapest_fuel.title()} could save you ₹{savings:,.2f}")
    
    print("\n" + "="*60)
    print("Thank you for using Car Price Predictor!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
