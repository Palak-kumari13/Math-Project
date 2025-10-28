#!/usr/bin/env python3
"""
generate_sample_dataset.py

Generates a sample pre-owned cars dataset for testing the price prediction model.
Run this if you don't have real data yet.

Usage:
    python generate_sample_dataset.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

# Sample brands and their models
BRANDS_MODELS = {
    "Maruti": ["Swift", "Alto", "Wagon R", "Baleno", "Vitara Brezza", "Dzire", "Ertiga"],
    "Hyundai": ["i20", "Grand i10", "Creta", "Verna", "Venue", "Santro", "Elantra"],
    "Honda": ["City", "Amaze", "Jazz", "WR-V", "Civic", "CR-V", "Accord"],
    "Tata": ["Nexon", "Altroz", "Harrier", "Safari", "Tiago", "Tigor", "Punch"],
    "Toyota": ["Innova", "Fortuner", "Glanza", "Urban Cruiser", "Camry", "Corolla"],
    "Mahindra": ["XUV500", "Scorpio", "Thar", "XUV300", "Bolero", "Marazzo"],
    "Ford": ["EcoSport", "Figo", "Aspire", "Endeavour", "Freestyle"],
    "Renault": ["Kwid", "Duster", "Triber", "Captur", "Lodgy"],
    "Volkswagen": ["Polo", "Vento", "Tiguan", "Passat"],
    "Kia": ["Seltos", "Sonet", "Carnival"],
}

STATE_CODES = ["DL", "MH", "KA", "TN", "AP", "TS", "GJ", "RJ", "UP", "HR", "PB", "WB"]
FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric"]
TRANSMISSIONS = ["Manual", "Automatic"]
OWNERSHIPS = ["First Owner", "Second Owner", "Third Owner"]

# Base prices for different brands (rough estimates in INR)
BASE_PRICES = {
    "Maruti": 400000,
    "Hyundai": 500000,
    "Honda": 600000,
    "Tata": 450000,
    "Toyota": 800000,
    "Mahindra": 700000,
    "Ford": 550000,
    "Renault": 400000,
    "Volkswagen": 650000,
    "Kia": 750000,
}

# Engine capacities for different segments
ENGINE_CC = {
    "small": [800, 1000, 1200],
    "medium": [1200, 1400, 1500, 1600],
    "large": [1800, 2000, 2200, 2500],
}


def generate_dataset(n_samples=500):
    """Generate synthetic car dataset."""
    
    data = []
    current_year = datetime.now().year
    
    for i in range(n_samples):
        # Select brand and model
        brand = np.random.choice(list(BRANDS_MODELS.keys()))
        model = np.random.choice(BRANDS_MODELS[brand])
        
        # Make year (between 2010 and 2023)
        make_year = np.random.randint(2010, 2024)
        car_age = current_year - make_year
        
        # KM driven (based on age and usage pattern)
        avg_km_per_year = np.random.normal(12000, 4000)  # avg km/year with variation
        km_driven = max(1000, int(car_age * avg_km_per_year + np.random.normal(0, 5000)))
        
        # Engine capacity (based on brand segment)
        if brand in ["Maruti", "Renault"]:
            engine_segment = "small"
        elif brand in ["Hyundai", "Honda", "Tata", "Ford"]:
            engine_segment = "medium"
        else:
            engine_segment = "large"
        engine_cc = np.random.choice(ENGINE_CC[engine_segment])
        
        # Transmission (automatic more common in newer cars)
        auto_prob = 0.15 if make_year < 2015 else 0.35
        transmission = np.random.choice(TRANSMISSIONS, p=[1-auto_prob, auto_prob])
        
        # Fuel type
        fuel_weights = [0.5, 0.35, 0.1, 0.05]  # Petrol, Diesel, CNG, Electric
        if make_year >= 2020:
            fuel_weights = [0.45, 0.35, 0.08, 0.12]  # More electric in recent years
        fuel_type = np.random.choice(FUEL_TYPES, p=fuel_weights)
        
        # Ownership (older cars more likely to have changed hands)
        if car_age <= 3:
            ownership_weights = [0.9, 0.08, 0.02]
        elif car_age <= 7:
            ownership_weights = [0.5, 0.4, 0.1]
        else:
            ownership_weights = [0.2, 0.5, 0.3]
        ownership = np.random.choice(OWNERSHIPS, p=ownership_weights)
        
        # State code
        state_code = np.random.choice(STATE_CODES)
        reg_number = f"{state_code}{np.random.randint(1, 20):02d}{np.random.choice(['A', 'B', 'C', 'D'])}{np.random.randint(1000, 9999)}"
        
        # Insurance and spare key (random with bias)
        has_insurance = np.random.choice(["Yes", "No"], p=[0.7, 0.3])
        spare_key = np.random.choice(["Yes", "No"], p=[0.6, 0.4])
        
        # Calculate price based on multiple factors
        base_price = BASE_PRICES[brand]
        
        # Depreciation (10-12% per year)
        depreciation_rate = 0.11
        price = base_price * ((1 - depreciation_rate) ** car_age)
        
        # Adjust for km driven (higher km = lower price)
        km_factor = 1 - (km_driven / 200000) * 0.15
        price *= max(0.5, km_factor)
        
        # Adjust for transmission (automatic adds value)
        if transmission == "Automatic":
            price *= 1.15
        
        # Adjust for fuel type
        fuel_multipliers = {"Petrol": 1.0, "Diesel": 1.1, "CNG": 0.95, "Electric": 1.2}
        price *= fuel_multipliers[fuel_type]
        
        # Adjust for ownership
        ownership_multipliers = {"First Owner": 1.0, "Second Owner": 0.92, "Third Owner": 0.85}
        price *= ownership_multipliers[ownership]
        
        # Adjust for engine size
        engine_multiplier = 1 + (engine_cc - 1000) / 10000
        price *= engine_multiplier
        
        # Add some random noise (±10%)
        noise = np.random.uniform(0.9, 1.1)
        price *= noise
        
        # Round to nearest 5000
        price = round(price / 5000) * 5000
        
        # Create entry
        entry = {
            "brand": brand,
            "model": model,
            "make year": make_year,
            "reg year": make_year,  # same as make year for simplicity
            "km driven": km_driven,
            "price": int(price),
            "transmission": transmission,
            "fuel type": fuel_type,
            "engine capacity(CC)": engine_cc,
            "ownership": ownership,
            "reg number": reg_number,
            "has insurance": has_insurance,
            "spare key": spare_key,
            "title": f"{brand} {model} {make_year}",
            "overall cost": int(price * 0.05),  # rough estimate of maintenance
        }
        
        data.append(entry)
    
    return pd.DataFrame(data)


def main():
    print("Generating sample pre-owned cars dataset...")
    
    # Generate dataset
    df = generate_dataset(n_samples=500)
    
    # Save to CSV
    output_file = "pre-owned cars.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Generated {len(df)} samples")
    print(f"✓ Saved to '{output_file}'")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Price range: ₹{df['price'].min():,} - ₹{df['price'].max():,}")
    print(f"Mean price: ₹{df['price'].mean():,.0f}")
    print(f"Median price: ₹{df['price'].median():,.0f}")
    
    print(f"\nBrands: {', '.join(df['brand'].unique())}")
    print(f"Total models: {df['model'].nunique()}")
    print(f"Year range: {df['make year'].min()} - {df['make year'].max()}")
    
    print("\nTop 5 most common brands:")
    print(df['brand'].value_counts().head())
    
    print("\nTransmission distribution:")
    print(df['transmission'].value_counts())
    
    print("\nFuel type distribution:")
    print(df['fuel type'].value_counts())
    
    print("\nOwnership distribution:")
    print(df['ownership'].value_counts())
    
    print("\n" + "="*60)
    print("Dataset is ready! You can now run:")
    print("  python improved_price_modeling.py")
    print("="*60)


if __name__ == "__main__":
    main()
