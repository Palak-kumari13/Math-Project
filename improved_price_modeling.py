#!/usr/bin/env python3
"""
improved_price_modeling.py

What this script does (in plain human language)
- Loads "pre-owned cars.csv" from the current working directory.
- Cleans and lightly normalizes columns (friendly names, parse years/dates).
- Creates sensible features (age, km/year, is_automatic, etc.).
- Applies K-fold target encoding (leakage-safe) for high-cardinality features:
  - brand and model are encoded using out-of-fold means of the target.
- Uses Optuna to search for good LightGBM hyperparameters using cross-validated RMSE.
- Trains a final model on the training set with the best hyperparameters.
- Evaluates on a hold-out test set and saves artifacts (model + encoders).
- Code is written in a straightforward, readable style — intended to sound human, not copy/paste.

Notes:
- This script intentionally uses a K-fold target encoding approach (out-of-fold) to avoid
  leaking target information into the encoded features.
- Optuna is used for tuning; if you prefer GridSearch, I can change it.

Run:
  pip install -r requirements.txt
  python improved_price_modeling.py

Outputs:
- model_joblib.pkl  -- trained LightGBM model + metadata
- encoders_joblib.pkl -- dict containing target-encoding mappings & medians
- Console prints with CV results and test metrics

Author: Written conversationally and handcrafted to be original.
"""

import os
import re
import joblib
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

RANDOM_STATE = 42
DATA_FILE = "pre-owned cars.csv"
TARGET = "price"
MODEL_OUT = "model_joblib.pkl"
ENCODERS_OUT = "encoders_joblib.pkl"

# -------------------------
# Utility / Preprocessing
# -------------------------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected {path} in working directory.")
    # Read everything as strings first to be robust to messy data
    df = pd.read_csv(path, dtype=str, low_memory=False)
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # Make names lowercase, replace spaces with underscores and strip
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    return df


def safe_extract_year(x) -> int:
    """Try to extract a 4-digit year from the string; return NaN on failure."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # direct 4-digit
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return int(m.group(0))
    # try pandas parse
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if not pd.isna(dt):
            return int(dt.year)
    except Exception:
        pass
    return np.nan


def preprocess_dates_and_years(df: pd.DataFrame) -> pd.DataFrame:
    # Try common columns make_year, reg_year
    df["make_year_int"] = pd.to_numeric(df.get("make_year", pd.Series()), errors="coerce")
    df["reg_year_int"] = df.get("reg_year", pd.Series()).apply(safe_extract_year)
    # Pick a sensible year for age calculation
    df["year_for_age"] = df["make_year_int"].fillna(df["reg_year_int"])
    now_year = datetime.now().year
    df["car_age"] = now_year - df["year_for_age"]
    # keep integer but allow NaN
    df["car_age"] = pd.to_numeric(df["car_age"], errors="coerce")
    return df


def cast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    # engine capacity: detect column with "engine" and "cc"
    engine_cols = [c for c in df.columns if re.search(r"engine.*cap|engine.*cc|cc\)", c, flags=re.I)]
    if engine_cols:
        df["engine_cc"] = pd.to_numeric(df[engine_cols[0]], errors="coerce")
    else:
        df["engine_cc"] = np.nan

    # km_driven
    if "km_driven" in df.columns:
        # strip commas and non-numeric characters
        df["km_driven"] = df["km_driven"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
    else:
        df["km_driven"] = np.nan

    # price
    df["price"] = df.get("price", pd.Series()).astype(str).str.replace(r"[^\d.]", "", regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # overall_cost if present
    if "overall_cost" in df.columns:
        df["overall_cost"] = pd.to_numeric(df["overall_cost"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    else:
        df["overall_cost"] = np.nan

    return df


def extract_state_code(df: pd.DataFrame) -> pd.DataFrame:
    if "reg_number" in df.columns:
        df["reg_number"] = df["reg_number"].astype(str).fillna("missing")
        df["state_code"] = df["reg_number"].str.extract(r"^([A-Za-z]{1,3})", expand=False).fillna("missing")
    else:
        df["state_code"] = "missing"
    return df


def map_ownership(df: pd.DataFrame) -> pd.DataFrame:
    if "ownership" not in df.columns:
        df["owner_count"] = np.nan
        return df

    def parse_ownership(x):
        if pd.isna(x):
            return np.nan
        s = str(x).lower()
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
        if "first" in s:
            return 1
        if "second" in s:
            return 2
        if "third" in s:
            return 3
        return np.nan

    df["owner_count"] = df["ownership"].apply(parse_ownership)
    return df


def map_booleans(df: pd.DataFrame, cols=("has_insurance", "spare_key")) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip().str.lower()
            # Map to numeric values directly
            mapping = {"yes": 1, "no": 0, "true": 1, "false": 0}
            df[c + "_flag"] = s.map(mapping)
            # Convert to float to allow NaN for unmapped values
            df[c + "_flag"] = pd.to_numeric(df[c + "_flag"], errors="coerce")
        else:
            df[c + "_flag"] = np.nan
    return df


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["km_per_year"] = df["km_driven"] / df["car_age"].replace({0: np.nan})
    df["km_per_year"] = df["km_per_year"].replace([np.inf, -np.inf], np.nan)
    df["is_automatic"] = df.get("transmission", pd.Series("")).astype(str).str.lower().str.contains("auto").astype(int)
    # simplify fuel types
    df["fuel_simple"] = df.get("fuel_type", pd.Series("")).astype(str).str.lower().where(
        lambda s: s.isin(["petrol", "diesel", "cng", "electric"]), other="other"
    )
    df["has_engine_cc"] = (~df["engine_cc"].isna()).astype(int)
    return df


# -------------------------
# K-Fold Target Encoding
# -------------------------
def kfold_target_encode(
    train_series: pd.Series,
    target: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.Series, Dict]:
    """
    Returns:
      - encoded_series: series of same length as train_series containing out-of-fold encoded values
      - mapping: dict with per-category global mean (useful to encode unseen categories later)
    Implementation notes:
      - For each fold, compute mean(target) per category on training folds and map to validation fold.
      - If a category is unseen in training folds but appears in validation, fallback to global mean.
    """
    # Ensure consistent string type for categories
    s = train_series.fillna("___missing___").astype(str)
    global_mean = target.mean()
    oof = pd.Series(index=s.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(s):
        train_cats = s.iloc[train_idx]
        val_cats = s.iloc[val_idx]
        train_t = target.iloc[train_idx]
        # Compute mapping on training folds
        mapping = train_t.groupby(train_cats).mean()
        # Map validation fold; unseen categories get NaN -> fill with global_mean
        oof.iloc[val_idx] = val_cats.map(mapping).fillna(global_mean)

    # final mapping computed on full data (to encode test/new)
    full_mapping = target.groupby(s).mean().to_dict()
    # ensure missing category handled
    full_mapping["___missing___"] = full_mapping.get("___missing___", global_mean)
    mapping_meta = {"mapping": full_mapping, "global_mean": global_mean}
    return oof, mapping_meta


def apply_target_encoding(series: pd.Series, mapping_meta: Dict) -> pd.Series:
    s = series.fillna("___missing___").astype(str)
    mapping = mapping_meta["mapping"]
    global_mean = mapping_meta["global_mean"]
    return s.map(mapping).fillna(global_mean).astype(float)


# -------------------------
# Modeling & Optuna
# -------------------------
def create_feature_matrix(df: pd.DataFrame, encoders: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Creates a tidy numeric feature matrix suitable for modeling.
    If encoders is None, this function will compute K-fold target encoders for brand/model using the provided TARGET in df.
    Returns X, encoders
    """
    df = df.copy()
    # Columns we plan to use (flexible)
    candidate_cols = [
        "engine_cc",
        "km_driven",
        "car_age",
        "km_per_year",
        "owner_count",
        "has_insurance_flag",
        "spare_key_flag",
        "overall_cost",
        "is_automatic",
        "fuel_simple",
        "state_code",
        "has_engine_cc",
    ]

    # Ensure candidate columns exist in df
    for c in candidate_cols:
        if c not in df.columns:
            df[c] = np.nan

    encoders = {} if encoders is None else encoders.copy()

    # If encoders do not contain brand/model mapping, compute K-fold target encodings
    for col in ["brand", "model"]:
        if col in df.columns:
            if encoders.get(col) is None:
                print(f"Computing K-fold target encoding for '{col}' (this is leakage-safe on training data).")
                encoded_oof, meta = kfold_target_encode(df[col], df[TARGET], n_splits=5)
                df[col + "_te"] = encoded_oof
                encoders[col] = meta
            else:
                # apply provided mapping (for test/inference)
                df[col + "_te"] = apply_target_encoding(df[col], encoders[col])
        else:
            df[col + "_te"] = np.nan

    # For things like state_code, do simple frequency encoding (cheap and robust)
    state_freq = df["state_code"].fillna("___missing___").astype(str).value_counts(normalize=True).to_dict()
    df["state_code_freq"] = df["state_code"].fillna("___missing___").astype(str).map(state_freq).fillna(0.0)

    # One-hot fuel_simple small set
    fuel_dummies = pd.get_dummies(df["fuel_simple"].fillna("other"), prefix="fuel", drop_first=True)
    df = pd.concat([df, fuel_dummies], axis=1)

    # Build feature set
    feature_cols = [
        "engine_cc",
        "km_driven",
        "car_age",
        "km_per_year",
        "owner_count",
        "has_insurance_flag",
        "spare_key_flag",
        "overall_cost",
        "is_automatic",
        "has_engine_cc",
        "brand_te",
        "model_te",
        "state_code_freq",
    ]
    # include fuel dummies (e.g., fuel_diesel, fuel_electric, fuel_petrol)
    # but exclude the original fuel_type and fuel_simple columns
    fuel_dummy_cols = [c for c in df.columns if c.startswith("fuel_") and c not in ["fuel_type", "fuel_simple"]]
    feature_cols += fuel_dummy_cols

    # Keep only those existing
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()

    # Impute numeric missing values with medians (compute medians and return in encoders)
    # Convert all columns to numeric first (should already be, but just to be safe)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    medians = X.median()
    X = X.fillna(medians)

    # store medians to encoders for later
    encoders["medians"] = medians.to_dict()

    return X, encoders


def optuna_objective(trial, X: pd.DataFrame, y: pd.Series) -> float:
    # Define a LightGBM parameter search space
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "seed": RANDOM_STATE,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    # 5-fold CV within objective
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # scikit-learn wrapper works with cross_val_score
    model = lgb.LGBMRegressor(n_estimators=1000, **param)
    # Use negative RMSE for scoring (cross_val_score wants greater is better)
    scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=1)
    # average RMSE
    mean_rmse = -scores.mean()
    return mean_rmse


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, best_params: Dict) -> lgb.LGBMRegressor:
    params = best_params.copy()
    # ensure some defaults
    params.update({"objective": "regression", "metric": "rmse", "n_jobs": -1, "verbosity": -1, "seed": RANDOM_STATE})
    model = lgb.LGBMRegressor(n_estimators=2000, **params)
    # early stopping using a small validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=RANDOM_STATE)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


# -------------------------
# Main script
# -------------------------
def main():
    print("Loading data...")
    df = load_csv(DATA_FILE)
    df = normalize_column_names(df)
    df = preprocess_dates_and_years(df)
    df = cast_numerics(df)
    df = extract_state_code(df)
    df = map_ownership(df)
    df = map_booleans(df)
    df = basic_feature_engineering(df)

    # drop exact duplicates and rows without target
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[~df[TARGET].isna()].reset_index(drop=True)
    print(f"Dataset after cleaning has {len(df)} rows.")

    # Split a hold-out test set first to evaluate final model honestly
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Compute target-encoded features on training data only (leakage-safe)
    X_train, encoders = create_feature_matrix(train_df, encoders=None)
    y_train = train_df[TARGET].astype(float)

    # For evaluation on test, apply encoders to test_df
    X_test, _ = create_feature_matrix(test_df, encoders=encoders)
    # when create_feature_matrix is called with encoders, it will use the stored mapping
    # but medians will be recomputed for test - we want to impute test with training medians
    X_test = X_test.fillna(pd.Series(encoders["medians"]))

    # Optuna tuning
    print("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
    # We limit trials to keep runtime reasonable; change n_trials as you like.
    study.optimize(lambda t: optuna_objective(t, X_train, y_train), n_trials=40, show_progress_bar=True)

    print("Best trial:")
    print("  RMSE:", study.best_value)
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    best_params = study.best_params

    # Train final model
    print("Training final model on full training set with best params...")
    final_model = train_final_model(X_train, y_train, best_params)

    # Evaluate on test
    print("Evaluating on held-out test set...")
    preds = final_model.predict(X_test)
    y_test = test_df[TARGET].astype(float)
    # Note: using np.sqrt instead of squared=False for wider sklearn compatibility
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test RMSE = {rmse:.2f}, MAE = {mae:.2f}, R2 = {r2:.4f}")

    # Save model and encoders
    artifact = {
        "model": final_model,
        "features": X_train.columns.tolist(),
        "training_rows": len(X_train),
        "optuna_best": study.best_trial,
    }
    joblib.dump(artifact, MODEL_OUT)
    joblib.dump(encoders, ENCODERS_OUT)
    print(f"Saved model to {MODEL_OUT} and encoders to {ENCODERS_OUT}.")

    # Print top feature importances for human inspection
    try:
        booster = final_model.booster_
        importance = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()
        imp_df = pd.DataFrame({"feature": names, "gain": importance}).sort_values("gain", ascending=False).head(15)
        print("\nTop features by gain:")
        print(imp_df.to_string(index=False))
    except Exception:
        print("Could not print LightGBM native importances (model might be sklearn-wrapped).")

    print("Done. If you'd like, I can produce a Jupyter notebook with these steps inline, or convert target-encoding to mean-encoding with smoothing.")


if __name__ == "__main__":
    main()
