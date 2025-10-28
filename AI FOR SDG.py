import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from google.colab import files
 
# ==============================
# 🗂️ STEP 1: UPLOAD CSV FILES
# ==============================
print("👉 Please upload 'malaria_cholera_cases.csv' and 'climate_data.csv'")
uploaded = files.upload()
 
# Read uploaded files
cases_df = pd.read_csv("malaria_cholera_cases.csv")
climate_df = pd.read_csv("climate_data.csv")
 
# Convert date to datetime
cases_df['date'] = pd.to_datetime(cases_df['date'])
climate_df['date'] = pd.to_datetime(climate_df['date'])
 
# ==============================
# 🧩 STEP 2: MERGE & CLEAN DATA
# ==============================
merged = pd.merge(cases_df, climate_df, on=["date", "district"], how="inner")
merged = merged.sort_values(["district", "date"])
 
# Feature engineering: lag + rolling mean for rainfall
merged['rainfall_lag1'] = merged.groupby('district')['rainfall_mm'].shift(1)
merged['rainfall_roll3'] = merged.groupby('district')['rainfall_mm'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
 
# Drop NaNs created by shift
merged = merged.dropna().reset_index(drop=True)
 
print("✅ Merged dataset preview:")
display(merged.head())
 
# ==============================
# ⚙️ STEP 3: DEFINE FEATURES
# ==============================
malaria_features = ['rainfall_mm','temp_avg_c','humidity_avg','rainfall_lag1','rainfall_roll3','ndvi','pop_density']
cholera_features = ['rainfall_mm','flood_event','soil_moisture','water_temp_c' if 'water_temp_c' in merged.columns else 'temp_avg_c','sanitation_access','pop_density','rainfall_lag1']
 
# Define X, y for each disease
X_mal = merged[malaria_features]
y_mal = merged['malaria_cases']
 
X_chol = merged[cholera_features]
y_chol = merged['cholera_cases']
 
# ==============================
# 🤖 STEP 4: TRAIN MODELS
# ==============================
def train_and_evaluate(X, y, disease_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    print(f"\n📊 Results for {disease_name}:")
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    print("R²:", round(r2_score(y_test, y_pred), 2))
 
    # Plot predictions
    plt.figure(figsize=(6,4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(f"{disease_name} Predictions")
    plt.legend()
    plt.show()
 
    # Feature importance
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f"{disease_name} - Feature Importance")
    plt.show()
 
    return model
 
malaria_model = train_and_evaluate(X_mal, y_mal, "Malaria")
cholera_model = train_and_evaluate(X_chol, y_chol, "Cholera")
 
# ==============================
# 🔮 STEP 5: FUTURE PREDICTIONS
# ==============================
print("\n🔮 Sample prediction using last month’s climate data:")
latest = merged.iloc[-1:]
mal_pred = malaria_model.predict(latest[malaria_features])
chol_pred = cholera_model.predict(latest[cholera_features])
 
print(f"Predicted malaria cases next period: {int(mal_pred[0])}")
print(f"Predicted cholera cases next period: {int(chol_pred[0])}")