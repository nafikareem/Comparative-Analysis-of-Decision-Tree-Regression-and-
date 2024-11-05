# import Libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_excel('Backhoe_Emission.xlsx')
df.dropna(inplace=True)

# Load Tuned Models
dtr_gs = joblib.load('dtr_tuned.pkl')
rfr_gs = joblib.load('rfr_tuned.pkl')

# Split dataset
X = df[['HP(watt)', 'Norm_MAP', 'RPM', 'Age', 'Engine_Tier','TEMP[C]']]
y = df['NOx[g/s]']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# function to predict

def predict_nox_with_models(hp, norm_map, rpm, age, engine_tier, temp):
    
    input_data = pd.DataFrame({
        'HP(watt)': [hp],
        'Norm_MAP': [norm_map],
        'RPM': [rpm],
        'Age': [age],
        'Engine_Tier': [engine_tier],
        'TEMP[C]': [temp]
    })

    input_scaled = scaler.transform(input_data)

    nox_prediction_rfr = rfr_gs.predict(input_scaled)
    nox_prediction_dtr = dtr_gs.predict(input_scaled)

    return nox_prediction_rfr[0], nox_prediction_dtr[0]

# run predict
def run_predict():
    hp = float(input("Enter HP (watt) [65621.6 - 73824.3]: "))
    norm_map = float(input("Enter Norm_MAP [0 - 1.0125]: "))
    rpm = float(input("Enter RPM [92 - 5000]: "))
    age = int(input("Enter Age (Months) [12 - 96]: "))
    engine_tier = int(input("Enter Engine_Tier [0 - 2]: "))
    temp = float(input("Enter TEMP[C] [12 - 127]: "))

    predicted_nox_rfr, predicted_nox_dtr = predict_nox_with_models(hp, norm_map, rpm, age, engine_tier, temp)

    closest_row = df.loc[
        (df['HP(watt)'] - hp).abs().idxmin()  
    ]

    actual_nox = closest_row['NOx[g/s]']

    error_rfr = abs(predicted_nox_rfr - actual_nox)
    error_dtr = abs(predicted_nox_dtr - actual_nox)

    # Format output to 4 decimal places using format method
    print("Predicted NOx[g/s] using Random Forest: {:.4f} (Error: {:.4f})".format(predicted_nox_rfr, error_rfr))
    print("Predicted NOx[g/s] using Decision Tree: {:.4f} (Error: {:.4f})".format(predicted_nox_dtr, error_dtr))
    
run_predict()