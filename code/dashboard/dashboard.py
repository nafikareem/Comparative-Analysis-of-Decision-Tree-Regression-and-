import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data and models
df = pd.read_excel('Backhoe_Emission.xlsx')
rfr_gs = joblib.load('rfr_tuned.pkl')
dtr_gs = joblib.load('dtr_tuned.pkl')

# Drop rows with missing values
df.dropna(inplace=True)

# Initialize the scaler
scaler = StandardScaler()
X = df[['HP(watt)', 'Norm_MAP', 'RPM', 'Age', 'Engine_Tier', 'TEMP[C]']]
scaler.fit(X)

# Function to predict NOx emissions
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


st.title("NOx Emission Prediction")

# Initialize session state for input values
if 'hp' not in st.session_state:
    st.session_state.hp = 70000.0
if 'norm_map' not in st.session_state:
    st.session_state.norm_map = 0.5
if 'rpm' not in st.session_state:
    st.session_state.rpm = 1000
if 'age' not in st.session_state:
    st.session_state.age = 24
if 'engine_tier' not in st.session_state:
    st.session_state.engine_tier = 0
if 'temp' not in st.session_state:
    st.session_state.temp = 25.0

# Functions to update session state
def update_hp():
    st.session_state.hp = st.session_state.hp_input
def update_norm_map():
    st.session_state.norm_map = st.session_state.norm_map_input
def update_rpm():
    st.session_state.rpm = st.session_state.rpm_input
def update_age():
    st.session_state.age = st.session_state.age_input
def update_engine_tier():
    st.session_state.engine_tier = st.session_state.engine_tier_input
def update_temp():
    st.session_state.temp = st.session_state.temp_input

# Input fields for user input with sliders
st.number_input("Enter HP (watt)", min_value=65621.6, max_value=73824.3, value=st.session_state.hp, key='hp_input', on_change=update_hp)
st.slider("Adjust HP (watt)", min_value=65621.6, max_value=73824.3, value=st.session_state.hp, key='hp')

st.number_input("Enter Norm_MAP", min_value=0.0, max_value=1.0125, value=st.session_state.norm_map, key='norm_map_input', on_change=update_norm_map)
st.slider("Adjust Norm_MAP", min_value=0.0, max_value=1.0125, value=st.session_state.norm_map, key='norm_map')

st.number_input("Enter RPM", min_value=92, max_value=5000, value=st.session_state.rpm, key='rpm_input', on_change=update_rpm)
st.slider("Adjust RPM", min_value=92, max_value=5000, value=st.session_state.rpm, key='rpm')

st.number_input("Enter Age (Months)", min_value=12, max_value=96, value=st.session_state.age, key='age_input', on_change=update_age)
st.slider("Adjust Age (Months)", min_value=12, max_value=96, value=st.session_state.age, key='age')

st.selectbox("Select Engine Tier", [0, 1, 2], index=st.session_state.engine_tier, key='engine_tier_input', on_change=update_engine_tier)
st.select_slider("Adjust Engine Tier", options=[0, 1, 2], value=st.session_state.engine_tier, key='engine_tier')

st.number_input("Enter TEMP[C]", min_value=12.0, max_value=127.0, value=st.session_state.temp, key='temp_input', on_change=update_temp)
st.slider("Adjust TEMP[C]", min_value=12.0, max_value=127.0, value=st.session_state.temp, key='temp')

# Button to trigger prediction
if st.button("Predict NOx"):
    predicted_nox_rfr, predicted_nox_dtr = predict_nox_with_models(
        st.session_state.hp, 
        st.session_state.norm_map, 
        st.session_state.rpm, 
        st.session_state.age, 
        st.session_state.engine_tier, 
        st.session_state.temp
    )
    
    # Find the closest actual NOx value for error calculation
    closest_row = df.loc[(df['HP(watt)'] - st.session_state.hp).abs().idxmin()]
    actual_nox = closest_row['NOx[g/s]']

    error_rfr = abs(predicted_nox_rfr - actual_nox)
    error_dtr = abs(predicted_nox_dtr - actual_nox)

    # Display results in a DataFrame
    results_df = pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree"],
        "Predicted NOx [g/s]": [predicted_nox_rfr, predicted_nox_dtr],
        "Actual NOx [g/s]": [actual_nox, actual_nox],
        "Error [g/s]": [error_rfr, error_dtr]
    })

    st.subheader("Prediction Results")
    st.dataframe(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', x='Model', y='Error [g/s]', ax=ax, color=['blue', 'orange'], legend=False)
    ax.set_ylabel("Error [g/s]")
    ax.set_title("Prediction Error by Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center') 
    ax.bar_label(ax.containers[0], fmt='%.4f')

    avg_error = results_df['Error [g/s]'].mean()
    ax.axhline(y=avg_error, color='red', linestyle='--', label='Average Error')
    ax.text(ax.get_xlim()[1], avg_error, f'Avg: {avg_error:.4f}', 
            verticalalignment='bottom', horizontalalignment='right', color='red')

    plt.tight_layout()
    st.pyplot(fig)

    # error information
    st.write("### Error Analysis")
    st.write(f"Random Forest Prediction Error: {error_rfr:.4f} g/s")
    st.write(f"Decision Tree Prediction Error: {error_dtr:.4f} g/s")


st.markdown("<br><br><hr><p style='text-align: center;'>Made by Nafi Kareem</p>", unsafe_allow_html=True)