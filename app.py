import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ===================== DATA LOADING & MODEL TRAINING =====================
@st.cache_data
def load_and_train():
    # Load dataset
    df = pd.read_csv('https://github.com/MOULIoooo/Windmill-load-prediction-in-tamilnadu/blob/main/TN_WindFarms_Weekly_IO_Static_2023_2025.csv')

    # Encode Maintenance Schedule
    schedule_map = {'Biannual': 2, 'Monthly': 1, 'Quarterly': 4}
    df['Maintenance Schedule'] = df['Maintenance Schedule'].map(schedule_map)

    # Encode District
    le = LabelEncoder()
    df['District'] = le.fit_transform(df['District'])

    # Train-test split
    X = df.drop(['Net Output (MW)'], axis=1)
    y = df['Net Output (MW)']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBRegressor()
    model.fit(x_train, y_train)

    return model, le, schedule_map

model, le, schedule_map = load_and_train()

# ===================== UI =====================
st.title("🌬️ Tamil Nadu Wind Farm Output Predictor")
st.write("Predict **Net Output (MW)** based on wind farm parameters.")

# District list (sorted to match encoding)
districts = ['Ambasamudram', 'Aralvaimozhi', 'Ariyalur', 'Chengalpattu',
       'Coimbatore', 'Cuddalore', 'Dharmapuri', 'Dindigul', 'Erode',
       'Kallakurichi', 'Kangeyam', 'Kanyakumari', 'Karur', 'Kayathar',
       'Krishnagiri', 'Madurai', 'Muppandal', 'Nagapattinam', 'Namakkal',
       'Nilgiris', 'Ooty', 'Palladam', 'Panagudi', 'Periyakulam',
       'Pudukottai', 'Ramanathapuram', 'Salem', 'Sankarankoil',
       'Sivaganga', 'Tenkasi', 'Theni', 'Thiruvallur', 'Thoothukudi',
       'Tirunelveli', 'Tirupur', 'Tiruvannamalai', 'Udumalpet',
       'Valiyoor', 'Villupuram', 'Virudhunagar']

# Input form
with st.form("input_form"):
    year = st.number_input("Year", min_value=2023, max_value=2025, step=1)
    wind_speed = st.number_input("Wind Speed (m/s)")
    efficiency = st.number_input("Machine Efficiency")
    power_loss = st.number_input("Power Loss")
    maintenance_loss = st.number_input("Maintenance Loss")
    total_loss = st.number_input("Total Loss")
    lifespan = st.number_input("Machine Lifespan (yrs)")
    district = st.selectbox("District", districts)
    hub_height = st.number_input("Hub Height (m)")
    rotor_area = st.number_input("Rotor Swept Area (m²)")
    loss_factor = st.number_input("Loss Factor")
    schedule = st.selectbox("Maintenance Schedule", list(schedule_map.keys()))

    submitted = st.form_submit_button("Predict Net Output")

# Prediction
if submitted:
    # Encode District & Maintenance Schedule
    district_encoded = le.transform([district])[0]
    schedule_encoded = schedule_map[schedule]

    # Prepare features
    features = np.array([[year, wind_speed, efficiency, power_loss,
                          maintenance_loss, total_loss, lifespan,
                          district_encoded, hub_height, rotor_area,
                          loss_factor, schedule_encoded]])

    # Predict
    prediction = model.predict(features)[0]
    st.success(f"✅ Predicted Net Output: **{prediction:.2f} MW**")
