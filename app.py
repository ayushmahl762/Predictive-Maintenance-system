import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("predictive_maintenance_model.joblib")

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="centered")
st.title("🔧 AI-DRIVENgit add requirements.txt Predictive Maintenance")
st.markdown("Use machine data to assess potential **failure risks** and plan proactive maintenance.")

# User input sliders
st.header("📊 Input Machine Parameters")
col1, col2 = st.columns(2)

with col1:
    hours_used = st.slider("Hours Used", 0, 1000, 500)
    temperature = st.slider("Temperature (°C)", 30, 120, 70)
with col2:
    vibration = st.slider("Vibration Level", 0.0, 15.0, 5.0)
    days_since_service = st.slider("Days Since Last Service", 0, 365, 180)

# Prediction logic
features = np.array([[hours_used, temperature, vibration, days_since_service]])
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]  # Probability of failure

# Display results
st.header("🔍 Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High Risk of Failure ({probability*100:.1f}%)")
    reasons = []
    if hours_used > 800: reasons.append("Overused machine")
    if temperature > 85: reasons.append("High temperature")
    if vibration > 8: reasons.append("Abnormal vibration")
    if days_since_service > 300: reasons.append("Long gap since last service")
    if reasons:
        st.markdown("**Possible Risk Factors:**")
        for r in reasons:
            st.markdown(f"- {r}")
else:
    st.success(f"✅ Equipment is Healthy ({(1 - probability)*100:.1f}%)")

# Sensor Trends (Fake Data)
st.header("📈 Recent Sensor Trends")
st.markdown("These simulated graphs show how the machine is behaving over time.")

temps = np.random.normal(temperature, 4, 20)
vibs = np.random.normal(vibration, 0.8, 20)
time_points = list(range(-19, 1))

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(time_points, temps, color='orange', marker='o', label='Temperature (°C)')
ax[0].set_ylabel("°C")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(time_points, vibs, color='blue', marker='o', label="Vibration Level" )
ax[1].set_ylabel("Level")
ax[1].set_xlabel("Time (minutes ago)")
ax[1].grid(True)
ax[1].legend()

st.pyplot(fig)
