import streamlit as st
import pandas as pd
import joblib

model = joblib.load("mobile_price_model.pkl")
scaler = joblib.load("scaler (1).pkl")
columns = joblib.load("column (1).pkl")


st.title("📱 Mobile Price Classification")
st.write("Enter the mobile specifications below.")


battery_power = st.number_input("Battery Power", min_value=501, max_value=1998, value=1000)

blue = st.selectbox("Bluetooth", [0, 1])

clock_speed = st.number_input("Clock Speed", min_value=0.5, max_value=3.0, value=2.0)

dual_sim = st.selectbox("Dual SIM", [0, 1])

fc = st.number_input("Front Camera (MP)", min_value=0, max_value=19, value=5)

four_g = st.selectbox("4G", [0, 1])

int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=64, value=32)

m_dep = st.number_input("Mobile Depth", min_value=0.1, max_value=1.0, value=0.5)

mobile_wt = st.number_input("Mobile Weight (g)", min_value=80, max_value=200, value=150)

n_cores = st.number_input("Number of Cores", min_value=1, max_value=8, value=4)

pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=20, value=10)

px_height = st.number_input("Pixel Height", min_value=0, max_value=1960, value=800)

px_width = st.number_input("Pixel Width", min_value=500, max_value=1998, value=1200)

ram = st.number_input("RAM", min_value=256, max_value=3998, value=2000)

sc_h = st.number_input("Screen Height", min_value=5, max_value=19, value=12)

sc_w = st.number_input("Screen Width", min_value=0, max_value=18, value=8)

talk_time = st.number_input("Talk Time (Hours)", min_value=2, max_value=20, value=10)

three_g = st.selectbox("3G", [0, 1])

touch_screen = st.selectbox("Touch Screen", [0, 1])

wifi = st.selectbox("WiFi", [0, 1])


if st.button("Predict"):
    input_data = pd.DataFrame(
    [[
        battery_power,
        blue,
        clock_speed,
        dual_sim,
        fc,
        four_g,
        int_memory,
        m_dep,
        mobile_wt,
        n_cores,
        pc,
        px_height,
        px_width,
        ram,
        sc_h,
        sc_w,
        talk_time,
        three_g,
        touch_screen,
        wifi
    ]],
    columns=columns
)
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.write("Prediction Value:", prediction[0])
    if prediction[0] == 0:
        st.success("Low Cost Mobile")

    elif prediction[0] == 1:
        st.success("Medium Cost Mobile")

    elif prediction[0] == 2:
        st.success("High Cost Mobile")

    else:
        st.success("Very High Cost Mobile")

