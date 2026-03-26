import streamlit as st
import pickle
import json

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Uber Demand Forecaster", layout="centered")
st.title("🚖 Uber Spatio-Temporal Demand Forecaster")
st.markdown("Enter the spatial and weather conditions below to forecast Uber ride demand.")

# ---------------------------------------------------------
# 2. Load Pre-trained Artifacts (Matching train.py exactly)
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        # Loading models exactly as named by train.py
        lr = pickle.load(open("model_linear_regression.pkl", "rb"))
        rf = pickle.load(open("model_random_forest.pkl", "rb"))
        xgb = pickle.load(open("model_xgboost.pkl", "rb"))
        
        # Load vectorizer
        vec = pickle.load(open("vectorizer.pkl", "rb"))
        
        # Load borough map
        with open("borough_map.json", "r") as f:
            borough_map = json.load(f)
            
        return lr, rf, xgb, vec, borough_map
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Please ensure your .pkl and .json files are uploaded to GitHub.")
        return None, None, None, None, None

lr, rf, xgb, vec, borough_map = load_assets()

if lr is not None:
    # ---------------------------------------------------------
    # 3. User Input Interface
    # ---------------------------------------------------------
    st.header("Simulation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        borough_list = list(borough_map.keys()) if borough_map else ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        sim_borough = st.selectbox("Borough", borough_list)
        
        sim_hour = st.slider("Hour of Day (0-23)", 0, 23, 17)
        
        sim_dow = st.selectbox(
            "Day of Week", 
            options=[0, 1, 2, 3, 4, 5, 6], 
            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
        )
        
        sim_month = st.slider("Month (1-12)", 1, 12, 6)
        
        sim_hday = st.radio("Is it a Holiday?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    with col2:
        sim_temp = st.number_input("Temperature (°F)", min_value=-20.0, max_value=120.0, value=65.0)
        sim_spd = st.number_input("Wind Speed (mph)", min_value=0.0, max_value=50.0, value=5.0)
        sim_vsb = st.number_input("Visibility (miles)", min_value=0.0, max_value=10.0, value=10.0)
        sim_pcp = st.number_input("Precipitation (inches)", min_value=0.0, max_value=10.0, value=0.0)

    # ---------------------------------------------------------
    # 4. Prediction Logic
    # ---------------------------------------------------------
    if st.button("Predict Demand", type="primary"):
        
        # Build the dictionary exactly matching the pipeline's feature expectations
        d = {
            "cluster_id": borough_map[sim_borough],
            "Hour": sim_hour,
            "DayOfWeek": sim_dow,
            "Month": sim_month,
            "temp": sim_temp,
            "spd": sim_spd,
            "vsb": sim_vsb,
            "pcp01": sim_pcp,
            "hday": sim_hday,
            "IsWeekend": 1 if sim_dow >= 5 else 0,
            "IsRushHour": 1 if sim_hour in [7, 8, 9, 17, 18, 19] and sim_dow < 5 else 0 
        }

        # Vectorize the input using the loaded DictVectorizer
        X_input = vec.transform([d])
        
        # Generate predictions
        pred_lr = float(lr.predict(X_input)[0])
        pred_rf = float(rf.predict(X_input)[0])
        pred_xgb = float(xgb.predict(X_input)[0])
        
        # Display results
        st.divider()
        st.subheader("Forecasted Ride Requests")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Linear Regression", int(max(0, pred_lr)))
        res_col2.metric("Random Forest", int(max(0, pred_rf)))
        res_col3.metric("XGBoost", int(max(0, pred_xgb)))
