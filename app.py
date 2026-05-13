import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Household Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run 'train_model.py' first to train and save the model.")
        st.stop()

# Main title
st.markdown('<h1 class="main-header">⚡ Household Energy Consumption Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/lightning-bolt.png", width=100)
    st.title("Navigation")
    page = st.radio("Select Mode:", ["🏠 Single Prediction", "📊 Batch Prediction", "📈 Model Info"])
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This app predicts household appliance energy consumption using machine learning based on indoor climate and outdoor weather conditions.")

# Load model
try:
    model, scaler = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.warning("Model not loaded. Some features may be unavailable.")

# Feature names (28 features - excluding 'date' and 'Appliances' target)
feature_names = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 
                 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
                 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']

# Page 1: Single Prediction
if page == "🏠 Single Prediction":
    st.markdown('<h2 class="sub-header">Enter Environmental Parameters</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🏠 Indoor Climate (Rooms 1-3)")
        lights = st.slider("Lights Energy (Wh)", 0, 100, 40)
        T1 = st.number_input("Room 1 Temperature (°C)", 15.0, 30.0, 20.0, 0.1)
        RH_1 = st.number_input("Room 1 Humidity (%)", 20.0, 70.0, 45.0, 0.1)
        T2 = st.number_input("Room 2 Temperature (°C)", 15.0, 30.0, 19.0, 0.1)
        RH_2 = st.number_input("Room 2 Humidity (%)", 20.0, 70.0, 44.0, 0.1)
        T3 = st.number_input("Room 3 Temperature (°C)", 15.0, 30.0, 19.5, 0.1)
        RH_3 = st.number_input("Room 3 Humidity (%)", 20.0, 70.0, 44.5, 0.1)
        
    with col2:
        st.markdown("#### 🏠 Indoor Climate (Rooms 4-6)")
        T4 = st.number_input("Room 4 Temperature (°C)", 15.0, 30.0, 19.0, 0.1)
        RH_4 = st.number_input("Room 4 Humidity (%)", 20.0, 70.0, 45.0, 0.1)
        T5 = st.number_input("Room 5 Temperature (°C)", 15.0, 30.0, 18.5, 0.1)
        RH_5 = st.number_input("Room 5 Humidity (%)", 20.0, 70.0, 45.5, 0.1)
        T6 = st.number_input("Room 6 Temperature (°C)", 15.0, 30.0, 18.0, 0.1)
        RH_6 = st.number_input("Room 6 Humidity (%)", 20.0, 70.0, 46.0, 0.1)
        
        st.markdown("#### 🏠 Indoor Climate (Rooms 7-9)")
        T7 = st.number_input("Room 7 Temperature (°C)", 15.0, 30.0, 17.5, 0.1)
        RH_7 = st.number_input("Room 7 Humidity (%)", 20.0, 70.0, 38.0, 0.1)
        
    with col3:
        T8 = st.number_input("Room 8 Temperature (°C)", 15.0, 30.0, 17.0, 0.1)
        RH_8 = st.number_input("Room 8 Humidity (%)", 20.0, 70.0, 40.0, 0.1)
        T9 = st.number_input("Room 9 Temperature (°C)", 15.0, 30.0, 17.0, 0.1)
        RH_9 = st.number_input("Room 9 Humidity (%)", 20.0, 70.0, 45.0, 0.1)
        
        st.markdown("#### 🌤️ Outdoor Weather")
        T_out = st.number_input("Outdoor Temperature (°C)", -10.0, 40.0, 6.0, 0.1)
        Press_mm_hg = st.number_input("Pressure (mmHg)", 720.0, 770.0, 733.5, 0.1)
        RH_out = st.number_input("Outdoor Humidity (%)", 20.0, 100.0, 92.0, 0.1)
        Windspeed = st.number_input("Wind Speed (m/s)", 0.0, 15.0, 7.0, 0.1)
        Visibility = st.number_input("Visibility (km)", 0.0, 100.0, 60.0, 0.1)
        Tdewpoint = st.number_input("Dew Point (°C)", -10.0, 30.0, 5.0, 0.1)
        
        st.markdown("#### 🔢 Random Variables")
        rv1 = st.number_input("Random Variable 1", 0.0, 50.0, 13.0, 0.1)
        rv2 = st.number_input("Random Variable 2", 0.0, 50.0, 13.0, 0.1)
    
    st.markdown("---")
    
    if st.button("🔮 Predict Energy Consumption", use_container_width=True):
        if model_loaded:
            # Prepare input data
            input_data = pd.DataFrame([[lights, T1, RH_1, T2, RH_2, T3, RH_3, T4, RH_4,
                                       T5, RH_5, T6, RH_6, T7, RH_7, T8, RH_8, T9, RH_9,
                                       T_out, Press_mm_hg, RH_out, Windspeed, Visibility,
                                       Tdewpoint, rv1, rv2]], columns=feature_names)
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Energy", f"{prediction:.2f} Wh", "Appliances")
            with col2:
                st.metric("Total Energy", f"{prediction + lights:.2f} Wh", "Appliances + Lights")
            with col3:
                daily_estimate = (prediction + lights) * 144  # 144 intervals per day (10-min each)
                st.metric("Daily Estimate", f"{daily_estimate/1000:.2f} kWh", "24 hours")
            
            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Appliances Energy Consumption"},
                delta={'reference': 100},
                gauge={
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 100], 'color': "#90EE90"},
                        {'range': [100, 300], 'color': "#FFD700"},
                        {'range': [300, 500], 'color': "#FF6B6B"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 400
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Model not available. Please check model files.")

# Page 2: Batch Prediction
elif page == "📊 Batch Prediction":
    st.markdown('<h2 class="sub-header">Upload CSV for Batch Predictions</h2>', unsafe_allow_html=True)
    
    st.info("📝 Upload a CSV file containing the environmental parameters. The file should have the same columns as the training data (excluding 'date' and 'Appliances').")
    
    # Show sample format
    with st.expander("📋 View Required CSV Format"):
        sample_df = pd.DataFrame({
            'lights': [30], 'T1': [19.89], 'RH_1': [47.6], 'T2': [19.2], 'RH_2': [44.8],
            'T3': [19.79], 'RH_3': [44.7], 'T4': [19.0], 'RH_4': [45.6], 'T5': [17.2],
            'RH_5': [55.2], 'T6': [18.9], 'RH_6': [50.5], 'T7': [17.2], 'RH_7': [38.5],
            'T8': [17.2], 'RH_8': [40.3], 'T9': [17.0], 'RH_9': [45.5], 'T_out': [6.6],
            'Press_mm_hg': [733.5], 'RH_out': [92.0], 'Windspeed': [7.0], 'Visibility': [63.0],
            'Tdewpoint': [5.3], 'rv1': [13.3], 'rv2': [13.3]
        })
        st.dataframe(sample_df)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None and model_loaded:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
            
            # Display the uploaded data
            with st.expander("👀 Preview Uploaded Data"):
                st.dataframe(df.head(10))
            
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                # Ensure all required columns are present
                missing_cols = set(feature_names) - set(df.columns)
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    # Select only the required features in the correct order
                    X = df[feature_names]
                    
                    # Scale the data
                    X_scaled = scaler.transform(X)
                    
                    # Make predictions
                    predictions = model.predict(X_scaled)
                    
                    # Add predictions to dataframe
                    df['Predicted_Appliances'] = predictions
                    df['Total_Energy'] = df['Predicted_Appliances'] + df['lights']
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        st.metric("Average Predicted Energy", f"{predictions.mean():.2f} Wh")
                    with col3:
                        st.metric("Max Predicted Energy", f"{predictions.max():.2f} Wh")
                    
                    # Show results table
                    st.dataframe(df[['lights', 'Predicted_Appliances', 'Total_Energy']].head(20))
                    
                    # Download button for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Visualization
                    st.markdown("### 📈 Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig = px.histogram(df, x='Predicted_Appliances', 
                                         title='Distribution of Predicted Energy',
                                         labels={'Predicted_Appliances': 'Energy (Wh)'},
                                         color_discrete_sequence=['#1f77b4'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Time series if there's enough data
                        if len(df) > 1:
                            fig = px.line(df.head(100), y='Predicted_Appliances',
                                        title='Energy Consumption Over Time',
                                        labels={'index': 'Time Index', 'Predicted_Appliances': 'Energy (Wh)'},
                                        color_discrete_sequence=['#ff7f0e'])
                            st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Page 3: Model Info
elif page == "📈 Model Info":
    st.markdown('<h2 class="sub-header">Model Performance & Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Model Details")
        st.info("""
        **Model Type:** Extra Trees Regressor  
        **Training Algorithm:** Ensemble Learning  
        **Hyperparameter Tuning:** GridSearchCV  
        **Features:** 27 environmental parameters  
        **Target:** Appliances Energy Consumption (Wh)
        """)
        
        st.markdown("### 📊 Model Performance")
        performance_data = {
            'Metric': ['R² Score (Test)', 'MSE (Test)', 'Training R²', 'Training MSE'],
            'Value': ['0.736', '0.0123', '1.000', '~0.000']
        }
        st.table(pd.DataFrame(performance_data))
        
    with col2:
        st.markdown("### 🎯 Hyperparameters")
        hyperparams = {
            'Parameter': ['n_estimators', 'max_depth', 'max_features', 'criterion', 'bootstrap'],
            'Value': ['1400', '70', 'log2', 'squared_error', 'False']
        }
        st.table(pd.DataFrame(hyperparams))
        
        st.markdown("### 📝 Feature Categories")
        st.success("""
        **Indoor Climate (18 features):**  
        - Temperature sensors: T1-T9  
        - Humidity sensors: RH_1-RH_9
        
        **Outdoor Weather (6 features):**  
        - Temperature, Pressure, Humidity  
        - Wind Speed, Visibility, Dew Point
        
        **Other (3 features):**  
        - Lights energy consumption  
        - Random variables (rv1, rv2)
        """)
    
    # Model comparison chart
    st.markdown("### 📊 Model Comparison")
    comparison_data = {
        'Model': ['Linear Regression', 'Ridge', 'Lasso', 'Decision Tree', 'Extra Trees'],
        'Test R² Score': [-0.0001, 0.162, 0.162, 0.577, 0.736],
        'Test MSE': [0.047, 0.039, 0.039, 0.020, 0.012]
    }
    df_comparison = pd.DataFrame(comparison_data)
    
    fig = px.bar(df_comparison, x='Model', y='Test R² Score',
                 title='Model Performance Comparison (Test R² Score)',
                 color='Test R² Score',
                 color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>⚡ Household Energy Prediction App | Built with Streamlit & Scikit-learn</p>
        <p>📧 For questions or feedback, please contact the developer</p>
    </div>
    """, unsafe_allow_html=True)