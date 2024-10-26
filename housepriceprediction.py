import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Page Configurations
st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #ff4b4b;
        color: white;
    }
    .stTextInput>div>div>input {
        color: #4F8BF9;
    }
    .stSelectbox>div>div>input {
        color: #4F8BF9;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header with Animation
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #ff4b4b;'>🏠 House Price Prediction App</h1>
        <p style='font-size: 1.2em;'>Predict house prices using advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)

# Load the dataset with progress bar
@st.cache_data
def load_data():
    with st.spinner('Loading dataset...'):
        file_path = "Housing.csv"
        if not os.path.exists(file_path):
            st.error("❌ Dataset not found! Please ensure 'Housing.csv' is in the same directory.")
            return None
        data = pd.read_csv(file_path)
        return data

# Sidebar styling
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #ff4b4b;'>📊 Input Parameters</h2>
    </div>
    """, unsafe_allow_html=True)

data = load_data()

if data is not None:
    # Data Preprocessing
    data['mainroad'] = data['mainroad'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

    # Feature Selection with Interactive Checkbox
    st.sidebar.markdown("### 🎯 Select Features")
    selected_features = []
    if st.sidebar.checkbox('Area', value=True):
        selected_features.append('area')
    if st.sidebar.checkbox('Bedrooms', value=True):
        selected_features.append('bedrooms')
    if st.sidebar.checkbox('Main Road', value=True):
        selected_features.append('mainroad')
    if st.sidebar.checkbox('Stories', value=True):
        selected_features.append('stories')

    if not selected_features:
        st.warning("⚠️ Please select at least one feature for prediction!")
        st.stop()

    # Prepare features and target
    X = data[selected_features]
    y = data['price']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model Training with Progress Bar
    with st.spinner('Training model...'):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # User Input Section
    st.sidebar.markdown("### 🏗️ House Details")
    
    input_values = {}
    if 'area' in selected_features:
        # Using text input for area with validation
        area_input = st.sidebar.text_input(
            'Area (sq ft)', 
            value='1650',
            help='Enter the area in square feet (numbers only)'
        )
        try:
            area_value = float(area_input)
            if area_value < 0:
                st.sidebar.error('❌ Area cannot be negative!')
                st.stop()
            elif area_value < data['area'].min():
                st.sidebar.warning(f'⚠️ Area is below minimum recorded value ({data["area"].min()} sq ft)')
            elif area_value > data['area'].max():
                st.sidebar.warning(f'⚠️ Area is above maximum recorded value ({data["area"].max()} sq ft)')
            input_values['area'] = area_value
        except ValueError:
            st.sidebar.error('❌ Please enter a valid number for area!')
            st.stop()

    if 'bedrooms' in selected_features:
        input_values['bedrooms'] = st.sidebar.number_input('Number of Bedrooms',
            min_value=int(data['bedrooms'].min()),
            max_value=int(data['bedrooms'].max()),
            value=int(data['bedrooms'].median()))

    if 'mainroad' in selected_features:
        mainroad_input = st.sidebar.selectbox('Near Main Road?', 
            options=['Yes', 'No'],
            index=0)
        input_values['mainroad'] = 1 if mainroad_input == 'Yes' else 0

    if 'stories' in selected_features:
        input_values['stories'] = st.sidebar.number_input('Number of Stories',
            min_value=int(data['stories'].min()),
            max_value=int(data['stories'].max()),
            value=int(data['stories'].median()))

    # Create tabs with icons
    tab1, tab2, tab3 = st.tabs(['🔮 Prediction', '📊 Analysis', '📈 Model Insights'])

    with tab1:
        st.markdown("### 🎯 House Price Prediction")
        
        if st.button('Calculate Price', key='predict'):
            with st.spinner('Calculating...'):
                # Prepare input data
                input_data = np.array([[input_values[feature] for feature in selected_features]])
                input_data_scaled = scaler.transform(input_data)
                
                # Make prediction
                predicted_price = model.predict(input_data_scaled)[0]
                
                # Display prediction
                st.balloons()
                st.success(f"### Predicted House Price: ₹{predicted_price:,.2f}")
                
                # Calculate error metrics for the model
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Display error metrics in the prediction tab
                st.markdown("#### Model Performance Metrics")
                st.write(f"- **Mean Absolute Error (MAE):** ₹{mae:,.2f}")
                st.write(f"- **Mean Squared Error (MSE):** ₹{mse:,.2f}")
                st.write(f"- **Root Mean Squared Error (RMSE):** ₹{rmse:,.2f}")
                st.write(f"- **R-squared (R²):** {r2:.2f}")

    with tab2:
        st.markdown("### 📊 Data Analysis")
        
        # Price vs Predicted Price
        st.markdown("### 🏠 Price vs Predicted Price")
        price_comparison_df = pd.DataFrame({
            'Actual Price': y_test,
            'Predicted Price': y_pred
        })
        
        fig3 = px.scatter(price_comparison_df, x='Actual Price', y='Predicted Price',
                         title='Actual Price vs Predicted Price',
                         labels={'Actual Price': 'Actual Price (₹)', 'Predicted Price': 'Predicted Price (₹)'})
        fig3.add_shape(type='line', line=dict(color='red', dash='dash'), 
                        x0=price_comparison_df['Actual Price'].min(), 
                        y0=price_comparison_df['Actual Price'].min(),
                        x1=price_comparison_df['Actual Price'].max(),
                        y1=price_comparison_df['Actual Price'].max())
        
        st.plotly_chart(fig3, use_container_width=True)


        # Correlation heatmap
        corr_matrix = data[selected_features + ['price']].corr()
        fig2 = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale='RdBu_r')
        st.plotly_chart(fig2, use_container_width=True)

        # Interactive Plotly visualizations
        fig1 = px.scatter(data, x='area', y='price', 
                         title='House Prices vs Area',
                         labels={'area': 'Area (sq ft)', 'price': 'Price (₹)'})
        
        # Add simple trend line
        z = np.polyfit(data['area'], data['price'], 1)
        p = np.poly1d(z)
        fig1.add_trace(go.Scatter(x=data['area'], y=p(data['area']),
                                 mode='lines', name='Trend',
                                 line=dict(color='red', dash='dash')))
        
        st.plotly_chart(fig1, use_container_width=True)


        # Display model coefficients
        coefficients_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': model.coef_
        })
        st.markdown("### 🧮 Model Coefficients")
        st.write(coefficients_df)

    with tab3:
        st.markdown("### 📈 Model Performance")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': np.abs(model.coef_)
        }).sort_values(by='Importance', ascending=False)

        fig4 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance', labels={'Importance': 'Coefficient Magnitude'})
        st.plotly_chart(fig4, use_container_width=True)

                # Residual plot
        residuals = y_test - y_pred
        fig4 = px.scatter(x=y_pred, y=residuals,
                         labels={'x': 'Predicted Price', 'y': 'Residuals'},
                         title='Residual Analysis')
        fig4.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig4, use_container_width=True)

# Footer
st.markdown("""
    <footer style='text-align: center; padding: 1rem;'>
        <p>&copy; 2024 House Price Prediction App</p>
    </footer>
    """, unsafe_allow_html=True)
