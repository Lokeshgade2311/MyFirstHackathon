import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import plotly.express as px

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff1c1c;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white;
    }
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data  # Cache data to improve performance
def load_data():
    df = pd.read_csv('F:/Grocery_Detailed_Sales_Data_processed.csv')
    return df

df = load_data()

# Title and Introduction
st.title("🚀 AI enabled demand forecasting")
st.write("""
Welcome to our *Sales Dashboard*! This project aims to revolutionize how businesses analyze and predict sales trends.
We've combined *data science, **machine learning, and **beautiful visualizations* to create an impactful solution.
""")

# Problem Statement
st.header("📌 Problem Statement")
st.write("""
Businesses often struggle to predict sales trends due to a lack of actionable insights. 
Our solution provides a *real-time, interactive dashboard* to help businesses make data-driven decisions.
""")

# Solution Overview
st.header("💡 Our Solution")
st.write("""
We built a *machine learning model* to predict sales trends and integrated it into an interactive dashboard. 
Here's what our solution offers:
- *Real-time sales predictions*
- *Interactive visualizations*
- *User-friendly interface*
""")

# Feature Engineering: Convert categorical variables into numerical
categorical_features = ['Category', 'Weather', 'Local Event']
existing_categorical_features = [col for col in categorical_features if col in df.columns]

if existing_categorical_features:
    df = pd.get_dummies(df, columns=existing_categorical_features, drop_first=True)
else:
    st.warning("Warning: No categorical columns found for encoding.")

# Ensure 'Date' column exists for time-series processing
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
else:
    st.error("Date column is required for the model.")
    st.stop()

# Define features (X) and target variable (y)
columns_to_drop = [col for col in ['Sales Quantity', 'Date', 'Product'] if col in df.columns]
X = df.drop(columns=columns_to_drop, errors='ignore')
y = df['Sales Quantity'] if 'Sales Quantity' in df.columns else None

if y is None:
    st.error("Target variable 'Sales Quantity' not found in dataset.")
    st.stop()

# Normalize data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False)

# Build Random Forest Model
st.write("Training the Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.ravel())

# Predict on test data
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_actual = scaler_y.inverse_transform(y_test)

# Evaluate model performance
mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)

st.write("Data successfully split into training and testing sets.")
st.write(f"Model Performance:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")

# Plot actual vs predicted values
st.write("### Actual vs Predicted Sales Quantity")
fig1 = px.scatter(x=y_test_actual.flatten(), y=y_pred.flatten(), labels={'x': 'Actual Sales', 'y': 'Predicted Sales'}, 
                  title="Actual vs Predicted Sales Quantity")
st.plotly_chart(fig1)

# Plot sales trends over time
st.write("### Actual vs Predicted Sales Over Time")
fig2 = px.line(df[-len(y_test_actual):], x='Date', y=[y_test_actual.flatten(), y_pred.flatten()], 
               labels={'value': 'Sales Quantity', 'variable': 'Legend'}, 
               title="Actual vs Predicted Sales Over Time")
st.plotly_chart(fig2)

# User prediction function
def predict_sales(user_input):
    user_df = pd.DataFrame([user_input])
    
    # Convert categorical variables
    user_df = pd.get_dummies(user_df, columns=existing_categorical_features, drop_first=True)
    
    # Align with training columns
    user_df = user_df.reindex(columns=X.columns, fill_value=0)
    
    # Scale the input
    user_scaled = scaler_X.transform(user_df)
    
    # Make prediction
    prediction_scaled = model.predict(user_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    
    return prediction[0][0]

# Streamlit UI for user input
st.write("### Predict Sales Quantity")
st.write("Enter the following details to predict sales:")

# Example user input fields
stock_level = st.number_input("Stock Level", min_value=0, value=50)
weather = st.selectbox("Weather", options=['Sunny', 'Rainy', 'Cloudy'])
local_event = st.selectbox("Local Event", options=['None', 'Festival', 'Sports Event'])
category = st.selectbox("Category", options=['Beverages', 'Snacks', 'Dairy'])

if st.button("Predict Sales"):
    user_input = {
        'Stock Level': stock_level,
        'Weather': weather,
        'Local Event': local_event,
        'Category': category
    }
    predicted_sales = predict_sales(user_input)
    st.success(f"Predicted Sales Quantity: {predicted_sales:.2f}")

# Impact Section
st.header("🌟 Impact")
st.write("""
Our solution has the potential to:
- *Increase sales by 20%*
- *Reduce decision-making time by 50%*
- *Improve customer satisfaction*
""")



# Call to Action
st.header("🚀 Get Started")
st.write("Ready to transform your business? Contact us today!")
if st.button("Contact Us"):
    st.write("📧 Email: team@hackathonproject.com")

# Footer
st.markdown("---")
st.write("© 2025 Hackathon Project. All rights reserved.")