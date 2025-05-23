import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Internshipprojects\datasets\Global_EV_Charging_Behavior_2024.csv")
    df.columns = df.columns.str.strip()
    return df

# ---------- EDA Visualizations ----------
def generate_eda_plots(df):
    plots = []

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    df['Charging Station Type'].value_counts().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'], ax=ax1)
    ax1.set_title("Number of Sessions per Charging Type")
    ax1.set_xlabel("Charging Type")
    ax1.set_ylabel("Count")
    plots.append(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    df.groupby('Charging Station Type')['Charging Duration (mins)'].mean().sort_values().plot(
        kind='bar', color='orange', edgecolor='black', ax=ax2)
    ax2.set_title("Average Charging Time by Station Type")
    ax2.set_xlabel("Charging Type")
    ax2.set_ylabel("Minutes")
    plots.append(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Charging Duration (mins)', y='Energy Delivered (kWh)', 
                    hue='Charging Station Type', palette='viridis', alpha=0.7, ax=ax3)
    ax3.set_title("Energy Delivered vs Charging Time")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plots.append(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Charging Station Type', y='Charging Cost ($)', palette='Set2', ax=ax4)
    ax4.set_title("Charging Cost Distribution by Type")
    ax4.set_xlabel("Charging Type")
    ax4.set_ylabel("Cost ($)")
    plots.append(fig4)

    st.write("### Pairplot of Numerical Features")
    pairplot = sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
    plots.append(pairplot.fig)

    return plots

# ---------- Train Model ----------
@st.cache_resource
def train_model(df):
    features = ['Charging Duration (mins)', 'Charging Cost ($)', 'Temperature (°C)', 'Battery Capacity (kWh)',
                'Charging Station Type', 'EV Model', 'Country']
    target = 'Energy Delivered (kWh)'

    df_model = df[features + [target]].dropna()
    numerical = ['Charging Duration (mins)', 'Charging Cost ($)', 'Temperature (°C)', 'Battery Capacity (kWh)']
    categorical = ['Charging Station Type', 'EV Model', 'Country']

    X = df_model[numerical + categorical]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('poly', PolynomialFeatures(degree=2, include_bias=False), numerical)
    ])

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return pipeline, df_model, numerical, categorical, rmse, r2

# ---------- Main App ----------
def main():
    st.set_page_config(page_title="EV Energy Prediction Dashboard", layout="wide")
    st.title("🔋 EV Charging Dashboard")

    df = load_data()
    pipeline, df_model, numerical, categorical, rmse, r2 = train_model(df)

    st.sidebar.header("Navigation")
    option = st.sidebar.radio("Go to", ["EDA Analysis", "Energy Prediction"])

    if option == "EDA Analysis":
        st.header("📊 Exploratory Data Analysis")
        with st.spinner("Generating visualizations..."):
            plots = generate_eda_plots(df)
            for plot in plots:
                st.pyplot(plot)
                st.markdown("---")
        st.success("EDA complete!")

        st.markdown("### 🔍 Model Evaluation on Test Set")
        st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

    elif option == "Energy Prediction":
        st.header("⚡ Energy Delivered Prediction")

        user_input = {}
        for col in numerical:
            min_val = float(df_model[col].min())
            max_val = float(df_model[col].max())
            mean_val = float(df_model[col].mean())
            user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

        for col in categorical:
            options = sorted(df_model[col].dropna().unique())
            user_input[col] = st.selectbox(f"Select {col}", options)

        if st.button("Predict Energy Delivered"):
            input_df = pd.DataFrame([user_input])
            prediction = pipeline.predict(input_df)[0]
            st.subheader("🔮 Prediction Result")
            st.write(f"**Estimated Energy Delivered (kWh):** {prediction:.2f}")
            st.markdown("---")
            st.markdown(f"**Model RMSE:** {rmse:.2f} &nbsp;&nbsp;&nbsp; **R² Score:** {r2:.2f}")

if __name__ == "__main__":
    main()
