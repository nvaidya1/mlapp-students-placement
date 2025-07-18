import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_PATH = "model.joblib"
ENCODER_PATH = "encoder.joblib"

# ------------------------- Utils -------------------------
@st.cache_data
def generate_sample_data(n_samples=100):
    np.random.seed(42)
    courses = ['Python', 'Web Dev', 'GIS', 'Data Viz']
    data = {
        'Student Name': [f'Student{i+1}' for i in range(n_samples)],
        'Course Name': np.random.choice(courses, n_samples),
        'Education (years)': np.random.randint(10, 18, size=n_samples),
        'Duration of Learning (months)': np.random.randint(3, 12, size=n_samples),
        'Duration of Internship (months)': np.random.randint(0, 12, size=n_samples),
        'Job Search (months)': np.random.randint(1, 12, size=n_samples),
        'Job Salary (k)': np.random.randint(30, 120, size=n_samples),
        'Job Retention (months)': np.random.randint(6, 36, size=n_samples),
    }
    return pd.DataFrame(data)

def train_model(df):
    encoder = OneHotEncoder(sparse_output=False)
    course_encoded = encoder.fit_transform(df[['Course Name']])
    course_df = pd.DataFrame(course_encoded, columns=encoder.get_feature_names_out(['Course Name']))
    df_encoded = pd.concat([course_df, df.drop(['Student Name', 'Course Name'], axis=1)], axis=1)

    X = df_encoded.drop(['Job Search (months)', 'Job Salary (k)', 'Job Retention (months)'], axis=1)
    y = df_encoded[['Job Search (months)', 'Job Salary (k)', 'Job Retention (months)']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model.fit(X_train, y_train)

    # Save model & encoder
    joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name}_model.joblib"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, f"{model_name}_encoder.joblib"))

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

    return model, encoder, mae, rmse

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return model, encoder
    return None, None

def reset_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(ENCODER_PATH):
        os.remove(ENCODER_PATH)

# ------------------------- Streamlit App -------------------------
# Create models directory
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

st.title("AI Prediction App for Training Outcomes")

st.sidebar.header("Actions")
option = st.sidebar.radio("Choose Action:", ["Train Model", "Predict Outcome", "Reset Model"])

# ------- Train -------
if option == "Train Model":
    st.header("Train the Model")

    uploaded_file = st.file_uploader("Upload CSV file or use sample data", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = generate_sample_data()
        st.info("Using generated sample data")

    st.write("Training Data Preview:", df.head())

    if st.button("Train Model"):
        model, encoder, mae, rmse = train_model(df)
        st.success("Model trained and saved!")
        st.write(f"MAE: {mae}")
        st.write(f"RMSE: {rmse}")

# ------- Predict -------
elif option == "Predict Outcome":
    st.header("Predict Job Outcome from Learning KPIs")

    model, encoder = load_model()
    if not model:
        st.error("Model not found. Please train the model first.")
    else:
        course_name = st.selectbox("Course Name", ['Python', 'Web Dev', 'GIS', 'Data Viz'])
        education = st.slider("Education (years)", 10, 20, 12)
        duration_learning = st.slider("Duration of Learning (months)", 1, 24, 10)
        duration_intern = st.slider("Duration of Internship (months)", 0, 24, 12)

        if st.button("Predict"):
            input_df = pd.DataFrame({
                'Course Name': [course_name],
                'Education (years)': [education],
                'Duration of Learning (months)': [duration_learning],
                'Duration of Internship (months)': [duration_intern],
            })

            course_encoded = encoder.transform(input_df[['Course Name']])
            course_df = pd.DataFrame(course_encoded, columns=encoder.get_feature_names_out(['Course Name']))
            final_input = pd.concat([course_df, input_df.drop(['Course Name'], axis=1)], axis=1)

            prediction = model.predict(final_input)
            job_search, salary, retention = prediction[0]

            st.success("Prediction:")
            st.write(f"Estimated Job Search Time: **{job_search:.1f} months**")
            st.write(f"Estimated Job Salary: **${salary:.1f}k**")
            st.write(f"Estimated Retention: **{retention:.1f} months**")

# ------- Reset -------
elif option == "Reset Model":
    st.header("Reset Model and Remove Saved State")
    if st.button("Delete model and encoder"):
        reset_model()
        st.success("Model and encoder removed. You can now re-train with fresh data.")

