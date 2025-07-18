import os
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Directory to store multiple models
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Generate synthetic sample data
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

# Train and save model
def train_model(df, model_name):
    encoder = OneHotEncoder(sparse_output=False)
    course_encoded = encoder.fit_transform(df[['Course Name']])
    course_df = pd.DataFrame(course_encoded, columns=encoder.get_feature_names_out(['Course Name']))
    df_encoded = pd.concat([course_df, df.drop(['Student Name', 'Course Name'], axis=1)], axis=1)

    X = df_encoded.drop(['Job Search (months)', 'Job Salary (k)', 'Job Retention (months)'], axis=1)
    y = df_encoded[['Job Search (months)', 'Job Salary (k)', 'Job Retention (months)']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model.fit(X_train, y_train)

    # Save model and encoder
    joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name}_model.joblib"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, f"{model_name}_encoder.joblib"))

    # Save metadata (number of records)
    meta = {"num_records": len(df)}
    with open(os.path.join(MODELS_DIR, f"{model_name}_meta.json"), "w") as f:
        json.dump(meta, f)


    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

    return model, encoder, mae, rmse

# UI layout
st.title("AI Prediction App for Training Outcomes")

st.sidebar.header("Actions")
option = st.sidebar.radio("Choose Action:", ["Train Model", "Predict Outcome", "Reset Model"])

# ---------------- Train Model ----------------
if option == "Train Model":
    st.header("Train a Model")

    uploaded_file = st.file_uploader("Upload CSV file or use sample data", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = generate_sample_data()
        st.info("Using generated sample data.")

    st.write("Training Data Preview:", df.head())

    st.subheader("Model Name")
    existing_models = [f.split("_model.joblib")[0] for f in os.listdir(MODELS_DIR) if f.endswith("_model.joblib")]
    selected_model = st.selectbox("Select an existing model or type a new name", ["(new model)"] + existing_models)

    if selected_model == "(new model)":
        new_model_name = st.text_input("Enter new model name").strip()
        model_name = new_model_name
    else:
        model_name = selected_model

    if model_name:
        if st.button("Train Model"):
            model, encoder, mae, rmse = train_model(df, model_name)
            st.success(f"Model '{model_name}' trained and saved!")
            st.info(f"Trained with {len(df)} records.")
            st.write(f"MAE: {mae}")
            st.write(f"RMSE: {rmse}")
    else:
        st.warning("Please enter or select a model name.")

# ---------------- Predict Outcome ----------------
elif option == "Predict Outcome":
    st.header("Predict Job Outcome from Learning KPIs")

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_model.joblib")]
    model_names = [f.replace("_model.joblib", "") for f in model_files]
    
    if not model_names:
        st.error("No models available. Train a model first.")
    else:
        selected_model_name = st.selectbox("Choose a trained model", model_names)
        model_path = os.path.join(MODELS_DIR, f"{selected_model_name}_model.joblib")
        encoder_path = os.path.join(MODELS_DIR, f"{selected_model_name}_encoder.joblib")

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)

        # Read metadata to show record count
        meta_path = os.path.join(MODELS_DIR, f"{selected_model_name}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            st.info(f"This model was trained on {meta['num_records']} records.")
        else:
            st.warning("No metadata found for this model.")

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

            st.success("Prediction Result:")
            st.write(f"Estimated Job Search: **{job_search:.1f} months**")
            st.write(f"Estimated Salary: **${salary:.1f}k**")
            st.write(f"Estimated Retention: **{retention:.1f} months**")

            # LLM prompt generation
            st.markdown("---")
            st.subheader("LLM Prompt for Reasoning")
            prompt_text = f"""
A student took the "{course_name}" course, has {education} years of education, studied for {duration_learning} months, and completed an internship of {duration_intern} months.
The statistical model predicts:
- Job Search Time: {job_search:.1f} months
- Expected Salary: ${salary:.1f}k
- Job Retention: {retention:.1f} months

As an AI assistant with general knowledge of job market trends, education levels, and career outcomes, analyze whether this prediction seems accurate or if there are adjustments you'd suggest based on context.
Provide reasoning in your answer.
"""
            st.text_area("Copy & Paste this prompt into an LLM:", prompt_text.strip(), height=250)



# ---------------- Reset Model ----------------
elif option == "Reset Model":
    st.header("Reset All Models")

    if st.button("Delete All Saved Models"):
        for f in os.listdir(MODELS_DIR):
            os.remove(os.path.join(MODELS_DIR, f))
        st.success("All models have been deleted.")
