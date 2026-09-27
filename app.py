import joblib
import pandas as pd
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Predictor")
st.write("Enter customer details below to predict whether they are likely to churn.")

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("customer_churn_random_forest.pkl")

model = load_model()

# ---------- Input form ----------
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
        usage_frequency = st.number_input("Usage Frequency (per month)", min_value=0, max_value=100, value=15)
        support_calls = st.number_input("Support Calls", min_value=0, max_value=50, value=2)
        payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=60, value=0)

    with col2:
        total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=100000.0, value=500.0, step=10.0)
        last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365, value=10)
        gender = st.selectbox("Gender", ["Female", "Male"])
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Premium", "Standard"])

    submitted = st.form_submit_button("Predict Churn")

# ---------- Prediction ----------
if submitted:
    expected_cols = list(model.feature_names_in_)

    # Base numeric fields, always present
    row = {
        "Age": age,
        "Tenure": tenure,
        "Usage Frequency": usage_frequency,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Total Spend": total_spend,
        "Last Interaction": last_interaction,
    }

    # Handle Gender: either a raw "Gender" column, or one-hot "Gender_Male"
    if "Gender" in expected_cols:
        row["Gender"] = gender
    if "Gender_Male" in expected_cols:
        row["Gender_Male"] = 1 if gender == "Male" else 0

    # Handle Subscription Type: either a raw column, or one-hot dummies
    if "Subscription Type" in expected_cols:
        row["Subscription Type"] = subscription_type
    if "Subscription Type_Premium" in expected_cols:
        row["Subscription Type_Premium"] = 1 if subscription_type == "Premium" else 0
    if "Subscription Type_Standard" in expected_cols:
        row["Subscription Type_Standard"] = 1 if subscription_type == "Standard" else 0

    input_df = pd.DataFrame([row])

    # Safety check: make sure every column the model needs is present
    missing = [c for c in expected_cols if c not in input_df.columns]
    if missing:
        st.error(
            f"The app doesn't know how to build these columns the model expects: {missing}. "
            "Please share model.feature_names_in_ so the form can be updated."
        )
        st.stop()

    # Ensure column order matches the model exactly
    input_df = input_df[expected_cols]

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **churn** (probability: {proba[1]:.1%})")
    else:
        st.success(f"✅ This customer is likely to **stay** (probability: {proba[0]:.1%})")

    st.progress(float(proba[1]))
    st.caption(f"Churn probability: {proba[1]:.1%} | Retention probability: {proba[0]:.1%}")

    with st.expander("See input data sent to the model"):
        st.dataframe(input_df)