# ============================================================
# VISIT WITH US
# TOURISM PACKAGE PURCHASE PREDICTION
# STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Visit With Us - Tourism Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

# ============================================================
# 3. APPLICATION TITLE
# ============================================================

st.title("✈️ Visit With Us")

st.subheader(
    "Tourism Package Purchase Prediction"
)

st.markdown(
    """
    This application predicts the probability of a customer
    purchasing a tourism package.

    The prediction helps the sales team identify potential
    customers and prioritize marketing activities.
    """
)

st.divider()


# ============================================================
# 4. LOAD MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# 5. LOAD MODEL METADATA
# ============================================================

@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r") as file:

        metadata = json.load(file)

    return metadata


try:

    model = load_model()

    metadata = load_metadata()

    st.success("Machine Learning model loaded successfully!")

except Exception as error:

    st.error(
        f"Unable to load model or metadata: {error}"
    )

    st.stop()


# ============================================================
# 6. MODEL CONFIGURATION
# ============================================================

FEATURE_COLUMNS = metadata.get(

    "feature_columns",

    [
        "Age",
        "CityTier",
        "Occupation",
        "Gender",
        "NumberOfPersonVisiting",
        "PreferredPropertyStar",
        "MaritalStatus",
        "NumberOfTrips",
        "Passport",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "Designation",
        "MonthlyIncome"
    ]

)

MODEL_NAME = metadata.get(
    "model_name",
    "Random Forest Classifier"
)

TARGET_COLUMN = metadata.get(
    "target_column",
    "ProdTaken"
)

CLASSIFICATION_THRESHOLD = float(
    metadata.get("classification_threshold", 0.50)
)

if not 0 <= CLASSIFICATION_THRESHOLD <= 1:

    st.error("Invalid classification threshold in model metadata.")

    st.stop()


if hasattr(model, "classes_"):

    positive_class_indices = np.where(
        np.asarray(model.classes_) == 1
    )[0]

    if len(positive_class_indices) == 0:

        st.error(
            "The loaded model does not contain positive class 1."
        )

        st.stop()

    POSITIVE_CLASS_INDEX = int(positive_class_indices[0])

else:

    st.error(
        "The loaded model does not expose its class labels."
    )

    st.stop()


# ============================================================
# 7. SIDEBAR INFORMATION
# ============================================================

st.sidebar.title("Project Information")

st.sidebar.info(
    """
    **Project:** Visit With Us

    **Objective:** Tourism Package Purchase Prediction

    **Algorithm:** Random Forest Classifier

    **Model:** Trained Machine Learning Pipeline

    **Deployment:** Streamlit Community Cloud
    """
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Enter customer details to estimate purchase probability."
)


# ============================================================
# 8. CUSTOMER INPUT FORM
# ============================================================

st.header("Customer Information")

st.write(
    "Please enter the customer details below."
)


# Use actual categories from the training data.
# Update these options if your dataset contains different values.

with st.form("customer_prediction_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # CUSTOMER DEMOGRAPHICS
    # --------------------------------------------------------

    with col1:

        st.subheader("Customer Details")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced",
                "Unmarried"
            ]
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Salaried",
                "Small Business",
                "Large Business",
                "Free Lancer"
            ]
        )

    # --------------------------------------------------------
    # TRAVEL INFORMATION
    # --------------------------------------------------------

    with col2:

        st.subheader("Travel Information")

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        number_of_trips = st.number_input(
            "Number of Trips",
            min_value=0,
            max_value=100,
            value=3
        )

        number_of_person_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=20,
            value=2
        )

        number_of_children_visiting = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            max_value=10,
            value=0
        )

        preferred_property_star = st.selectbox(
            "Preferred Property Star",
            [3, 4, 5]
        )

    # --------------------------------------------------------
    # CUSTOMER FINANCIAL AND OTHER INFORMATION
    # --------------------------------------------------------

    with col3:

        st.subheader("Additional Information")

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0,
            max_value=1000000,
            value=25000,
            step=1000
        )

        passport = st.selectbox(
            "Do you have a Passport?",
            ["Yes", "No"]
        )

        own_car = st.selectbox(
            "Do you own a Car?",
            ["Yes", "No"]
        )

        designation = st.selectbox(
            "Designation",
            [
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP"
            ]
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "Predict Tourism Package Purchase",
        use_container_width=True
    )


# ============================================================
# 9. CREATE CUSTOMER DATAFRAME
# ============================================================

if submitted:

    input_data = {

        "Age": age,

        "CityTier": city_tier,

        "Occupation": occupation,

        "Gender": gender,

        "NumberOfPersonVisiting": number_of_person_visiting,

        "PreferredPropertyStar": preferred_property_star,

        "MaritalStatus": marital_status,

        "NumberOfTrips": number_of_trips,

        "Passport": 1 if passport == "Yes" else 0,

        "OwnCar": 1 if own_car == "Yes" else 0,

        "NumberOfChildrenVisiting": number_of_children_visiting,

        "Designation": designation,

        "MonthlyIncome": monthly_income

    }


    # Convert dictionary into DataFrame

    input_df = pd.DataFrame([input_data])


    # Arrange columns in the same order as model training

    missing_features = set(FEATURE_COLUMNS) - set(input_df.columns)

    extra_features = set(input_df.columns) - set(FEATURE_COLUMNS)

    if missing_features or extra_features:

        st.error(
            f"""
            Input feature mismatch.

            Missing features: {missing_features}

            Unexpected features: {extra_features}
            """
        )

        st.stop()


    input_df = input_df[FEATURE_COLUMNS]


    # ========================================================
    # 10. DISPLAY CUSTOMER INPUT
    # ========================================================

    st.divider()

    st.header("Customer Input Summary")

    st.dataframe(
        input_df,
        use_container_width=True
    )


    # ========================================================
    # 11. MODEL PREDICTION
    # ========================================================

    try:

        if not hasattr(model, "predict_proba"):

            st.error(
                "The model does not support probability prediction."
            )

            st.stop()


        probabilities = model.predict_proba(input_df)[0]

        purchase_probability = float(
            probabilities[POSITIVE_CLASS_INDEX]
        )

        prediction = int(
            purchase_probability >= CLASSIFICATION_THRESHOLD
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

        st.stop()


    # ========================================================
    # 12. CUSTOMER SEGMENTATION
    # ========================================================

    if purchase_probability >= 0.70:

        customer_segment = "High Potential"

        recommendation = (
            "Prioritize this customer for sales follow-up. "
            "Offer suitable tourism packages and personalized promotions."
        )

    elif purchase_probability >= 0.40:

        customer_segment = "Medium Potential"

        recommendation = (
            "Nurture this customer through targeted email campaigns, "
            "travel offers and follow-up communication."
        )

    else:

        customer_segment = "Low Potential"

        recommendation = (
            "Use low-cost digital marketing and educational content. "
            "Reassess the customer after further engagement."
        )


    # ========================================================
    # 13. DISPLAY PREDICTION RESULTS
    # ========================================================

    st.divider()

    st.header("Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Purchase Probability",
            f"{purchase_probability * 100:.2f}%"
        )


    with result_col2:

        st.metric(
            "Predicted Purchase",
            "Yes" if prediction == 1 else "No"
        )


    with result_col3:

        st.metric(
            "Customer Segment",
            customer_segment
        )


    st.subheader("Purchase Probability")

    st.progress(purchase_probability)


    # ========================================================
    # 14. BUSINESS RECOMMENDATIONS
    # ========================================================

    st.subheader("Business Recommendation")

    if customer_segment == "High Potential":

        st.success(recommendation)

    elif customer_segment == "Medium Potential":

        st.warning(recommendation)

    else:

        st.info(recommendation)


    # ========================================================
    # 15. PREDICTION HISTORY
    # ========================================================

    if "prediction_history" not in st.session_state:

        st.session_state.prediction_history = []


    history_record = input_data.copy()

    history_record.update({

        "Purchase_Probability": round(
            purchase_probability * 100, 2
        ),

        "Predicted_Purchase": prediction,

        "Customer_Segment": customer_segment,

        "Prediction_Date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    })


    st.session_state.prediction_history.append(
        history_record
    )


# ============================================================
# 16. DISPLAY PREDICTION HISTORY
# ============================================================

if st.session_state.get("prediction_history"):

    st.divider()

    st.header("Prediction History")

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )


    # Download prediction history

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(

        label="Download Prediction History",

        data=csv_data,

        file_name="tourism_prediction_history.csv",

        mime="text/csv"

    )


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style='text-align:center'>

    <h4>Visit With Us - Tourism Package Prediction</h4>

    <p>Powered by Machine Learning and Streamlit</p>

    <p>Automated MLOps Project</p>

    </div>
    """,
    unsafe_allow_html=True
)
