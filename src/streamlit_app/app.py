import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide",
)


st.title("🏠 Real Estate Price Prediction")

st.write(
    "Predict the estimated price of a property "
    "using a trained Random Forest model."
)

st.divider()

st.subheader("Property information")


with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        bed = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=30,
            value=3
        )

        house_size = st.number_input(
            "House size (sq ft)",
            min_value=300.0,
            max_value=50_000.0,
            value=1800.0,
            step=100.0
        )

        state = st.text_input(
            "State",
            value="California"
        )

        zip_code = st.text_input(
            "ZIP code",
            value="90001"
        )

    with col2:
        bath = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=30,
            value=2
        )

        acre_lot = st.number_input(
            "Lot size (acres)",
            min_value=0.0,
            max_value=1000.0,
            value=0.25,
            step=0.05
        )

        city = st.text_input(
            "City",
            value="Los Angeles"
        )

    submitted = st.form_submit_button(
        "Predict price",
        type="primary"
    )


if submitted:

    property_data = {
        "bed": bed,
        "bath": bath,
        "house_size": house_size,
        "acre_lot": acre_lot,
        "state": state.strip(),
        "city": city.strip(),
        "zip_code": zip_code.strip(),
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=property_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            predicted_price = result["predicted_price"]

            st.success("Prediction completed successfully.")

            st.metric(
                label="Estimated property price",
                value=f"${predicted_price:,.2f}"
            )

        else:
            st.error(
                f"API error ({response.status_code}): "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:
        st.error("The prediction request timed out.")

    except Exception as error:
        st.error(f"Unexpected error: {error}")