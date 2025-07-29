import streamlit as st
import requests
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- FastAPI Backend URL ---
# For local testing, this is typically http://127.0.0.1:8000/predict/
# IMPORTANT: If using Docker Compose, change '127.0.0.1' to the service name (e.g., 'api')
API_URL = "http://api:8000/predict/"

# --- UI Styling and Helper Dictionaries ---
# Emojis and colors for each sentiment
SENTIMENT_EMOJIS = {
    "Positive": "😊",
    "Negative": "😠",
    "Neutral": "😐",
    "Irrelevant": "🤷"
}

SENTIMENT_COLORS = {
    "Positive": "#28a745", # Green
    "Negative": "#dc3545", # Red
    "Neutral": "#6c757d",  # Gray
    "Irrelevant": "#ffc107" # Yellow
}

# --- UI Design ---
st.title("🐦 Twitter Sentiment Analysis")
st.write(
    "Enter a tweet or any text below to analyze its sentiment. "
    "The app will classify the text as Positive, Negative, Neutral, or Irrelevant."
)

# --- Text Input Area ---
user_text = st.text_area(
    "Enter text for analysis:",
    "I love using this new product, it's absolutely fantastic!",
    height=150
)

# --- Prediction Logic ---
if st.button('Analyze Sentiment', key='predict_button', use_container_width=True):
    if user_text:
        with st.spinner('Sending text to the model for analysis...'):
            try:
                # Prepare the request payload
                payload = {"text": user_text}

                # Send the POST request to the FastAPI backend
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes

                result = response.json()
                sentiment = result.get('sentiment')
                confidence = result.get('confidence')
                all_scores = result.get('all_scores', {})

                st.subheader("🧠 Analysis Result")

                # Get the emoji and color for the predicted sentiment
                emoji = SENTIMENT_EMOJIS.get(sentiment, "❓")
                color = SENTIMENT_COLORS.get(sentiment, "#000000")

                # Display the main result in a styled box
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid {color};
                        border-radius: 10px;
                        padding: 20px;
                        text-align: center;
                        background-color: #f8f9fa;
                    ">
                        <span style="font-size: 50px;">{emoji}</span>
                        <h2 style="color:{color};">{sentiment.upper()}</h2>
                        <p style="font-size: 18px;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Display all scores in an expander with a bar chart
                with st.expander("See detailed scores for all categories"):
                    if all_scores:
                        # Convert scores to a pandas DataFrame for easy plotting
                        df_scores = pd.DataFrame(
                            list(all_scores.items()),
                            columns=['Sentiment', 'Confidence']
                        ).sort_values('Confidence', ascending=False)

                        st.bar_chart(df_scores.set_index('Sentiment'))
                    else:
                        st.write("No detailed scores were returned.")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Please ensure the backend is running. Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter some text to analyze.")
