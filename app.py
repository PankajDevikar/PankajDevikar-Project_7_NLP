import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load("model_relu_pickle.pkl")  # This should load the actual trained model, not a dictionary
optimizer = joblib.load("optimizer_config.pkl")  # Load optimizer config if needed

# Streamlit app layout
st.title("Disaster Tweet Classifier")
st.write("This app classifies tweets as related to disasters or not.")

# Input form
tweet_input = st.text_area("Enter a tweet to classify", "")

if tweet_input:
    # Pre-process the input tweet (e.g., clean and tokenize if needed)
    cleaned_tweet = tweet_input.lower()  # Example: convert to lowercase
    # Add more pre-processing steps as needed (e.g., punctuation removal, stopword removal)

    # Ensure that model is a valid classifier and can predict
    prediction = model.predict([cleaned_tweet])

    # Show the result
    if prediction == 1:
        st.write("This tweet is about a disaster!")
    else:
        st.write("This tweet is not about a disaster.")

# Instructions
st.write("### How this works:")
st.write(
    "1. Type or paste a tweet into the box above."
    "\n2. Click Enter to classify the tweet as disaster-related or not."
    "\n3. The result will be shown below the input box."
)
