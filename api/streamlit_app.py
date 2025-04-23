import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("model/svm_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

# Define emotion mapping
emotion_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Streamlit App UI
st.title("Emotion Classifier")

# Text input for user
user_input = st.text_area("Enter a sentence to predict emotion:")

if st.button("Predict Emotion"):
    if user_input:
        # Vectorize the input text
        vec = vectorizer.transform([user_input])
        
        # Get prediction (numeric)
        pred = model.predict(vec)
        
        # Map numeric prediction to emotion label
        predicted_emotion = emotion_map[pred[0]]
        
        # Display the result
        st.write(f"The predicted emotion is: **{predicted_emotion.capitalize()}**")
    else:
        st.write("Please enter some text to classify the emotion.")

