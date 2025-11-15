import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_trained_model():
    return load_model("simple_rnn.keras")

model = load_trained_model()

## Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in encoded_review])

## Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    # Add check for out-of-vocabulary words
    encoded_review = []
    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2)  # Unknown word token
    
    # Ensure we have valid data
    if len(encoded_review) == 0:
        encoded_review = [2]  # At least one token

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Prediction function
def predict_sentiment(review):
    try:
        preprocess_input = preprocess_text(review)
        pred = model.predict(preprocess_input, verbose=0)  # verbose=0 to suppress output
        
        # Check for NaN
        if np.isnan(pred[0][0]):
            return None, None, "Error: Model output is NaN"
        
        sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'
        return sentiment, float(pred[0][0]), None
    except Exception as e:
        return None, None, f"Error: {str(e)}"

## Streamlit app
st.title(" IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie review to classify it as positive or negative.')

## User input
user_input = st.text_area('Movie Review', placeholder='Type your movie review here...', height=150)

if st.button('Classify', type='primary'):
    if user_input.strip():  # Check if input is not empty
        with st.spinner('Analyzing sentiment...'):
            # Make prediction using the function
            sentiment, score, error = predict_sentiment(user_input)
            
            if error:
                st.error(error)
            else:
                # Display the result with better formatting
                st.success(f'**Sentiment:** {sentiment}')
                st.metric(label="Prediction Score", value=f"{score:.4f}")
                
                # Add visual indicator
                if sentiment == 'Positive':
                    st.progress(score)
                    st.balloons()
                else:
                    st.progress(1 - score)
                
                # Show confidence level
                confidence = max(score, 1 - score) * 100
                st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning('Please enter a movie review before clicking Classify.')
else:
    st.info(' Enter a movie review above and click "Classify" to get started.')

# Optional: Add example reviews
with st.expander(" Try these example reviews"):
    if st.button("Example 1: Positive Review"):
        st.session_state.example = "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. Highly recommend!"
    
    if st.button("Example 2: Negative Review"):
        st.session_state.example = "Terrible movie. Complete waste of time. Poor acting and a boring storyline that went nowhere."
    
    if st.button("Example 3: Mixed Review"):
        st.session_state.example = "The movie had some good moments but overall it was just okay. Nothing special but not terrible either."

# Auto-fill if example is selected
if 'example' in st.session_state:
    user_input = st.session_state.example
    del st.session_state.example
    st.rerun()