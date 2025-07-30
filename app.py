import string
import nltk
from nltk.corpus import stopwords  # Fixed import
import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer

# Download required NLTK data (add this for first-time setup)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

# Load models with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("Enter the message")


def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # Clone the list
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Only process if there's input
if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Model predict (fixed typo)
    result = model.predict(vector_input)[0]  # Get the first element

    # 4. Display
    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")

    # Optional: Show the processed text for debugging
    with st.expander("Show processed text"):
        st.write(f"Original: {input_sms}")
        st.write(f"Processed: {transformed_sms}")