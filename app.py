import streamlit as st
from PIL import Image
from summa import summarizer
import pytesseract
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags

# Download NLTK resources for sentiment analysis
import nltk
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Add title on the page
st.title("Text summarization from Image or User Input")

# Ask user to choose between image upload or manual input
input_option = st.radio("Choose input option:", ("Upload Image", "Input Text Manually"))

# Initialize variables
extracted_text = ""

if input_option == "Upload Image":
    # Ask user for uploaded image
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Check if an image is uploaded
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

        # Perform OCR using Tesseract on the uploaded image
        image_data = Image.open(uploaded_image)
        extracted_text = pytesseract.image_to_string(image_data)

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.write(extracted_text)

elif input_option == "Input Text Manually":
    # Ask user to input text manually
    extracted_text = st.text_area("Enter Text:")

    # Display the entered text
    st.subheader("Entered Text:")
    st.write(extracted_text)

# Perform sentiment analysis
sid = SentimentIntensityAnalyzer()
sentiment_scores = sid.polarity_scores(extracted_text)

# Convert raw scores to percentages
pos_percentage = sentiment_scores['pos'] * 100
neu_percentage = sentiment_scores['neu'] * 100
neg_percentage = sentiment_scores['neg'] * 100

# Display sentiment analysis results in percentage
st.subheader("Sentiments:")
st.write(f"Positive: {pos_percentage:.2f}%")
st.write(f"Neutral: {neu_percentage:.2f}%")
st.write(f"Negative: {neg_percentage:.2f}%")

# Perform text summarization
summarized_text = summarizer.summarize(extracted_text, ratio=0.2, language="english")

# Display the summarized text
st.subheader("Summarized Text:")
st.write(summarized_text)

# Perform chunking
tokens = pos_tag(extracted_text.split())
chunk_tree = ne_chunk(tokens)
chunked_data = tree2conlltags(chunk_tree)

# Display the chunked data
st.subheader("Chunked Data:")
st.write(chunked_data)
