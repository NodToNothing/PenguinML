import streamlit as st
from transformers import pipeline #had to pip install transformers directly

st.title("Hugging Face Demo")
text = st.text_input("Enter text to analyze")

@st.cache_resource #so we don't download the model every time
#model = pipeline("sentiment-analysis") #should have a model name and revision

def get_model():
    return pipeline("sentiment-analysis")
model = get_model()


if text:
    result = model(text)
    st.write("Sentiment: ", result[0]["label"])
    st.write("Confidence: ", result[0]["score"])
    st.write(result[0])