import streamlit as st
from transformers import pipeline

# Load the question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define the context about AI
context = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines that are designed to think and act like humans.
It includes various subfields such as machine learning, deep learning, and natural language processing.
Machine learning allows machines to learn from data without being explicitly programmed, while deep learning uses neural networks for data analysis.
Natural Language Processing (NLP) helps machines understand and generate human language. Applications of AI include robotics, autonomous systems, computer vision, and more.
AI can be categorized into narrow AI, which is focused on specific tasks, and general AI, which has a broader understanding and application of intelligence.
"""

# Streamlit app
st.title("AI Chatbot")

# Get user input
user_input = st.text_input("Ask me anything about Artificial Intelligence:")

if user_input:
    # Use the model to answer the question
    result = qa_pipeline(question=user_input, context=context)
    st.write(f"**Answer:** {result['answer']}")
