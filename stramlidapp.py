import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from utils import read_file, get_table_data

# from mcqgenerator.utils import read_file, get_table_data
from mcqgenerator import generate_evaluate_chain
# from mcqgenerator import logging

# Load environment variables
load_dotenv()

# Load response template
with open("Response.json", "r") as f:
    response_json = json.load(f)

# Streamlit UI
st.title("MCQ Creator Application with LangChain")

with st.form("user_input"):
    uploaded_file = st.file_uploader(
        "Upload PDF or Text File",
        type=["pdf", "txt"]
    )

    mcq_count = st.number_input(
        "Number of MCQs",
        min_value=1,
        max_value=20,
        value=5
    )

    subject = st.text_input(
        "Subject",
        placeholder="Python"
    )

    tone = st.text_input(
        "Complexity of Questions",
        max_chars=20,
        placeholder="Simple"
    )

    button = st.form_submit_button("Generate MCQs")

if button:
    if uploaded_file is None:
        st.error("Please upload a file.")
    elif not subject:
        st.error("Please enter a subject.")
    elif not tone:
        st.error("Please enter the complexity level.")
    else:
        with st.spinner("Generating MCQs and Evaluating them..."):
            try:
                text = read_file(uploaded_file)

                quiz, review = generate_evaluate_chain(
                    text=text,
                    number=mcq_count,
                    subject=subject,
                    tone=tone
                )

                st.success("MCQs Generated Successfully!")

                quiz_table_data = get_table_data(quiz)

                df = pd.DataFrame(quiz_table_data)

                st.subheader("Generated MCQs")
                st.table(df)

                st.subheader("Review")
                st.write(review)

            except Exception as e:
                logging.exception("Error while generating MCQs")
                st.error(f"Error: {str(e)}")
