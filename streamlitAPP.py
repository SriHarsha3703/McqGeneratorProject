import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqGenrator.logger import logging
from src.mcqGenrator.utils import read_file, get_table_data
import streamlit as st
# Updated import based on deprecation warning
from langchain_community.callbacks.manager import get_openai_callback
from src.mcqGenrator.MCQGenerator import generate_evaluate_chain

# Loading JSON file
with open('D:\GenAI\McqGenerator\Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Creating a title for the app
st.title("MCQ Creator using LangChain")

# Creating the form in the Web
with st.form("Inputs"):
    # For file upload
    uploaded_file = st.file_uploader("Upload a .pdf or .txt file")

    # Input field for the number of questions
    counter = st.number_input("No. of MCQ's", min_value=3, max_value=20)

    # Subject field
    subject = st.text_input("Insert Subject", max_chars=20)

    # Quiz difficulty
    tone = st.text_input("Complexity Level of Questions", max_chars=20, placeholder="Simple")

    # Submit button
    button = st.form_submit_button("Create MCQ's")

    # Validation
    if button and uploaded_file is not None and counter and subject and tone:
        with st.spinner("Wait for a few moments, please.."):
            try:
                text = read_file(uploaded_file)

                # Using `get_openai_callback`
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain({
                        "text": text,
                        "number": counter,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    })
                print(response)
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")
            else:
                if isinstance(response, dict):
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)

                            # Displaying the review
                            st.text_area(label="Review", value=response['review'])
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)
