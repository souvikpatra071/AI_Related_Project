from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain, SequentialChain
# from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv
import os
import json
import pandas as pd
import traceback
import PyPDF2
import logging

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
# print(key)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# print(llm)

TEMPLATE = """
Text:{text}
you are an expert MCQ maker. Given the above text,it is your job to create a quiz of {number} multiple choice questions for {subject} students in
    {tone}. Make sure the questions are not repeated and check all questions to be confirmation. Make sure  your format your response like RESPONSE_JSON below and use it as a guide
.Ensure to make {number} MCQs.{response_json}"""


with open("Response.json","r") as f:
    response_json = json.load(f)
print(response_json)

quiz_generation_prompt =PromptTemplate (
    input_variables = ["text","number","subject","tone","response_json"],
    template = TEMPLATE
)

# quiz_chain = LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)
quiz_chain = quiz_generation_prompt | llm

TEMPLATE2 = """
You are an expert English Grammar expert. Given multiple choice quizes for {subject} . You need to evaluate the complexity of the question and give a complete analysis of the question
Only use max 50 words if the quiz is not at per with cognitive and analytical abilities of the students. Update the quiz which need to be
changed and change the toneso that it perfectly fit for student abilities{quiz}"""

quiz_evaluation_prompt =PromptTemplate (
    input_variables = ["subject","quiz"],
    template = TEMPLATE2
)

# review_chain = LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key="review",verbose=True)
review_chain = quiz_evaluation_prompt | llm


def generate_evaluate_chain(text, number, subject, tone):
    
    # Generate quiz
    quiz_result = quiz_chain.invoke({
        "text": text,
        "number": number,
        "subject": subject,
        "tone": tone,
        "response_json": json.dumps(response_json)
    })

    # Evaluate quiz
    review_result = review_chain.invoke({
        "subject": subject,
        "quiz": quiz_result.content
    })

    return quiz_result.content, review_result.content




