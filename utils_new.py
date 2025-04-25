import os
import streamlit as st
import google.generativeai as genai
from google import genai
import pathlib

def setup_page():
    st.set_page_config(layout="wide")
    hide = """
    <style>
    MainMenu {visibility:hidden;}
    header {visibility:hidden;}
    footer {visibility:hidden;}
    </style>
    """
    st.markdown(hide, unsafe_allow_html=True)

def setup_session(session):
    if 'transcript' not in session:
        session.transcript = []
    if 'input_disabled' not in session:
        session.input_disabled = True
    if 'analyze_disabled' not in session:
        session.analyze_disabled = False
    if 'institute' not in session:
        session.institute = ""

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def compare_answer( question, docs):
    retrival_system_prompt = f"""You are a Reg Reporting Assistant, You need to extract as much content as you can which is related or relevant to the answer of the user question from the context provided.
Do not try to answer the question, just extract the text relevant to the answer of the user question that will help user to find their answer further.
Use the document for finding out the relevant text: from question: {question}

"""
    
    summary = dict()
    for doc_name,  path in docs.items():
        filepath = pathlib.Path(path)
        sample_file = client.files.upload(file = filepath)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[sample_file, retrival_system_prompt]
        )
        summary[doc_name] = response.text

        compare_context = "\n\n".join([f"Relevant points from {doc_name}:\n\n{doc_summary}" for doc_name, doc_summary in summary.items()])
        institute = doc_name
    details = "\n\n" + question + "\n\n" + compare_context
    context = compare_context
    finalprompt = f"""You are a Reg Reporting Assistant who has to answer the question of a user from the institute {institute}.
Below is a list of relevant points along with the name of the document from where these points are from.
Consider all the documents provided to you and answer the question by analyzing the relevant points from the {institute} and Payment Services Act both.
Just give the concluded response between the relevant points from document 1 and document 2(put it as points or step by step if possible).
It's banking related question give if possible mention the references from the documents.
Exception: If you have asked a question like based on the Payment service that {institute} follows that is not mentioned in the annual report, then go through the following context:
(Account issuance service, Domestic money transfer service, Cross-border money transfer service, Merchant acquisition service, E-money issuance service) are the set of payment services {institute} provides out of this list and answer the question by taking these payment services into account.
{context}
the human question is as follow:
human question: {question}
""" 
    final_response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[ finalprompt]
    )
    
    return details, final_response.text



def two_documents(question, docs):
    retrival_system_prompt = f"""You are a Reg Reporting Assistant who has to answer the question of a user from the institute .
Below is a list of documents
Consider all the documents provided to you and answer the question by analyzing the relevant points from the particular bank and BCAR documnent.
Just give the concluded response between the relevant points from documents.
It's banking related question give if possible mention the references from the documents. 
The human question is as follow:
human question: {question}
""" 
    uploaded_files = {}

    for i, (doc_name, path) in enumerate(docs.items(), start=1):
        filepath = pathlib.Path(path)
        uploaded_file = client.files.upload(file=filepath)
        uploaded_files[f"sample_file_{i}"] = uploaded_file
        
    response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[uploaded_files["sample_file_1"],
        uploaded_files["sample_file_2"], retrival_system_prompt]
        )
    details = "tgygyg"
    
    return details, response.text

