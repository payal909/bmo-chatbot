import streamlit as st
import utils_new 
# Page layout and styling
util_new.setup_page()

# Session initialization
session = st.session_state
util_new.setup_session(session)

# Load Gemini model
# gemini_llm = setup_llm()

# List of documents for selection
all_documents = {
    "Payment Services Act": {
        "data": "./data/Payment Services Act 2019 - Notice on regulatory returns.pdf"
    },
    "OCBC": {
        "data": "./data/2023-annual-report-OCBC.pdf"
    },
    "Bank X": {
        "data": "./data/First column Second column Third column Payment service Requ (1).pdf"
    },
    "BMO": {
        "data": "./data/bmo_ar2022 (2).pdf"
        },
    "NBC": {
        "data": "./data/NATIONAL BANK OF CANADA_ 2022 Annual Report (1).pdf"
    }
}

institutes = all_documents.copy()
del institutes["Payment Services Act"]

# Sidebar UI
with st.sidebar:
    st.markdown("# Reg Reporting Assistant")
    institute = st.selectbox("Institute", options=institutes.keys(), disabled=session.analyze_disabled)

    def analyse():
        with st.spinner("Loading documents..."):
            session.analyze_disabled = True
            session.institute = institute
            session.docs = {
                "BCAR": "./data/Basel Capital Adequacy Reporting (BCAR) 2023 (2).pdf",
                f"{session.institute} Annual Report": all_documents[session.institute]["data"],
                
            }
            session.input_disabled = False
            session.transcript.append(["assistant", "How can I help you today?"])

    st.button("Load Documents", use_container_width=True, disabled=session.analyze_disabled, on_click=analyse)

# Chat input
user_input = st.chat_input("Query", disabled=session.input_disabled)

if user_input:
    session.transcript.append(["user", user_input])
    with st.spinner("Processing..."):
        bot_details, bot_output = utils_new.compare_answer(
            user_input, session.docs
        )
    session.transcript.append(["system", bot_details, user_input])
    session.transcript.append(["assistant", bot_output])

# Chat output rendering
if session.transcript:
    for message in session.transcript:
        if message[0] == "system":
            with st.sidebar:
                with st.expander(message[2]):
                    st.write(message[1])
        else:
            st.chat_message(message[0]).write(message[1])
