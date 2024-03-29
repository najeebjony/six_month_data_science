import streamlit as st 
from langchain.llms import OpenAI 
# set titlt with smybol of langchain 
st.title("🦜🔗Quickstart App LangChain By Najeeb")
# openai_api_key = st.sidebar.text_input("OpenAI API Key")
openai_api_key = st.secrets["OPENAI_API_KEY"]
def generate_responese(input_text):
    llm = OpenAI(temperature=0.9, openai_api_key= openai_api_key)
    st.info(llm(input_text))
with st.form(key='my_form'):
    input_text = st.text_area("Enter your text here:",'Write Text Here' )
    submit_button = st.form_submit_button(label='Submit')
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key", icon="🔑")
    if submit_button and openai_api_key.startswith("sk-"):
        generate_responese(input_text)  