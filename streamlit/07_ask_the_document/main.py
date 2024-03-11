import streamlit as st
import PyPDF2
import io
import openai
import docx2txt
import pyperclip
import os

# Set OpenAI API key
openai.api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() if page.extract_text() else ""
    return text

# Function to list PDF files in a directory
def list_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

# Function to list DOCX files in a directory
def list_docx_files(directory):
    docx_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            docx_files.append(os.path.join(directory, filename))
    return docx_files

# Function to generate a question from text using OpenAI's GPT
def get_question_from_gpt(text):
    prompt = text[:4096]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", messages=[{"role": "system", "content": "Extractive QA:"}, {"role": "user", "content": prompt}])
    return response.choices[0].message['content'].strip()

# Function to generate an answer to a question using OpenAI's GPT
def get_answer_from_gpt(text, question):
    prompt = text[:4096] + "\n\nQuestion: " + question + "\nAnswer:"
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", messages=[{"role": "system", "content": "Extractive QA:"}, {"role": "user", "content": prompt}])
    return response.choices[0].message['content'].strip()

# Defining the main function of the Streamlit app
def main():
    st.title("Ask Questions From Pdf Document in Folder")

    # Get the folder containing PDF files using folder input
    pdf_folder = st.text_input("Enter the folder path containing PDF files:")

    if pdf_folder and os.path.isdir(pdf_folder):
        pdf_files = list_pdf_files(pdf_folder)
        
        if not pdf_files:
            st.warning("No PDF files found in the folder")
        else:
            st.info(f"Number of PDF files found: {len(pdf_files)}")
        
        # Select a PDF
        selected_pdf = st.selectbox("Select a PDF file", pdf_files)
        st.info(f"Selected PDF file: {selected_pdf}")

        # Extract text from the selected PDF
        text = extract_text_from_pdf(selected_pdf)
       
        # Generating a question from the extracted text using OpenAI's GPT
        question = get_question_from_gpt(text)
        st.write("Question: " + question)

        # Creating a text input for the user to ask a question
        user_question = st.text_input("Ask a question about the document:")
        if user_question:
            answer = get_answer_from_gpt(text, user_question)
            st.write("Answer: " + answer)
            if st.button("Copy Answer Text"):
                pyperclip.copy(answer)
                st.success("Answer text copied to clipboard")

# Run the main function if the script is run directly
if __name__ == "__main__":
    main()
