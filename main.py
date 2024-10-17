import streamlit as st
import logging

# Suppress SageMaker SDK info messages
sagemaker_logger = logging.getLogger("sagemaker")
sagemaker_logger.setLevel(logging.ERROR)

from pdf_processing import pdf_page_to_base64, count_pdf_pages
from image_summarizer import summarizer
from vector_embedding import embedd
from text_extracting import extract
from query_handling import query_handling

def handle_research_paper(file_path):
    try:
        extract(file_path)
        return "Research paper summaries have been embedded successfully."
    except Exception as e:
        return f"Error processing research paper: {e}"

def handle_query(query):
    response = query_handling(query)
    return response

def main():
    st.title("LIT BOT")
    st.write("Welcome to the research paper summarizer and query handler!")

    option = st.selectbox("Choose an option:", ('Upload Research Paper', 'Chat with Query'))

    if option == 'Upload Research Paper':
        uploaded_file = st.file_uploader("Choose a PDF file...", type="pdf")

        if uploaded_file is not None:
            file_path = f"{uploaded_file.name}"  # Save the file temporarily
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("Process Paper"):
                message = handle_research_paper(file_path)
                st.success(message)

    elif option == 'Chat with Query':
        query = st.text_input("Enter your query:")
        if st.button("Submit Query"):
            if query:
                response = handle_query(query)
                st.write("Response:")
                st.write(response)
            else:
                st.warning("Please enter a query before submitting.")

if __name__ == "__main__":
    main()
