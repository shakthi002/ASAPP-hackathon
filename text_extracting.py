import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings  # Updated import
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from image_summarizer import summarizer
from pdf_processing import count_pdf_pages, pdf_page_to_base64

def extract(file_path):
    # Set API keys as environment variables
	os.environ['PINECONE_API_KEY'] = '63cc481a-df02-4a58-8dea-cc4905005f70'
	os.environ["COHERE_API_KEY"] = "V5jqL0gNm1xo9gIzN0x32c9Nc814b4gh9gpOY6EY"
	loader = PyMuPDFLoader(file_path)
	documents = loader.load()
	# Combine the text of all the documents into a single string
	docs = [doc.page_content for doc in documents]
	# Initialize Cohere embeddings
	embeddings = CohereEmbeddings(
		model="embed-english-v3.0"
	)
	
	# Initialize the text splitter with specified parameters
	text_splitter = RecursiveCharacterTextSplitter(
		separators=['\n\n'],  # Adjust the separators as needed
		chunk_size=1000,
		chunk_overlap=200,
		length_function=len,
		is_separator_regex=False
	)
	
	# Split the text into chunks (docs)
	docs = text_splitter.create_documents(docs)
	
	# Define your Pinecone index name
	index_name = "asapp-temp1"

	vector_store = PineconeVectorStore(index=index_name, embedding=embeddings)
	
	# Embed and store the documents in Pinecone
	PineconeVectorStore.from_documents(
		docs, 
		embedding=embeddings, 
		index_name=index_name  # Ensure the index name is correct
	)

	count = count_pdf_pages(file_path)
	img_paths = pdf_page_to_base64(file_path,count)
	summaries_image = summarizer(img_paths)

	# Embed and store the documents in Pinecone
	PineconeVectorStore.from_texts(
		summaries_image, 
		embedding=embeddings, 
		index_name=index_name  # Ensure the index name is correct
	)