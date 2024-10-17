import os
import cohere
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_cohere import CohereEmbeddings
from langchain.llms.cohere import Cohere
from langchain_groq import ChatGroq

def query_handling(query):
	os.environ['PINECONE_API_KEY'] = '63cc481a-df02-4a58-8dea-cc4905005f70'
	os.environ["COHERE_API_KEY"] = "V5jqL0gNm1xo9gIzN0x32c9Nc814b4gh9gpOY6EY"
	groq_api_key = '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'
	os.environ['GROQ_API_KEY'] = groq_api_key
	cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

	# Define system and human templates
	system_template = "System message template: {context}"
	human_template = "Human message template: {question}"

	# Create message prompts
	messages = [
		SystemMessagePromptTemplate.from_template(system_template),
		HumanMessagePromptTemplate.from_template(human_template)
	]

	# Create chat prompt template
	chat_prompt = ChatPromptTemplate.from_messages(messages)

	# Use the Cohere client directly as the LLM
	llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])

	# Initialize Cohere embeddings
	# embeddings = CohereEmbeddings(model="embed-english-v3.0")

	# Create Pinecone Vector Store retriever
	index_name = "asapp-temp1"
	embeddings = CohereEmbeddings(
        model="embed-english-v3.0"
    )

	retriever = PineconeVectorStore.from_existing_index(index_name, embeddings).as_retriever(search_kwargs={'k': 3})
	context = [i.page_content for i in retriever.invoke(query)]

	prompt = f"Question: {query} answer using context: {context}"

	response = cohere_client.chat(message=prompt,model="command-r",temperature=0)

	return response.text

	# Create ConversationalRetrievalChain
	'''qa_chain = ConversationalRetrievalChain.from_llm(
		llm=llm,
		retriever=retriever,
		combine_docs_chain_kwargs={"prompt": chat_prompt},
		chain_type="stuff"  # Add chain type for Cohere
	)

	# Example usage

	# query = "ticket price for 4 adult ?"
	context = "Provide the context or background information here."
	chathistory = []

	# search_results = retriever.search(query)
	# print(search_results)
	response = qa_chain.run({"question": query,"chat_history":""})'''
	#return "hi"

