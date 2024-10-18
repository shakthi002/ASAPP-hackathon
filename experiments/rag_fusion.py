import os
import fitz
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import (
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    ChatPromptTemplate
)
from langchain.load import dumps, loads
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#from pinecone import Pinecone
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, EnsembleRetriever
from langchain.llms.cohere import Cohere
from langchain_groq import ChatGroq


import warnings
warnings.filterwarnings("ignore")

pinecone_api_key = "a60a1d84-e360-4473-9338-41eb04046fba"
os.environ['PINECONE_API_KEY'] = pinecone_api_key

groq_api_key = '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'
os.environ['GROQ_API_KEY'] = groq_api_key

# Initialize embeddings model
def init_bge_embedding_model():
    """Initialize HuggingFace BGE embedding model."""
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    return HuggingFaceBgeEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )

# Initialize Pinecone vector store
def init_pinecone_vector_store(docs):
    """Initialize Pinecone vector store from documents.
    
    Args:
        docs (list): List of chunked documents.
        
    Returns:
        PineconeVectorStore: Initialized vector store with documents.
    """
    api_key = "a60a1d84-e360-4473-9338-41eb04046fba"
    os.environ['PINECONE_API_KEY'] = api_key
    pc = Pinecone(api_key=api_key)
    index_name = "paper-text"
    
    return PineconeVectorStore.from_documents(
        docs,
        index_name=index_name,
        embedding=bge_embed_model
    )

def load_pinecone_vector_store(index_name, embed_model):
    """Initialize Pinecone vector store from documents.
    
    Args:
        index_name : Existing Pinecone Vector index name
        embed_model : Embedding model used
        
    Returns:
        PineconeVectorStore: Initialized vector store with documents.
    """

    return Pinecone.from_existing_index(index_name = index_name, embedding = embed_model)

# Load and split PDF documents
def load_and_split_pdfs(root_dir):
    """Load and split PDF documents into chunks.
    
    Args:
        root_dir (str): Directory containing PDF files.
    
    Returns:
        list: List of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    docs = []
    
    for file in os.listdir(root_dir):
        if file.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(root_dir, file))
            document = loader.load()
            chunked_documents = text_splitter.split_documents(document)
            docs.extend(chunked_documents)
    
    return docs

def init_groq_llm():
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        max_retries=2,
        max_tokens = 3500,
        temperature = 0.2
    )

    return llm

# Initialize Cohere LLM
def init_cohere_llm():
    """Initialize Cohere LLM using API key.
    
    Returns:
        Cohere: Initialized Cohere LLM.
    """
    api_key = 'V5jqL0gNm1xo9gIzN0x32c9Nc814b4gh9gpOY6EY'
    os.environ["COHERE_API_KEY"] = api_key
    
    return Cohere(cohere_api_key=api_key,
                max_retries = 3,
                max_tokens = 3500,
                temperature = 0.2
                )

# Generate search queries using a prompt template
def generate_search_queries(prompt, llm):
    """Generate search queries based on an input query.
    
    Args:
        prompt (ChatPromptTemplate): Prompt template to generate queries.
        llm (Cohere): Language model to generate the output.
    
    Returns:
        function: Function to generate multiple search queries.
    """
    return prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))

# Reciprocal rank fusion for ranking results
def reciprocal_rank_fusion(results: list[list], k=60):
    """Perform reciprocal rank fusion to rerank the search results.
    
    Args:
        results (list[list]): List of lists of search results.
        k (int): Ranking parameter.
    
    Returns:
        list: Reranked results.
    """
    fused_scores = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return reranked_results

# Main function to execute RAG chain with query
def execute_rag_chain(query, ragfusion_chain):
    """Execute the RAG chain and return results for a given query.
    
    Args:
        query (str): User's query.
    
    Returns:
        list: Reranked results.
    """    
    results = ragfusion_chain.invoke({"question": query})
    return reciprocal_rank_fusion(results)

# Set up retriever, compressor, and final RAG chain
def setup_retrievers_and_chain(docs, vectorstore_from_docs):
    """Set up retrievers, compressor, and final RAG chain."""

    retriever = vectorstore_from_docs.as_retriever(k=3)
    #bm25_retriever = BM25Retriever.from_documents(docs, k = 2)
    
    #ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],
    #                                       weights=[0.4, 0.6])
    
    compressor = LLMChainExtractor.from_llm(llm)
    global compression_retriever
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    prompt = ChatPromptTemplate(input_variables=['original_query'],
                                messages=[SystemMessagePromptTemplate(
                                    prompt=PromptTemplate(input_variables=[], 
                                                          template='You are a helpful assistant that generates multiple search queries based on a single input query.')),
                                          HumanMessagePromptTemplate(
                                              prompt=PromptTemplate(input_variables=['original_query'], 
                                                                    template='Generate multiple search queries related to: {question} \n OUTPUT (2 queries):'))])

    generate_queries = generate_search_queries(prompt, llm)
    
    ragfusion_chain = generate_queries | compression_retriever.map()

    return ragfusion_chain

# Final RAG chain for question answering
def final_rag_question_answering(context, question):
    """Generate answer using the final RAG chain based on context and question.
    
    Args:
        context (list): Context from reranked results.
        question (str): User's question.
    
    Returns:
        str: Final answer generated by the RAG chain.
    """
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    final_rag_chain = prompt | llm | StrOutputParser()
    
    
    return final_rag_chain.invoke({"context": context, "question": question})

# Main execution
if __name__ == "__main__":
    # Initialize components
    bge_embed_model = init_bge_embedding_model()
    docs = load_and_split_pdfs(os.getcwd())

    #vectorstore_from_docs = init_pinecone_vector_store(docs)

    index_name = 'paper-text'
    vectorstore_from_docs = load_pinecone_vector_store(index_name, bge_embed_model)

    #llm = init_cohere_llm()

    llm = init_groq_llm()

    # Set up retrievers and compression chain
    ragfusion_chain = setup_retrievers_and_chain(docs, vectorstore_from_docs)

    # Execute the RAG chain with a sample query
    query = 'What is a rolling buffer cache? Which model uses it?'
    reranked_results = execute_rag_chain(query, ragfusion_chain)

    # Get the final answer
    final_answer = final_rag_question_answering(reranked_results, query)

    print(f'User Query: \n{query}\n++++++\n')
    print(f'Response: \n{final_answer}\n\n')
