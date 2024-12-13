import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access your Google API key
api_key = os.getenv('GOOGLE_API_KEY')

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Sidebar (Left panel)
st.sidebar.title("Have Conversation With your PDF :")
uploaded_file = st.sidebar.file_uploader("Upload the PDF", type="pdf")


# Main Content (Right panel)
st.title("I am your Book Assistant:📖") 

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load PDF document using the temporary file path
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Vector Embeddings and Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key=api_key)
    seerah_db = Chroma.from_documents(split_docs, embeddings)

    # Chat interface
    st.markdown("### Chat with the Book Assistant")

    # User input
    query = st.text_input("Ask a question about your Uploaded PDF:")

    if query:
        # Perform a similarity search on the vector store
        result = seerah_db.similarity_search(query)
        st.write("Search Results:", result)

        # Use the ChatGoogleGenerativeAI model for language generation
        llm1 = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-002",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key
        )

        # Create a prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context. 
        Think step by step before providing a detailed answer. 
        <context>
        {context}
        </context>
        Question: {input}
        """)

        # Create a document chain using the LLM and prompt template
        document_chain = create_stuff_documents_chain(llm1, prompt)

        # Create a retriever from the vector store
        retriever = seerah_db.as_retriever()

        # Create a retrieval chain using the retriever and document chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get the response to the query
        response = retrieval_chain.invoke({"input": query})
        st.write("Answer:", response['answer'])

# Footer in sidebar
st.sidebar.markdown("Knowledge Assistant - Powered by LangChain and Google Generative developed by [@kamran_dar]: https://github.com/kamrandar \n \n LinkedIn: https://www.linkedin.com/in/kamran-dar-144460b9/")
