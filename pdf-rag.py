import streamlit as st
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import json

from htmlTemplates import css, bot_template, user_template

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = r"pdfs\nyayabandhu.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = r".\chroma_db"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = PyPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        raise FileNotFoundError(f"PDF file not found at path: {doc_path}")



def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")

    logging.info(f"Number of documents in vector DB: {vector_db._collection.count()}")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")

    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    st.header("NYAYSETU")

    # Inject CSS
    st.markdown(css, unsafe_allow_html=True)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display previous chat messages
    for message in st.session_state.chat_history:
        if message['type'] == 'user':
            st.markdown(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

    # User input
    user_input = st.chat_input("Enter your question:")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input
        })

        with st.spinner("Generating response..."):
            try:
                # Alternative LLM initialization
                llm = ChatOllama(
                    model=MODEL_NAME,
                    temperature=0.7,
                    format="json"
                )

                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    st.stop()

                retriever = create_retriever(vector_db, llm)
                chain = create_chain(retriever, llm)

                # Process response
                raw_response = chain.invoke(input=user_input)




                try:
                    # Parse the JSON response
                    parsed_response = json.loads(raw_response)

                    # Check if the response contains data
                    if isinstance(parsed_response, dict) and parsed_response:
                        # Extract the description (value of the first key)
                        response_description = next(iter(parsed_response.values()))
                    else:
                        # Handle empty or invalid response
                        response_description = "Sorry, I could not generate a response. Please try rephrasing your question."

                    # Log the extracted description
                    logging.info(f"Extracted Description: {response_description}")

                    # Add bot response to chat history
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response_description
                    })

                except json.JSONDecodeError:
                    logging.error("Failed to decode JSON response.")
                    st.error("An error occurred while processing the bot's response.")
                except StopIteration:
                    logging.error("Response dictionary is empty.")
                    st.error("No response could be generated. Please try again.")




                # Rerun to update the display
                st.rerun()

            except Exception as e:
                logging.error(f"Detailed Error: {e}", exc_info=True)
                st.error(f"An error occurred: {str(e)}")

    # Clear chat history button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()