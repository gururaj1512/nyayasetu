import os
import time
import logging
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from googletrans import Translator
from deep_translator import GoogleTranslator

# Import HTML templates (ensure this file exists in your project)
from htmlTemplates import css, bot_template, user_template

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
PDF_DIRECTORY = 'pdfs'  # Path to the folder containing PDFs
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Language mapping
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi', 
    'Marathi': 'mr', 
    'Spanish': 'es', 
    'French': 'fr', 
    'German': 'de', 
    'Chinese': 'zh-CN', 
    'Arabic': 'ar', 
    'Russian': 'ru', 
    'Japanese': 'ja', 
    'Portuguese': 'pt'
}

# Initialize translator
translator = Translator()

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    """Split the extracted text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Create or load a Chroma vector store
def load_vector_db():
    """Load or create a Chroma vector database."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Check if the ChromaDB vector store already exists
    if os.path.exists(PERSIST_DIRECTORY):
        vectorstore = Chroma(
            collection_name=VECTOR_STORE_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        logging.info("Loaded existing Chroma vector database.")
    else:
        # Load PDFs and process them
        pdf_docs = [os.path.join(PDF_DIRECTORY, f) for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
        if not pdf_docs:
            raise FileNotFoundError(f"No PDF files found in {PDF_DIRECTORY}.")

        # Extract text and split into chunks
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)

        # Create a new Chroma vector store
        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        vectorstore.persist()
        logging.info("Chroma vector database created and persisted.")

    return vectorstore

# Create a conversation chain using Ollama LLM
def get_conversation_chain(vectorstore):
    """Create a conversation retrieval chain."""
    llm = ChatOllama(
        model=MODEL_NAME, 
        max_tokens=300, 
        temperature=0.7
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Limit to top 5 results
        memory=memory
    )
    logging.info("Conversation chain created successfully.")
    return conversation_chain

# Translate text to target language
def translate_text(text, target_lang):
    """Translate text to target language."""
    try:
        if target_lang == 'en':
            return text
        
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

# Handle user input
def handle_userinput(user_question):
    """Process user questions and retrieve answers."""
    if not user_question.strip():
        st.error("Please enter a valid question.")
        return
    
    if st.session_state.conversation is None:
        st.error("The documents need to be processed first.")
        return
    
    try:
        # Get selected language
        selected_language = st.session_state.selected_language
        target_lang_code = LANGUAGES.get(selected_language, 'en')
        
        # Translate input question if not in English
        if target_lang_code != 'en':
            translated_question = translator.translate(user_question, dest='en').text
        else:
            translated_question = user_question
        
        start_time = time.time()
        response = st.session_state.conversation({'question': translated_question})
        processing_time = time.time() - start_time
        logging.info(f"Response generated in {processing_time:.2f} seconds.")
        
        # Translate response back to selected language
        response_text = response['answer']
        translated_response = translate_text(response_text, target_lang_code)
        
        st.session_state.chat_history.append({"user": user_question, "bot": translated_response})
    except Exception as e:
        st.error(f"An error occurred while processing your question: {e}")
        logging.error(f"Error in handle_userinput: {e}")
        return

    # Display chat history
    for message in st.session_state.chat_history[-5:]:  # Show last 5 messages
        st.write(user_template.replace("{{MSG}}", message["user"]), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message["bot"]), unsafe_allow_html=True)

# Streamlit app configuration
st.set_page_config(page_title="Multilingual PDF Chat", page_icon=":books:")
st.write(css, unsafe_allow_html=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"

# Main app logic
def main():
    st.header("NYAYASETU: Multilingual PDF Assistant")
    
    # Language selection in sidebar
    with st.sidebar:
        st.subheader("Language Selection")
        st.session_state.selected_language = st.selectbox(
            "Choose your preferred language:", 
            list(LANGUAGES.keys()), 
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
        )
        
        st.subheader("Documents Loading")
        st.write("Load and process PDF documents:")

        if st.button("Process Documents"):
            with st.spinner("Processing"):
                try:
                    # Load or create the Chroma vector store
                    vectorstore = load_vector_db()

                    # Create a conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Documents have been processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred while processing the PDFs: {e}")
                    logging.error(f"Error in processing PDFs: {e}")

    # Chat input
    user_question = st.chat_input("Ask a question:")

    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()