import os
import time
import logging
import streamlit as st
from gtts import gTTS
import tempfile
import pygame  # For playing audio
import threading  # To handle audio playback in a separate thread
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from googletrans import Translator
from deep_translator import GoogleTranslator

from htmlTemplates import css, bot_template, user_template

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
PDF_DIRECTORY = 'pdfs'
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

logging.basicConfig(level=logging.INFO)

# Language mapping
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi', 
    'Marathi': 'mr', 
    'Kannada': 'kn', 
    'Punjabi': 'pa', 
    'Tamil': 'ta', 
    'Telugu': 'te', 
    'Arabic': 'ar', 
    'Urdu': 'ur', 
    'Japanese': 'ja', 
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Chinese': 'zh-CN',
    'Portuguese': 'pt'
}

# Supported TTS languages
TTS_SUPPORTED_LANGUAGES = ['en', 'hi', 'es', 'fr', 'de', 'zh-CN', 'pt', 'ja']

# Manually defined FAQ mapping
FAQ_QUESTIONS = {
    "What is the purpose of this application?": 
        "This is a multilingual PDF chat assistant that allows you to interact with PDF documents in multiple languages. "
        "You can load PDFs, ask questions, and get answers in your preferred language.",
    
    "How do I use the application?": 
        "1. Select your preferred language in the sidebar. "
        "2. Click 'Process Documents' to load your PDFs. "
        "3. Ask questions about the documents in the chat input.",
    
    "What languages are supported?": 
        "Currently supported languages include: English, Hindi, Marathi, Spanish, French, German, "
        "Chinese, Arabic, Russian, Japanese, and Portuguese.",
    
    "Can I ask complex questions?": 
        "Yes! The application uses advanced retrieval techniques to help you find detailed "
        "information from your uploaded PDFs across multiple languages.",
    
    "Is my data secure?": 
        "Your documents and conversations are processed locally and are not stored or shared externally. "
        "The application uses advanced language models to provide responses."
}

# Global variable to track audio playback
is_playing = False

# Initialize translator
translator = Translator()

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def load_vector_db():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vectorstore = Chroma(
            collection_name=VECTOR_STORE_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        logging.info("Loaded existing Chroma vector database.")
    else:
        pdf_docs = [os.path.join(PDF_DIRECTORY, f) for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
        if not pdf_docs:
            raise FileNotFoundError(f"No PDF files found in {PDF_DIRECTORY}.")

        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)

        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        vectorstore.persist()
        logging.info("Chroma vector database created and persisted.")

    return vectorstore

def get_conversation_chain(vectorstore):
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
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory
    )
    logging.info("Conversation chain created successfully.")
    return conversation_chain

def translate_text(text, target_lang):
    try:
        if target_lang == 'en':
            return text
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

# Text-to-Speech Function with Comprehensive Error Handling
def speak_text(text, selected_language):
    """
    Convert text to speech using gTTS and play audio with improved error handling
    
    Args:
        text (str): Text to be converted to speech
        selected_language (str): Selected language name
    """
    global is_playing
    
    # Prevent multiple simultaneous playbacks
    if is_playing:
        st.warning("Audio is already playing. Please wait.")
        return
    
    def play_audio_thread():
        global is_playing
        is_playing = True
        
        try:
            # Determine language code
            language_code = LANGUAGES.get(selected_language, 'en')
            
            # Fallback to English if language not supported by gTTS
            if language_code not in TTS_SUPPORTED_LANGUAGES:
                language_code = 'en'
                st.warning(f"Language {selected_language} not supported. Falling back to English.")
            
            # Use a unique temporary filename with a specific extension
            temp_dir = tempfile.gettempdir()
            temp_filename = os.path.join(temp_dir, f"tts_audio_{time.time()}.mp3")
            
            # Generate speech
            tts = gTTS(text=text, lang=language_code)
            tts.save(temp_filename)
            
            # Play the audio
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            
            # Safely remove the temporary file
            try:
                os.unlink(temp_filename)
            except Exception as cleanup_error:
                logging.warning(f"Could not delete temporary audio file: {cleanup_error}")
        
        except Exception as e:
            st.error(f"Text-to-Speech error: {e}")
            logging.error(f"TTS Error: {e}")
        
        finally:
            is_playing = False
    
    # Start audio playback in a separate thread
    threading.Thread(target=play_audio_thread, daemon=True).start()

def handle_userinput(user_question, is_faq=False):
    """Process user questions and retrieve answers."""
    if not user_question.strip():
        st.error("Please enter a valid question.")
        return

    try:
        selected_language = st.session_state.selected_language
        target_lang_code = LANGUAGES.get(selected_language, 'en')
        
        # Check if it's a predefined FAQ
        if is_faq and user_question in FAQ_QUESTIONS:
            response_text = FAQ_QUESTIONS[user_question]
        else:
            # Existing conversation chain logic for non-FAQ questions
            if st.session_state.conversation is None:
                st.error("The documents need to be processed first.")
                return
            
            if target_lang_code != 'en':
                translated_question = translator.translate(user_question, dest='en').text
            else:
                translated_question = user_question
            
            start_time = time.time()
            response = st.session_state.conversation({'question': translated_question})
            processing_time = time.time() - start_time
            logging.info(f"Response generated in {processing_time:.2f} seconds.")
            
            response_text = response['answer']
        
        # Translate response back to selected language
        translated_response = translate_text(response_text, target_lang_code)
        # Append to chat history
        st.session_state.chat_history.append({"user": user_question, "bot": translated_response})    
    except Exception as e:
        st.error(f"An error occurred while processing your question: {e}")
        logging.error(f"Error in handle_userinput: {e}")
        return None

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
    
    # Language selection and FAQ buttons in sidebar
    with st.sidebar:
        st.subheader("Language Selection")
        st.session_state.selected_language = st.selectbox(
            "Choose your preferred language:", 
            list(LANGUAGES.keys()), 
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
        )

        st.subheader("Documents Loading")
        if st.button("Process Documents"):
            with st.spinner("Processing"):
                try:
                    vectorstore = load_vector_db() # Load or create the Chroma vector store
                    
                    st.session_state.conversation = get_conversation_chain(vectorstore) # Create a conversation chain
                    st.success("Documents have been processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred while processing the PDFs: {e}")
                    logging.error(f"Error in processing PDFs: {e}")
        
        # FAQ Buttons
        st.subheader("Frequently Asked Questions")
        faq_questions = list(FAQ_QUESTIONS.keys())
        
        # Create two columns for FAQ buttons
        col1= st.columns(1)[0]
        
        with col1:
            for i in range(0, len(faq_questions), 1):
                if st.button(faq_questions[i]):
                    handle_userinput(faq_questions[i], is_faq=True)

    # Chat input
    user_question = st.chat_input("Ask a question:")

    if user_question:
        handle_userinput(user_question)

    # Display chat history
    if st.session_state.chat_history:
        for idx, message in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 messages
            # User message
            st.write(user_template.replace("{{MSG}}", message["user"]), unsafe_allow_html=True)
            
            # Bot message with TTS button
            bot_msg_div = bot_template.replace("{{MSG}}", message["bot"])
            st.write(bot_msg_div, unsafe_allow_html=True)
            
            # Add TTS button next to bot message
            tts_button = st.button(f"🔊 Listen", key=f"tts_1_{idx}")
            if tts_button:
                # Pass the full language name, not the code
                speak_text(message["bot"], st.session_state.selected_language)


if __name__ == '__main__':
    main()