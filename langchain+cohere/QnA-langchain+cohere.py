import streamlit as st
import fitz
from langchain.llms import Cohere
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Initialize Streamlit session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        pdf_contents = uploaded_file.read()

        # Use PyMuPDF to extract text from the PDF
        pdf_document = fitz.open(stream=pdf_contents, filetype="pdf")
        texts = [page.get_text() for page in pdf_document.pages()]

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(texts)

        # Select embeddings
        cohere_api_key = ''  # Replace with your actual API key
        embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)

        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)

        # Create retriever interface
        retriever = db.as_retriever()

        # Create QA chain without prompt_truncation
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(cohere_api_key=cohere_api_key),
            chain_type='stuff',
            retriever=retriever
        )

        # Run the query
        response_text = qa.run(query_text)

        # Add the current conversation to the session state
        st.session_state.conversation.append({"user": query_text, "bot": response_text})

        return response_text

# Page title
st.set_page_config(page_title='🤖🔗 Cohere Q&A Chatbot')
st.title('🤖🔗 Cohere Q&A Chatbot')

# File upload
uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')
# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file, key="user_input")

# Generate response on Enter key press
if st.session_state.user_input and uploaded_file:
    with st.spinner('Calculating...'):
        response = generate_response(uploaded_file, st.session_state.user_input)
    st.info(response)

# Display previous conversations in descending order
if st.session_state.conversation:
    st.header("Previous Conversations")
    for conv in reversed(st.session_state.conversation):
        st.text(f"User 👨‍🦱: {conv['user']}")
        st.text(f"Bot 🤖: {conv['bot']}")
        st.text("----------------")
