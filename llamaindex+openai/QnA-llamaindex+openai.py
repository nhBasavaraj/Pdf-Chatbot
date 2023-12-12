import streamlit as st
import uuid
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import os

st.set_page_config(page_title="Chat with your uploaded docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = "ENTER_API_KEY_HERE"
st.title("Chat with your uploaded docs, powered by LlamaIndex 💬🦙")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a document and ask me questions about it!"}
    ]

uploaded_file = st.file_uploader("Upload a document", type="pdf")

if uploaded_file:
    # Create the "uploads" directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    # Create a unique filename for the uploaded file using uuid4
    filename = f"uploaded_document_{uuid.uuid4().hex}.pdf"

    # Save the uploaded file directly to the permanent location
    with open(os.path.join("uploads", filename), "wb") as f:
        f.write(uploaded_file.read())

    # Use SimpleDirectoryReader to read the uploaded document
    reader = SimpleDirectoryReader(input_dir="uploads", recursive=True)
    docs = reader.load_data()

    # Index the uploaded document
    index = VectorStoreIndex.from_documents(
        docs,
        service_context=ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the uploaded document and your job is to answer technical questions. Assume that all questions are related to the uploaded document. Keep your answers technical and based on facts – do not hallucinate features.")
        ),
    )

    # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)  # Add response to message history
else:
    st.info("Please upload a document to start the Q&A session.")
