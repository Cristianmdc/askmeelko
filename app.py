import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
import openai

# Load OpenAI API key
openai.api_key = st.secrets["openai_key"]

# App Header
st.header("Chat with the Pellet Mill Manual 📘💬")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! Ask me anything about the Pellet Mill manual."}
    ]

# Load and index data
@st.cache_resource
def load_data():
    with st.spinner("Loading and indexing the Pellet Mill manual..."):
        reader = SimpleDirectoryReader(input_dir="./data")
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on Pellet Mill manuals.")
        )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Chat functionality
if user_input := st.chat_input("Your question:"):
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(user_input)
            st.write(response.response)
            st.session_state["messages"].append({"role": "assistant", "content": response.response})

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
