import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, Document
from llama_index.llms import OpenAI
import openai

# Set OpenAI API key
openai.api_key = st.secrets["openai_key"]

# Title for the Streamlit app
st.header("Chat with Your Documents 💬 📄")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything about the data you uploaded."}
    ]

# Function to load and index data
@st.cache_resource
def load_data(directory="./data"):
    reader = SimpleDirectoryReader(input_dir=directory)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5)
    )
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

# Load data
index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Chat UI
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if user_input := st.chat_input("Ask your question here:"):
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(user_input)
            st.write(response.response)
            st.session_state["messages"].append({"role": "assistant", "content": response.response})
