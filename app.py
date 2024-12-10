import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
import openai

# Load OpenAI API key
openai.api_key = st.secrets["openai_key"]

# App title
st.title("Pellet Mill Chatbot")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Ask me anything about the Pellet Mill manual!"}
    ]

# Load and index the data
@st.cache_resource(show_spinner=True)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5)
    )
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Chat UI
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if user_input := st.chat_input("Type your question here:"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(user_input)
            st.write(response.response)
            st.session_state["messages"].append({"role": "assistant", "content": response.response})
