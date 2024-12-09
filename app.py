!pip install streamlit colab-everything PyPDF2 pandas

# Save the Streamlit app as app.py
%%writefile app.py
import streamlit as st
import pandas as pd
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize models and FAISS index
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
dimension = 384
index = faiss.IndexFlatL2(dimension)

# Upload file
def process_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return [line.decode('utf-8').strip() for line in uploaded_file.readlines()]
    elif uploaded_file.type == "application/vnd.ms-excel" or uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        return df.to_string(index=False).split("\n")
    elif uploaded_file.type == "application/pdf":
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()
        return text.split("\n")
    else:
        return []

def update_knowledge_base(new_documents):
    global documents, index
    documents.extend(new_documents)
    new_doc_embeddings = embedding_model.encode(new_documents, convert_to_tensor=False)
    index.add(np.array(new_doc_embeddings))

def retrieve_relevant_doc(query, top_k=1):
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]

def generate_response(query):
    relevant_docs = retrieve_relevant_doc(query)
    context = " ".join(relevant_docs)
    input_text = f"Context: {context} Query: {query}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("RAG Chatbot with File Upload")
uploaded_file = st.file_uploader("Upload your knowledge base file (.txt, .csv, .pdf)", type=["txt", "csv", "pdf"])
if uploaded_file:
    new_docs = process_file(uploaded_file)
    update_knowledge_base(new_docs)
    st.success("Knowledge base updated!")

query = st.text_input("Ask your question:")
if query:
    response = generate_response(query)
    st.write("Response:", response)

# Run Streamlit app
from colab_everything import share_streamlit
share_streamlit('app.py')
