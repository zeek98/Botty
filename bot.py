import streamlit as st
from pathlib import Path
from llama_index import VectorStoreIndex, ServiceContext, Document, download_loader
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with the DP bot, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the DP, powered by LlamaIndex 💬🦙")
st.info("To help go beyond and above", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the nomenclature!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the nomenclature – hang tight! This should take 1-2 minutes."):
        # Load data from Markdown (.md) files
        md_reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        md_docs = md_reader.load_data()

        # Load data from CSV file
        csv_loader = download_loader("SimpleCSVReader")
        csv_reader = csv_loader(encoding="utf-8")
        csv_path = Path('./data/Energy Consumption Analysis.csv')
        csv_docs = csv_reader.load_data(file=csv_path)

        # Combine both sets of documents
        all_docs = md_docs + csv_docs

        # Initialize LlamaIndex with the combined documents
        service_context = ServiceContext.from_defaults(
            chunk_size=512, chunk_overlap=50,
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2, system_prompt="You will give the answers truthfully and do not deviate but if you don't know then say I don't Know as the answer."))
        index = VectorStoreIndex.from_documents(all_docs, service_context=service_context)
        query_engine = index.as_query_engine(similarity_top_k=2)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add the response to the message history
