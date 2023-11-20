import streamlit as st
from pathlib import Path
from llama_index import VectorStoreIndex, ServiceContext, download_loader
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with the DP bot, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the DP, powered by LlamaIndex 💬🦙")
st.info("To help go beyond and above", icon="📃")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the nomenclature!"}
    ]

@st.cache_resource(show_spinner=False)
def load_markdown_data():
    with st.spinner(text="Loading and indexing the Markdown data – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()

        service_context = ServiceContext.from_defaults(
            chunk_size=512, chunk_overlap=50, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2, system_prompt="Respond truthfully, and if you don't know the answer, indicate so. Thank you!")
        )

        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        query_engine = index.as_query_engine(similarity_top_k=2)
        return index, query_engine

@st.cache_resource(show_spinner=False)
def load_excel_data():
    with st.spinner(text="Loading and indexing the Excel data – hang tight! This should take 1-2 minutes."):
        PandasExcelReader = download_loader("PandasExcelReader")
        loader = PandasExcelReader(pandas_config={"header": 0})
        # Assuming the sheet name is "2023"
        docs = loader.load_data(file=Path('./data/Energy Consumption Analysis.xlsx'), sheet_name="2023")

        service_context = ServiceContext.from_defaults(
            chunk_size=512, chunk_overlap=50, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2, system_prompt="Given the nature of the question, please provide a detailed response based on the Excel (.xlsx) data. If you don't know the answer, indicate so. Thank you!")
        )

        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        query_engine = index.as_query_engine(similarity_top_k=2)
        return index, query_engine

# Load separate indices for Markdown and Excel data
markdown_index, markdown_query_engine = load_markdown_data()
excel_index, excel_query_engine = load_excel_data()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = {"md": markdown_query_engine, "xlsx": excel_query_engine}

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Determine the source based on a condition or prompt
source_type = "md"  # Default to Markdown
prompt = st.session_state.messages[-1]["content"].lower() if st.session_state.messages else ""

# Change the source type based on the presence of a keyword in the prompt
if "estimate" in prompt:
    source_type = "xlsx"

# Use the corresponding index based on the source type
if source_type == "md":
    index, query_engine = markdown_index, st.session_state.chat_engine["md"]
else:
    index, query_engine = excel_index, st.session_state.chat_engine["xlsx"]

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
