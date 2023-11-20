import streamlit as st
from llama_index import GPTVectorStoreIndex, ServiceContext, download_loader
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with the DP bot, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the DP, powered by LlamaIndex 💬🦙")
st.info("To help go beyond and above", icon="📃")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the nomenclature!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the nomenclature – hang tight! This should take 1-2 minutes."):
        # Determine the source based on a condition or prompt
        source_type = "md"  # Default to Markdown
        prompt = st.session_state.messages[-1]["content"].lower() if st.session_state.messages else ""

        # Change the source type based on the presence of a keyword in the prompt
        if "estimate" in prompt:
            source_type = "gsheets"

        try:
            if source_type == "md":
                # Load data from Markdown files
                reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
                docs = reader.load_data()
            elif source_type == "gsheets":
                # Load data from Google Sheets
                GoogleSheetsReader = download_loader("GoogleSheetsReader")
                # Use the Google Sheets link and specify the sheet name (e.g., "Sheet1")
                loader = GoogleSheetsReader(sheet_link="https://docs.google.com/spreadsheets/d/1VikaU0VGE2_8jwAREDQZT1nG7rMia_3l2ttr9Y95VbQ/edit#gid=0", sheet_name="Sheet1")
                docs = loader.load_data()
            else:
                # Default to Markdown if the condition is not met
                raise ValueError(f"Invalid source type: {source_type}")

            service_context = ServiceContext.from_defaults(
                chunk_size=512, chunk_overlap=50, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2, system_prompt="Given the nature of the question, please provide a detailed response based on either the Markdown (.md) files or Google Sheets data. If the question is related to estimation, use information from Google Sheets; otherwise, use data from Markdown files. Respond truthfully, and if you don't know the answer, indicate so. Thank you!")
            )

            index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
            query_engine = index.as_query_engine(similarity_top_k=2)
            return index, query_engine
        except Exception as e:
            st.error(f"An error occurred while loading data: {e}")
            return None, None

index, query_engine = load_data()

if index is not None:  # Check if data loading was successful
    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = query_engine  # Use query engine for chat

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If the last message is not from the assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
