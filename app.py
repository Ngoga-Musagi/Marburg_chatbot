import streamlit as st
import requests
import io
from PyPDF2 import PdfReader
import tabula
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Function to load PDF from GitHub
def load_single_pdf_from_github(repo, file_path, access_token):
    url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
    headers = {"Authorization": f"token {access_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        
        tables = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True, encoding='ISO-8859-1')
        table_text = ""
        for i, table in enumerate(tables):
            table_text += f"Table {i+1}:\n{table.to_string()}\n\n"
        
        combined_text = text + "\nExtracted Tables:\n" + table_text
        return combined_text
    else:
        st.error(f"Failed to download the file. Status code: {response.status_code}")
        return None

# Function to create vector store
def create_vector_store(pdf_content):
    doc = Document(page_content=pdf_content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    documents = text_splitter.split_documents([doc])
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Initialize components
@st.cache_resource
def initialize_components(pdf_content):
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = create_vector_store(pdf_content)
    
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "Marburg_news_search",
        "Search for information about Marburg!",
    )
    search = TavilySearchResults(api_key=st.secrets["TAVILY_API_KEY"])
    tools = [retriever_tool, search]
    
    llm_with_tools = llm.bind_tools(tools)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Medical assistant of RBC that helps health practitioners get information from guidelines about Marburg. Use the content to get the summary about the disease and prevention.If a question is not related to Marburg, politely inform the user that you can only answer Marburg-related questions"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

# Streamlit UI
st.title("Marburg Chatbot")

# GitHub PDF retrieval
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = None

if st.session_state.pdf_content is None:
    st.write("Retrieving PDF from GitHub...")
    REPO = "Ngoga-Musagi/Medical_chatbot"
    FILE_PATH = "Rwanda National Viral Hemorrhagic Fevers Guideline final Draft 2024 V(1)(1).pdf"
    ACCESS_TOKEN = st.secrets["GITHUB_ACCESS_TOKEN"]
    
    pdf_content = load_single_pdf_from_github(REPO, FILE_PATH, ACCESS_TOKEN)
    if pdf_content:
        st.session_state.pdf_content = pdf_content
        st.success("PDF retrieved successfully!")
    else:
        st.error("Failed to retrieve PDF. Please check your GitHub settings and try again.")
        st.stop()

# Initialize chat components
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = initialize_components(st.session_state.pdf_content)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)
    
    with st.spinner("Thinking..."):
        result = st.session_state.agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history
        })
    
    ai_response = result["output"]
    st.session_state.chat_history.append(AIMessage(content=ai_response))
    st.chat_message("assistant").write(ai_response)

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()