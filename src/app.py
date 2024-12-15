import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# LLM settings
llm_model = 'llama-3.2-11b-vision-preview'
llm_temperature = 0

# app config
st.set_page_config(page_title="WebChat v1")
st.title("Targeted Conversation")

# sidebar
with st.sidebar:
    st.header("SETUP")
    website_url = st.text_input("Website URL")

with st.sidebar:
    GROQ_API_KEY = st.text_input("Groq API Key", type="password")
    "[Get Groq API Key](https://console.groq.com/keys)"

with st.sidebar:
    st.text("developed by Lutfi Salim")


def get_vectorstore_from_url(url):
    # load document
    loader = WebBaseLoader(url)
    document = loader.load()

    # split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        # chuck_overlap = 30,
        # length_fuction = len,
    )
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from chunks
    vector_store = FAISS.from_documents(document_chunks, HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-mpnet-base-v2',
        model_kwargs = {'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': False}
    ))

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGroq(
        groq_api_key = GROQ_API_KEY,
        model_name = llm_model,
        temperature = llm_temperature,
    )

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search question to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversation_rag_chain(retriever_chain):
    llm =  ChatGroq(
        groq_api_key = GROQ_API_KEY,
        model_name = llm_model,
        temperature = llm_temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response["answer"]
 

if website_url is None or website_url == "":
    st.info("please place a url")

elif GROQ_API_KEY is None or GROQ_API_KEY == "":
    st.info('please input Groq API Key')

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Yes master, I'll provide relevant information from the link you placed"),
        ]
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # user input
    user_query = st.chat_input("What's information you need?")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))


    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)