import os
from config import OPEN_AI_API

from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI


import streamlit as st
from PyPDF2 import PdfReader


os.environ["OPENAI_API_KEY"] = OPEN_AI_API


template = """You are an assistant helping the user with his questions from the given context. Answer the following question concisely and professionally.

Question : {question}"""

prompt_template = PromptTemplate(
    input_variables=['question'],
    template=template
)

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #dae3dc
}
.chat-message.bot {
    background-color: #969187
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''

def get_pdf_text(pdf_files):
    
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks

def get_vector_store(text_chunks):
    

    embeddings = OpenAIEmbeddings()
    
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore


def get_conversation_chain(vector_store):
    
    # OpenAI Model

    llm = ChatOpenAI(temperature=0, max_tokens=500)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )

    return conversation_chain

def handle_user_input(question):

    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def clear_input():
    st.session_state.input_text = ""

def main():
    st.set_page_config(page_title='Q&A with documents using OpenAI', page_icon=':ai:')

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Ask questions with reference to documents uploaded')
    question = st.text_input("Type your question here: ")

    if question:
        handle_user_input(question)

    with st.sidebar:
        st.subheader("Document Upload ")
        pdf_files = st.file_uploader("Upload PDF files and press done", type=['pdf'], accept_multiple_files=True)

        if st.button("Done"):
            with st.spinner("Loading PDF..."):

                # Get PDF Text
                raw_text = get_pdf_text(pdf_files)

                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)
                

                # Create Vector Store
                
                vector_store = get_vector_store(text_chunks)
                st.write("DONE")

                # Create conversation chain

                st.session_state.conversation =  get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()