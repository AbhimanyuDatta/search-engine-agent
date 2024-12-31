import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler # displays thoughts and actions of agents when they are interacting with the tools


load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name='Search')

tools = [arxiv, wiki, search]

st.title('LangChain - Chat with Search')

st.sidebar.title('Settings')

api_key = st.sidebar.text_input('Enter your Groq API Key:', value=os.getenv('GROQ_API_KEY'), type='password')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': 'Hi, I am a chatbot who can search the web. How can I help you?'}
    ]
    
for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])
    
if prompt := st.chat_input(placeholder='What is machine learning'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.chat_message('user').write(prompt)
    
    llm = ChatGroq(api_key=api_key, model='gemma2-9b-it', streaming=True)
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error=True)
    
    with st.chat_message('assistant'):
        callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[callback])
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)
        