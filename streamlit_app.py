import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain, ConversationChain
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    SystemMessage,
)
import requests
from streamlit_lottie import st_lottie
import re


from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
ctx = get_script_run_ctx()
session_id = ctx.session_id

st.set_page_config(page_title="The Quiz Bot",
                   layout="wide", page_icon="🤖")
#setup sidebar for topics
if 'topics' not in st.session_state:
    st.session_state.topics = ''
if 'tutorial' not in st.session_state:
    st.session_state.tutorial = False
#style buttons
st.markdown(
    """<style>
        .element-container button {
            height: 3em;
            width: 15em;
            text-align: center;
        }
        </style>""",
    unsafe_allow_html=True,
)
def setSessionState(topics, tutorial=False):
    st.session_state.topics = topics
    st.session_state.tutorial =tutorial

if st.sidebar.button('String'):
    setSessionState('String Manipulation')
if st.sidebar.button('Number'): 
    setSessionState('String Manipulation, Simple arithmatic calculation')
if st.sidebar.button('Function'): 
    setSessionState('String Manipulation, Simple arithmatic calculation, Simple Function')
if st.sidebar.button('Loop'): 
    setSessionState('String Manipulation, Simple arithmatic calculation, Simple Function, Simple Loop')
if st.sidebar.button('Conditional'): 
    setSessionState('String Manipulation, Simple arithmatic calculation, Simple Function, Simple Loop, Basic If Else conditionals')
if st.sidebar.button('List and Dictionary'): 
    setSessionState('String Manipulation, Simple arithmatic calculation, Simple Function, Simple Loop, Basic If Else conditionals, Basic List and Dictionary')
if st.sidebar.button('File IO'): 
    setSessionState('String Manipulation, Simple arithmatic calculation, Simple Function, Simple Loop, Basic If Else conditionals, Basic List and Dictionary, Basic File IO on pathlib')
if st.sidebar.button('Reset'):
    setSessionState('')
if st.sidebar.button('Check Tutorial Answer'):
    setSessionState('',True)

if st.session_state.topics:
    topics=f'Limit coverage to the topics: {st.session_state.topics}'

else:
    topics='Will work on all these topics: String Manipulation, Simple arithmatic calculation, Simple Function, Simple Loop, Basic If Else conditionals, Basic List and Dictionary, Basic File IO on pathlib'
st.sidebar.write(topics)

# HuggingFace Modelcard
model_mistral8B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llama3_70B = "meta-llama/Meta-Llama-3-70B-Instruct"
llama3p1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"


llm = HuggingFaceEndpoint(
    repo_id=llama3p1_70B,
    # task="text-generation",
    max_new_tokens=1000,
    do_sample=True,
    temperature=1.6,
    repetition_penalty=1.1,
    return_full_text=False,
    top_k=400,
    top_p=0.5,
    huggingfacehub_api_token=st.secrets["hf_token"]
)

# You are a chatbot who specialized in quiz questions on Python programming language for beginners.

prompt = ChatPromptTemplate.from_messages(
    [
        # SystemMessage(
        #     content="""
        #     - You are a chatbot that specialized in quiz questions on General knowledge,Python programming, Microsoft Excel, Statistics and Economics. 
        #     - Always begin your conversation by asking which topic the user would like to quiz.
        #     - The quiz should contain different levels of difficulty.
        #     - Keep track of the number of right and wrong answers.
        #     - Review the strength and weakness at the end of the quiz.
        #    """
        # ),  # The persistent system prompt
        SystemMessage(
            content=f"""
            - You are a chatbot that specialized in handling questions and queries on Python programming. {topics}
            - You will help to create practical hands-on question for user to practice, upon request, you will provide answers as well
            - When you provide practical hands-on questions, try your best to provide one random question on all type of businesses, including but not limited to accounting, marketing, human capital, tourism, banking and finance, international trade, etc
            - You will help to grade and troubleshoot submitted code based on correctness, readability, efficiency and proper comment of the code, and award a mark to the submission, then provid an explanation and improvement
            - You will help to set MCQ quiz questions on various difficulty levels
            - Review the strength and weakness at the end of the quiz.
        """
        ),  # The persistent system prompt
        # MessagesPlaceholder(
        #     variable_name="chat_history"
        # ),  # Where the memory will be stored.
        MessagesPlaceholder(
             variable_name="history"
        ),
        HumanMessagePromptTemplate.from_template(
            "{input}"
        ),  # Where the human input will injected
    ]
)

def getResponse(question):
    #response = chat_llm_chain.predict(human_input=question)
    response = chat_llm_chain.invoke({'input': question},{"configurable": {"session_id": session_id}})
    # exclude 'assistant' from response
    response = response[9:]

    # regex on human to remove Humam
    human = re.search(r"Human:.*|human:.*", response)

    if human is not None:
        # exclude "Human:" located at end of string
        response = response[:human.start()]
        st.chat_message("ai").write(response)
    else:
        st.chat_message("ai").write(response)

# ---- set up lottie icon ---- #
# python icon
url = "https://lottie.host/afd755b7-2ead-4ac6-a75e-02b05054871e/SKQzuvxmW2.json"
# quiz icon
url = "https://lottie.host/1897011a-cf70-491f-8618-82c16d5c2fa2/d4LAW696Ly.json"
url = requests.get(url)
url_json = dict()

if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in the URL")


def setTopics(topic):
    topic_string=f'Limit coverage to the topics: {topic}'
    return topic_string

st_lottie(url_json,
          reverse=True,  # change the direction
          height=230,  # height and width
          width=230,
          speed=1,  # speed
          loop=True,  # run forever like a gif
          quality='high',  # options include "low" and "medium"
          key='bot'  # Uniquely identify the animation
          )

# ----- page header -----#
# st.markdown("<p style='text-align: left; font-size:2rem'>The Quiz Show</p>",
#            unsafe_allow_html=True)

# ----- set up chat history to retain memory-----#

# set up chat history in streamlit as session state does not work
chat_msgs = StreamlitChatMessageHistory(key="special_app_key")
# set history size
chat_history_size = 10

# if len of chat history is 0, clear and add first message
if len(chat_msgs.messages) == 0:
    chat_msgs.clear()
    chat_msgs.add_ai_message(
        """Hi there! What is your name?""")
elif len(chat_msgs.messages) >= 10:
    chat_msgs.messages=chat_msgs.messages[1:]

# handle LLMA3.1 prompt format: 'human', 'assistant'
# refer to https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/

# strings are located in msg.content
for msg in chat_msgs.messages:
    # remove assistant role
    msg.content = msg.content.replace('assistant', '')
    # regex on human to remove Humam
    human = re.search(r"Human:.*|human:.*", msg.content)
    if human is not None:
        # human.start() is index position 9
        msg.content = msg.content[:human.start()]
        # remove <|eot_id|> before writing to chat history
        st.chat_message(msg.type).write(msg.content.replace(
            '<|eot_id|>', ''))
    else:
        st.chat_message(msg.type).write(msg.content.replace(
            '<|eot_id|>', ''))

# Langchain ConversationBufferMemory
# memory = ConversationBufferMemory(
#     memory_key='chat_history',
#     chat_memory=chat_msgs,
#     k=chat_history_size,
#     return_messages=True
# )
# memory = ConversationSummaryMemory(
#     llm=llm,
#     memory_key='chat_history',
#     chat_memory=chat_msgs,
#     k=chat_history_size,
#     return_messages=True
# )



def get_session_history() -> BaseChatMessageHistory:
    if 'store' not in st.session_state:
        st.session_state.store = ChatMessageHistory()
    # if len(st.session_state.store) >= 5:
    #     return st.session_state.store[-5:]
    # else:
    return st.session_state.store

runnable=prompt | llm
chat_llm_chain = RunnableWithMessageHistory(
    runnable,
    lambda session_id: chat_msgs, #get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    
)
# ------ set up llm chain -----#
# chat_llm_chain = LLMChain(
#     llm=llm,
#     prompt=prompt,
#     verbose=True,
#     memory=memory,
# )
# chat_llm_chain=ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True,
#     prompt=prompt
# )
if st.session_state.tutorial:
    question = st.text_area("Paste tutorial question here")
    answer = st.text_area("Paste tutorial answer here")
    if st.button('Check Answer'):
        question = f"Verify the answer to the question and provide feedback only. You are forbidden to provide correct answer directly.\n Question: {question}\n Answer: {answer} <|eot_id|>"
        with st.spinner("Checking your answer ..."):
            getResponse(question)
    if st.button('Generate Similar Questions'):
        question=f'Given a question below, generate a similar question for practice.\nQuestion: {question} <|eot_id|>'
        with st.spinner("Working on similar question..."):
            getResponse(question)
else:
    # ------ generate questions and responses ------#

    if question := st.chat_input("Your Answer..."):
        with st.spinner("Grrrr..."):
            # question needs to include <|eot_id|> to end interaction
            # refer to llama3.1 prompt format
            question = f"{question} <|eot_id|>"
            getResponse(question)

    st.write(topics)
