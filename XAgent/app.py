import os, sys
import random
import pickle

# from XAgent.Agent.agent import Agent
from Agent.utils import state_log
from Agent import constraints
import streamlit as st
import logging
import time
from importlib import reload
import json
from XAgent.Agent.agent import Agent
from XAgent.Agent.nlu import NLU

logging.shutdown()
reload(logging)
from Agent.mode import *
from datetime import datetime
import transformers
import torch
PATH = os.path.dirname(__file__)
sys.path.append(PATH)
model = "meta-llama/Llama-2-7b-chat-hf"

st.title("XAgent")


# Initialize chat history
# @st.cache_resource
# def get_agent():
#     with open("./agent.pkl", "rb") as f:
#         agent = pickle.load(f)
#         return agent
@st.cache_resource
def get_nlu_component():
    nlu = NLU()
    return nlu

if 'xagent' not in st.session_state:
    agent = Agent(get_nlu_component())
    st.session_state.mode = None
    st.session_state.use_llm = True
    # 
    #     agent = pickle.load(f)
    st.session_state.xagent = agent
    st.session_state.suggest_question = False
    st.session_state.question = None
    now = datetime.now()
    st.session_state.dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    logging.CON = 25

    # st.llm = transformers.pipeline(
    # "text-generation",
    # model=model,
    # torch_dtype=torch.float16,
    # device_map="auto")

logging.basicConfig(filename=f'logs/{st.session_state.dt_string}.log', level=logging.CON)
# with server_state_lock["llm"]:  # Lock the "count" state for thread-safety
@st.cache_resource
def get_llm():
    llm = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto")
    return llm


st.llm = get_llm()


def status_response(text):
    if "how are you" in text:
        return "great, and you?"


def greeting_response(text):
    text = text.lower()

    # Bots greeting response
    bot_greetings = ['Howdy!', 'Hi!', 'Hello!', 'Greetings!']

    # User greeting
    user_greetings = ['hi', 'hey', 'hello', 'greetings', 'wassup']

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)


exit_list = ['exit', 'see you later', 'bye', 'quit', 'break', 'stop', 'ok, thanks, bye']

# conversations.append(f"Xagent: {msg}")
llm_prompt = """
Improve the language of the text but keep the original intent of the below text. Do not add or omit any information, only adapt the language. Please keep the information in square brackets unchanged. Only return the text without the double quotes. Do not ask for anything else. Do not add any comments.
"{}". 
Improved text:
"""


def response(user_input):
    print(st.session_state.mode)
    if user_input.lower() in exit_list:
        msg = 'Chat with you later, and remember... stay safe!'
        # print_log("xagent", msg)
    else:
        # print('\033[1m\033[94m' + bot_name+': ' + '\033[0m')
        if msg := greeting_response(user_input):
            pass
        elif msg := status_response(user_input):
            pass
            # print(status)
            # logging.log(logging.CON,f"Xagent: {status}")
            # print_log("xagent", msg)
            #             conversations.append(f"Xagent: {status}")
        elif msg := st.session_state.xagent.response(user_input, []):
            pass
            # logging.log(logging.CON,f"Xagent: {dataset}")
            # print_log("xagent", msg)
    return msg


# def ask_for_feature():
#     if len(st.session_state.exist_feature) == 0:
#         msg = "which feature?"
#         # print(f"\033[1m\033[94mX-Agent:\033[0m {msg}")
#         # logging.log(25, f"Xagent: {msg}")
#         # print_log("xagent", msg)
#         # user_input = print_log("user")
#         while user_input not in st.session_state.feature:
#             msg = f"please choose one of the following features: {st.session_state.feature}"
#             print_log("xagent", msg)
#             user_input = print_log("user")
#         st.session_state.exist_feature.append(user_input)


def on_input_change(prompt):
    user_input = prompt
    # print("app 22")
    print(st.session_state.mode)
    print(st.session_state.question)
    if st.session_state.mode == MODE_SUGGEST_QUESTION:
        st.session_state.choice = user_input
    # if st.session_state.mode == MODE_ASK_FOR_CLS:
    #     st.session_state.choice = user_input
    #     if st.session_state.choice not in st.session_state.data:
    #         msg = f"Please give me the target label in {str(st.session_state.data)}: "
    #         # print_log("xagent", msg)
    #         st.experimental_rerun()

        # st.session_state.mode = MODE_QUESTION


    # st.session_state.dialog.append({'type': 'normal', 'role': 'user', 'data': user_input})
    # st.session_state.dialog.append({'type': 'normal', 'role': 'user', 'data': '<img width="100%" height="200" src="./temp.png"/>'})
    # st.session_state.user_input = ''
    # print(f"User: {user_input}" )
    msg = response(user_input)
    return msg
    # answer = "hello"
    # st.session_state.bot.append({'type': 'normal', 'data': f'{answer}'})


def on_btn_click():
    del st.session_state.dialog[:]


bot_name = "X-Agent"
# conversations = [{'type': 'normal', 'role': 'bot', 'data': constraints.welcome_msg}]
# conversations = [{'type': 'normal', 'role': 'bot', 'data': '<img width="100%" height="100%" src="/app/static/temp.png"/>'}]
# st.session_state.setdefault(
#     'dialog',
#     conversations
# )
# st.session_state.dialog.append({'type': 'normal', 'role': 'bot', 'data': '<br><br><img width="100%" height="100%" src="/app/static/temp.png"/> <br><br><br> '})

# st.title("Chat placeholder")

if "messages" not in st.session_state:
    st.session_state.messages = []
    intro_msg = constraints.welcome_msg_test
    # f = open("dataset_info/german-credit.json","r")
    # german_info = json.load(f)
    # feature_info = german_info["feature_description"]
    # for feature in feature_info:
    #     intro_msg += f"  \n{feature} | {feature_info[feature]}  \n "
    st.session_state.messages.append(
                {"role": "assistant", "content": constraints.welcome_msg_test, "type": "text"})
    state_log("assistant", intro_msg)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message['type'] == "image":
            st.image(message["content"])
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
    state_log("user", prompt)
    # logging.log(25, f"user: {prompt}")
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        assistant_response = on_input_change(prompt)
        image = None
        if type(assistant_response) is tuple:
            image = assistant_response[1]
            st.image(image)

            assistant_response = assistant_response[0]
        full_response = ""
        # assistant_response = random.choice(
        #     [
        #         "Hello there! How can I assist you today?",
        #         "Hi, human! Is there anything I can help you with?",
        #         "Do you need help?",
        #     ]
        # )
        # Simulate stream of response with milliseconds delay
        if st.session_state.mode != MODE_SUGGEST_QUESTION and st.session_state.use_llm == True:
            temp_prompt = llm_prompt.format(assistant_response)
            len_txt = len(assistant_response.split())
            sequence = st.llm(temp_prompt,
                              do_sample=True,
                              top_k=50,
                              num_return_sequences=1,
                              max_new_tokens=len_txt*3
                              )
            assistant_response = sequence[0]['generated_text'].split("Improved text:")[1]
        for chunk in assistant_response.split(" "):
            if chunk == "\n":
                chunk = "  \n"
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history

    st.session_state.messages.append({"role": "assistant", "content": full_response, "type": "text"})
    state_log("assistant", full_response)
    if image is not None:
        st.session_state.messages.append({"role": "assistant", "content": image, "type": "image"})
        state_log("assistant", image)

# chat_placeholder = st.empty()
# with chat_placeholder.container():
#     for i in range(len(st.session_state['dialog'])):
#         if st.session_state['dialog'][i]['role'] == 'user':
#             message(st.session_state['dialog'][i]['data'], is_user=True, key=f"{i}_user")
#         else:
#             if st.session_state['dialog'][i]['type'] == "image":
#                 st.image(st.session_state['dialog'][i]['data'])
#             else:
#                 print(st.session_state['dialog'][i]['data'])
#                 message(
#                     st.session_state['dialog'][i]['data'],
#                     key=f"{i}",
#                     allow_html=True,
#                     is_table=True if st.session_state['dialog'][i]['type'] == 'table' else False
#                 )


#     st.button("Clear message", on_click=on_btn_click)
# with st.container():

#     input_text = st.text_input("User Input:", on_change=on_input_change, key="user_input")
# js = f"""
# <script>
# function scroll(dummy_var_to_force_repeat_execution){{
# var textAreas = parent.document.querySelectorAll('section.main');
# for (let index = 0; index < textAreas.length; index++) {{
#     textAreas[index].style.color = 'red'
#     textAreas[index].scrollTop = textAreas[index].scrollHeight;
# }}
# }}
# scroll({len(st.session_state.dialog)})
# </script>
# """

# st.components.v1.html(js)
#     # if input_text:
#     #     st.experimental_rerun()
#     # else:
#     #     st.stop()
