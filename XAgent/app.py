# import os, sys
import random
from XAgent.Agent.agent import Agent
from XAgent.Agent.utils import print_log
from XAgent.Agent import constraints
import streamlit as st
from streamlit_chat import message
import logging
from importlib import reload
logging.shutdown()
reload(logging)
from XAgent.Agent.mode import *
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
print(dt_string)
logging.CON = 25
logging.basicConfig(filename=f'logs/{dt_string}.log', level=logging.CON)

def on_input_change():
    user_input = st.session_state.user_input
    print("app 22")
    print(st.session_state.mode)
    print(st.session_state.question)
    if st.session_state.mode == MODE_SUGGEST_QUESTION:
        st.session_state.choice = user_input
    if st.session_state.mode == MODE_ASK_FOR_CLS:
        st.session_state.choice = user_input
        if st.session_state.choice not in st.session_state.data :
            msg = f"Please give me the target label in {str(st.session_state.data)}: "
            print_log("xagent", msg)
            st.experimental_rerun()
    if st.session_state.mode == MODE_ASK_FOR_FEATURE:
        if user_input not in st.session_state.feature:
            msg = f"please choose one of the following features: {st.session_state.feature}"
            print_log("xagent", msg)
            st.experimental_rerun()
            st.session_state.exist_feature.append(user_input)
        st.session_state.mode = MODE_QUESTION
    st.session_state.dialog.append({'type': 'normal', 'role': 'user', 'data': user_input})
    # st.session_state.dialog.append({'type': 'normal', 'role': 'user', 'data': '<img width="100%" height="200" src="./temp.png"/>'})
    st.session_state.user_input = ''
    print(f"User: {user_input}" )
    response(user_input)
    # answer = "hello"
    # st.session_state.bot.append({'type': 'normal', 'data': f'{answer}'})

def on_btn_click():
    del st.session_state.dialog[:]
bot_name = "X-Agent"
conversations = [{'type': 'normal', 'role': 'bot', 'data': constraints.welcome_msg}]
# conversations = [{'type': 'normal', 'role': 'bot', 'data': '<img width="100%" height="100%" src="/app/static/temp.png"/>'}]
st.session_state.setdefault(
    'dialog',
    conversations
)
# st.session_state.dialog.append({'type': 'normal', 'role': 'bot', 'data': '<br><br><img width="100%" height="100%" src="/app/static/temp.png"/> <br><br><br> '})
if 'xagent' not in st.session_state:
    agent = Agent()
    st.session_state.mode = None
    st.session_state.xagent = agent
    st.session_state.suggest_question = False
    st.session_state.question = None
st.title("Chat placeholder")

chat_placeholder = st.empty()
with chat_placeholder.container():
    for i in range(len(st.session_state['dialog'])):
        if st.session_state['dialog'][i]['role'] == 'user':
            message(st.session_state['dialog'][i]['data'], is_user=True, key=f"{i}_user")
        else:
            if st.session_state['dialog'][i]['type'] == "image":
                st.image(st.session_state['dialog'][i]['data'])
            else:
                print(st.session_state['dialog'][i]['data'])
                message(
                    st.session_state['dialog'][i]['data'],
                    key=f"{i}",
                    allow_html=True,
                    is_table=True if st.session_state['dialog'][i]['type'] == 'table' else False
                )




    st.button("Clear message", on_click=on_btn_click)
with st.container():

    input_text = st.text_input("User Input:", on_change=on_input_change, key="user_input")
js = f"""
<script>
function scroll(dummy_var_to_force_repeat_execution){{
var textAreas = parent.document.querySelectorAll('section.main');
for (let index = 0; index < textAreas.length; index++) {{
    textAreas[index].style.color = 'red'
    textAreas[index].scrollTop = textAreas[index].scrollHeight;
}}
}}
scroll({len(st.session_state.dialog)})
</script>
"""

st.components.v1.html(js)
    # if input_text:
    #     st.experimental_rerun()
    # else:
    #     st.stop()
def status_response(text):
    if "how are you" in text:
        return "great, and you?"
def greeting_response(text):
    text = text.lower()

    #Bots greeting response
    bot_greetings = ['Howdy!', 'Hi!', 'Hello!', 'Greetings!']

    #User greeting
    user_greetings = ['hi', 'hey', 'hello', 'greetings', 'wassup']

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)

exit_list = ['exit', 'see you later', 'bye', 'quit', 'break', 'stop', 'ok, thanks, bye']


# conversations.append(f"Xagent: {msg}")


def response(user_input):
    print(st.session_state.mode)
    logging.log(logging.CON, f"Xagent: {user_input}")
    if user_input.lower() in exit_list:
        msg = 'Chat with you later, and remember... stay safe!'
        print_log("xagent", msg)
    else:
        # print('\033[1m\033[94m' + bot_name+': ' + '\033[0m')
        if greeting := greeting_response(user_input):
            # print(greeting)
            # logging.log(logging.CON,f"Xagent: {greeting}")
            print_log("xagent", greeting)
            #             conversations.append(f"Xagent: {greeting}")
        elif status := status_response(user_input):
            # print(status)
            # logging.log(logging.CON,f"Xagent: {status}")
            print_log("xagent", status)
            #             conversations.append(f"Xagent: {status}")
        elif dataset := st.session_state.xagent.dataset_response(user_input, conversations):
            # logging.log(logging.CON,f"Xagent: {dataset}")
            if dataset != None:
                print_log("xagent", dataset)
def ask_for_feature():
    if len(st.session_state.exist_feature) == 0:
        msg = "which feature?"
        # print(f"\033[1m\033[94mX-Agent:\033[0m {msg}")
        # logging.log(25, f"Xagent: {msg}")
        print_log("xagent",msg)
        user_input = print_log("user")
        while user_input not in st.session_state.feature:
            msg = f"please choose one of the following features: {st.session_state.feature}"
            print_log("xagent", msg)
            user_input = print_log("user")
        st.session_state.exist_feature.append(user_input)