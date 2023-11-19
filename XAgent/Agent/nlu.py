import pandas as pd
import os
import sys
import  logging
import streamlit as st
# sys.path.append('/homes/bach/XAGENT/XAgent/Agent')
# silence command-line output temporarily
# sys.stdout, sys.stderr = os.devnull, os.devnull
from importlib_resources import files
PATH = os.path.dirname(__file__)
sys.path.append(PATH)
from simcse import SimCSE
from mode import *

from constraints import select_msg, l_support_questions_ids, request_number_msg, request_more_msg


# unsilence command-line output

enabled = True
topk = 400
class NLU:
    def __init__(self):
        self.model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
        self.df = pd.read_csv(files("XAgent").joinpath('Agent/Median_4.csv'), index_col=0).drop_duplicates()
        self.model.build_index(list(self.df['Question']))

    def get_list_questions(self, match_results):
        labels = []
        questions = []
        # print(l_support_questions_ids)
        # print(l_support_questions_ids)
        df_original_questions = pd.read_csv(files("XAgent").joinpath('Agent/original_question.csv'))
        for idx, (question,_) in enumerate(match_results):
            label = self.df.query('Question == @question')['Label'].iloc[0]
            if len(questions) < 5:
                question = df_original_questions.query("Label == @label").iloc[0]['Question']
            if label not in labels and label in l_support_questions_ids:
                questions.append(question)
                labels.append(label)
        # print(questions)
        return questions

    def replace_information(self, question, features, prediction, current_instance, labels):
        if "{X}" in question:
            question = question.replace("{X}", f"{{{features[0]},{features[1]}, ...}}")
        if "{P}" in question:
            question = question.replace("{P}", str(prediction))
        if "{Q}" in question:
            question = question.replace("{Q}", str([label for label in labels if label != prediction]))
        return question
    def match(self, question):
        threshold = 0.6
        match_results = self.model.search(question, threshold=threshold)
        print("Match_result")
        print(match_results)
        logging.log(26, f"question = {question}")
        logging.log(26, f"result = {match_results}")
        if len(match_results) > 0:
            match_question, score = match_results[0]
            # print("hallo" + match_question)
            return match_question
        return None
    def suggest_questions(self, question, features, prediction, current_instance, labels):
        if "match_results" not in st.session_state:
            st.session_state.match_results = self.model.search(question, threshold=0, top_k=topk)
        questions = self.get_list_questions(st.session_state.match_results)
        # print(match_results)
        # ans = "I am not sure what you mean. Can you please choose one of the following questions if there is a match, otherwise, choose 6 to get more questions\n"
        if "choice" not in st.session_state:
            ans = select_msg
            # print_log("xagent", ans)
            for idx, question in enumerate(questions[:5]):
                question = self.replace_information(question, features, prediction, current_instance, labels)
                msg = f"{idx+1}. {question}"
                # print(msg)
                ans += f"{msg}\n"
            msg = "6. See more questions"
            # print(msg)
            ans += f"{msg}"
            # msg = "Please choose the number of the corresponding question"
            # print(msg)
            # ans += f"{msg}\n"
            # logging.log(25, f"Xagent: {ans}")
            # print_log("xagent", ans)
            # print_log("user")
            return ans

        if st.session_state.choice.isnumeric() == False or int(st.session_state.choice) >= topk+1:
            # msg = request_number_msg
            # # print_log("xagent", msg)
            # # print_log("user")
            # return msg
            st.session_state.question = "unknown"
            st.session_state.suggest_question = False
            st.session_state.mode = MODE_QUESTION
            st.session_state.pop("choice")
            st.session_state.pop("match_results")
            return None
        if st.session_state.choice.isnumeric() == True and int(st.session_state.choice) >= topk + 1:
            msg = request_number_msg
            # print_log("xagent", msg)
            # print_log("user")
            return msg
        ans = request_more_msg
        if int(st.session_state.choice) == 0:
            st.session_state.question = "unknown"
            st.session_state.suggest_question = False
            st.session_state.mode = MODE_QUESTION
            st.session_state.pop("choice")
            st.session_state.pop("match_results")
            return None
        if int(st.session_state.choice) == 6:
            for idx, question in enumerate(questions[5:15], start=5):
                question = self.replace_information(question, features, prediction, current_instance, labels)
                msg = f"{idx+1}. {question}"
                ans+=f"{msg}\n"
            # print_log("xagent", ans)
            return ans
            # choice = print_log("user")
            # if int(st.sesssion_state.choice) == 0:
            #     st.session_state.question = "unknown"
            #     st.session_state.suggest_question = False
            # while (True):
            #     if st.sesssion_state.choice.isnumeric():
            #         if int(st.sesssion_state.choice) < 15:
            #             break
            #     msg = request_number_msg
            #     # print_log("xagent", msg)
            #     # choice = print_log("user")
            #     return msg
        st.session_state.mode = MODE_QUESTION
        st.session_state.question = questions[int(st.session_state.choice) - 1]
        st.session_state.pop("match_results")
        st.session_state.pop("choice")
        return None
