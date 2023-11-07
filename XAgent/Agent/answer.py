import copy
import json
import logging
import sys, os
import shap
from dtreeviz.trees import *

from mode import *
from utils import state_log
from xai_methods import *

# print the JS visualization code to the notebook
shap.initjs()
PATH = os.path.dirname(__file__)
sys.path.append(PATH)
import streamlit as st
import constraints
import tensorflow as tf

tf.get_logger().setLevel(40)  # suppress deprecation messages
tf.compat.v1.disable_v2_behavior()  # disable TF2 behaviour as alibi code still relies on TF1 constructs
import matplotlib.pyplot as plt
import numpy as np

import os

l_shap_questions = []


class Answers:
    def __init__(self, list_node, clf, clf_display, current_instance, question, l_exist_classes, l_exist_features,
                 l_instances, data, df_display_instance, predicted_class, dataset_anchor, clf_anchor, orig_quest,
                 preprocessor=None):
        self.list_node = list_node
        self.clf = clf
        self.clf_display = clf_display
        self.clf_anchor = clf_anchor
        self.dataset_anchor = dataset_anchor
        self.question = question
        self.current_instance = current_instance
        self.l_exist_classes = l_exist_classes
        self.l_exist_features = l_exist_features
        st.session_state.exist_feature = l_exist_features
        self.l_instances = l_instances
        self.l_classes = data['classes']
        st.session_state.data = data['classes']
        self.l_features = data['features']
        st.session_state.feature = data['features']
        self.data = data
        self.df_display_instance = df_display_instance
        self.predicted_class = predicted_class
        self.preprocessor = preprocessor
        self.original_question = orig_quest
        file = open('Agent/qa.json', 'r')
        self.question_answer = json.load(file)
        file.close()

    def extract_relation(self, test_data, clf_list, feature_name):
        relation = []
        for i, clf in enumerate(clf_list):
            relation.append({})
            for j, (v1, v2) in enumerate(zip(test_data, clf[:-1])):
                if v1 != v2:
                    if type(v1) == str or type(v2) == str:
                        relation[i][feature_name[j]] = "not suitable"
                    else:
                        if v1 < v2:
                            relation[i][feature_name[j]] = "too low"
                        else:
                            relation[i][feature_name[j]] = "too high"
        return relation

    def answer(self, question, conversations=[]):
        print(f"question: {question}")
        if st.session_state.mode != MODE_ASK_FOR_CLS:

            if type(question) is tuple:
                question = question[0]
            if question == "unknown":
                return constraints.question_msg
            df = pd.read_csv(os.path.join(PATH, "Median_4.csv"))
            st.session_state.id_question = df[df['Question'] == question]["Label"].iloc[0]
            print(f"question id: {st.session_state.id_question}")
            if st.session_state.id_question in constraints.l_dice_question_ids:
                if len(self.l_classes) == 2:
                    class_P = [label for label in self.l_classes if label != self.predicted_class][0]
                else:
                    class_P = None
                for c in self.l_exist_classes:
                    if c != self.predicted_class:
                        class_P = c
                if class_P == None:
                    msg = f"Please give me the target label in {str(self.data['classes'])}: "
                    # print_log("xagent", msg)
                    st.session_state.mode = MODE_ASK_FOR_CLS
                    return msg
        else:
            class_P = st.session_state.choice
            st.session_state.mode = MODE_QUESTION
            # class_P = print_log("user")
        if st.session_state.id_question in constraints.l_dice_question_ids:
            if st.session_state.id_question in constraints.l_feature_questions_ids:
                if len(self.l_exist_features) == 0:
                    return ask_for_feature(self)
                e1 = dice_answer(self, class_P, st.session_state.exist_feature)
                st.session_state.mode = MODE_QUESTION
                if e1.cf_examples_list[0].final_cfs_df is None:
                    return constraints.no_cf_msg.format(st.session_state.exist_feature)
            else:
                if self.data['info']['name'] == "mnist":
                    return cf_proto(self, class_P)
                e1 = dice_answer(self, class_P)

            # json_e1 = e1.to_json()
            # js = json.loads(json_e1)
            ans = ""
            # if id_question in constraints.l_feature_questions_ids:
            #     ans = ""
            # else:
            # js = {'cfs_list': e1.cf_examples_list[0].final_cfs_df.to_list()

            if e1.cf_examples_list[0].final_cfs_df is None:
                return constraints.no_cf_msg.format(st.session_state.exist_feature)
            test_instance_df = e1.cf_examples_list[0].test_instance_df
            features = test_instance_df.columns[:-1]
            test_instance_df = test_instance_df.values.tolist()[0]
            cf_instance = e1.cf_examples_list[0].final_cfs_df.values.tolist()

            if st.session_state.id_question in constraints.l_dice_question_relation_ids:
                relation = self.extract_relation(test_instance_df, cf_instance, features )
                ans = "There are multiple reasons for this result, one of them is: \n"
                ans += " and ".join([str(k) + " is " + str(v) for k, v in relation[0].items()]) + "."
            ans_relation = []
            for j, (v1, v2) in enumerate(zip(test_instance_df, cf_instance[0][:-1])):
                if v1 != v2:
                    if type(v1) == str or type(v2) == str:
                        s = features[j] + " should be changed to " + str(v2)
                    else:
                        if v1 < v2:
                            s = features[j] + " should be increased to " + str(v2)
                        else:
                            s = features[j] + " should be decreased to " + str(v2)
                    ans_relation.append(s)
            ans += "The " + " and ".join(ans_relation) + ", to get " + self.data['info']['change_ans'][
                self.l_classes.index(class_P)]
            return ans
        if st.session_state.id_question in constraints.l_new_predict_question_ids:
            temp_instance = None
            for i in range(0, len(st.session_state.exist_feature)):
                f = st.session_state.exist_feature[i]
                index_feature = self.l_features.index(f)
                temp_instance = copy.copy(self.current_instance)
                temp_instance[index_feature] = self.l_instances[0][i]
            return "new instance is: " + str(temp_instance) + " and the predicted class is" + str(
                self.clf.predict([temp_instance]))
        if st.session_state.id_question in constraints.l_shap_question_ids:
            img = shap_explainer(self, st.session_state.id_question)
            if img == "Which feature?":
                return img
            if st.session_state.id_question in constraints.l_shap_question_feature:
                return (self.data['info']["feature_ans"] + self.data['info']['why_ans'], img)
            elif st.session_state.id_question in constraints.l_shap_question_single_feature:
                return (constraints.ans_shap_question_single_feature + self.data['info']['why_ans'], img)
            else:
                return (self.data['info']['why_ans'], img)
        if st.session_state.id_question in constraints.l_anchor_question_ids:
            return anchor_answer(self)
        if st.session_state.id_question in constraints.l_terminology_question_ids:
            prompt = """
            Question:{}. Explain it in 100 tokens.
            Answer:
            """.format(self.original_question)
            sequence = st.llm(prompt,
                              do_sample=True,
                              top_k=50,
                              num_return_sequences=1,
                              max_new_tokens=100
                              )
            answer = sequence[0]['generated_text'].split("Answer:")[1]
            return answer
        if st.session_state.id_question in constraints.l_others:
            str_id = str(st.session_state.id_question)
            if str_id in self.question_answer:
                return self.question_answer[str_id]['answer']

        return constraints.cant_answer_msg
