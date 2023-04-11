import copy
import json
import logging

import shap
from dtreeviz.trees import *

from XAgent.Agent.utils import print_log
from XAgent.Agent.xai_methods import *
# print the JS visualization code to the notebook
shap.initjs()


from XAgent.Agent import constraints
import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
from alibi.explainers import CounterfactualProto

import os
PATH = os.path.dirname(__file__)
l_shap_questions = []
class Answers:
    def __init__(self, list_node, clf, clf_display, current_instance, question, l_exist_classes, l_exist_features,
                 l_instances, data, df_display_instance, predicted_class, dataset_anchor, clf_anchor, preprocessor=None):
        self.list_node = list_node
        self.clf = clf
        self.clf_display = clf_display
        self.clf_anchor = clf_anchor
        self.dataset_anchor = dataset_anchor
        self.question = question
        self.current_instance = current_instance
        self.l_exist_classes = l_exist_classes
        self.l_exist_features = l_exist_features
        self.l_instances = l_instances
        self.l_classes = data['classes']
        self.l_features = data['features']
        self.data = data
        self.df_display_instance = df_display_instance
        self.predicted_class = predicted_class
        self.preprocessor = preprocessor

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
        if type(question) is tuple:
            question = question[0]
        if question == "unknown":
            return constraints.question_msg
        df = pd.read_csv(os.path.join(PATH,"Median_4.csv"))
        id_question = df[df['Question'] == question]["Label"].iloc[0]
        if id_question in constraints.l_dice_question_ids:
            if len(self.l_classes) == 2:
                class_P =[label for label in self.l_classes if label != self.predicted_class][0]
            else:
                class_P = None
            for c in self.l_exist_classes:
                if c != self.predicted_class:
                    class_P = c
            if class_P == None:
                msg = f"Please give me the target label in {str(self.data['classes'])}: "
                print_log("xagent",msg)
                class_P = print_log("user")
            if id_question in constraints.l_feature_questions_ids:
                ask_for_feature(self)
                e1 = dice_answer(self, class_P, self.l_exist_features)
                if e1.cf_examples_list[0].final_cfs_df is None:
                    return constraints.no_cf_msg.format(self.l_exist_features)
            else:
                if self.data['info']['name'] == "mnist":
                    return cf_proto(self, class_P)
                e1 = dice_answer(self, class_P)

            json_e1 = e1.to_json()
            js = json.loads(json_e1)
            ans = ""
            # if id_question in constraints.l_feature_questions_ids:
            #     ans = ""
            # else:
            if id_question in constraints.l_dice_question_relation_ids:
                relation = self.extract_relation(js['test_data'][0][0], js['cfs_list'][0], js['feature_names'])
                ans = "There are multiple reasons for this result, one of them is: \n"
                ans += " and ".join([str(k) + " is " + str(v) for k, v in relation[0].items()]) + "."
            ans_relation = []
            for j, (v1, v2) in enumerate(zip(js['test_data'][0][0], js['cfs_list'][0][0][:-1])):
                if v1 != v2:
                    if type(v1) == str or type(v2) == str:
                        s = js['feature_names'][j] + " should be changed to " + str(v2)
                    else:
                        if v1 < v2:
                            s = js['feature_names'][j] + " should be increased to " + str(v2)
                        else:
                            s = js['feature_names'][j] + " should be decreased to " + str(v2)
                    ans_relation.append(s)
            ans += "The " + " and ".join(ans_relation) + ", to get " + self.data['info']['change_ans'][
                self.l_classes.index(class_P)]
            return ans
        if id_question in constraints.l_new_predict_question_ids:
            temp_instance = None
            for i in range(0, len(self.l_exist_features)):
                f = self.l_exist_features[i]
                index_feature = self.l_features.index(f)
                temp_instance = copy.copy(self.current_instance)
                temp_instance[index_feature] = self.l_instances[0][i]
            return "new instance is: " + str(temp_instance) + " and the predicted class is" + str(
                self.clf.predict([temp_instance]))
        if id_question in constraints.l_shap_question_ids:
            shap_explainer(self, id_question)
            if id_question in constraints.l_shap_question_feature:
                return self.data['info']["feature_ans"] + self.data['info']['why_ans']
            elif id_question in constraints.l_shap_question_single_feature:
                return constraints.ans_shap_question_single_feature + self.data['info']['why_ans']
            else:
                return self.data['info']['why_ans']
        if id_question in constraints.l_anchor_question_ids:
            return anchor_answer(self)
        return constraints.cant_answer_msg
