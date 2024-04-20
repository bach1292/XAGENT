import logging
from datetime import datetime
# now = datetime.now()
# dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# logging.basicConfig(filename=f'{dt_string}.log', level=logging.INFO)
# logging.info('something happened')
import openml

import json
import random
import re
import os
import openml
import pandas as pd
import shap
import sklearn
from importlib_resources import files
import pickle
import constraints
import sys

# sys.path.append('/homes/bach/XAGENT/XAgent/Agent/')
# from dtreeviz.trees import *
from openml.tasks import TaskType
import numpy as np
# print the JS visualization code to the notebook
# shap.initjs()
import streamlit as st
import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from answer import Answers
import utils
from anchor import anchor_tabular

from constraints import *
# import the desired library
from nlu import NLU

# import sys
# sys.stdout = sys.__stdout__
# sys.stderr = sys.__stderr__
import tensorflow as tf
from tensorflow.keras .layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from mode import *
import PIL

from utils import state_log

# from IPython.display import display
# MODE_INPUT = 0
# MODE_QUESTION = 1
# MODE_DATASET = 1


# datetime object containing current date and time


PATH = os.path.dirname(__file__)

conversations = []


class Agent:
    def __init__(self, nlu):
        self.dataset = "german-credit"
        self.current_instance = None
        self.clf = None
        self.predicted_class = None
        # self.mode = None
        self.data = {"X": None, "y": None, "features": None, "classes": None}
        self.nlu_model = nlu
        self.list_node = []
        self.clf_display = None
        self.l_exist_classes = None
        self.l_exist_features = None
        self.l_exist_values = {}
        self.l_instances = None
        self.df_display_instance = None
        self.current_feature = None
        self.dataset_anchor = None
        self.clf_anchor = None
        self.preprocessor = None

    def preprocess_question(self, question):
        for c in self.data['cls_mapping']:
            for v in self.data['cls_mapping'][c]:
                question = question.replace(str(v), str(c))
        l_features = self.data['features']
        l_classes = self.data['classes']
        # extract instance from question
        l_instances = [[], []]
        self.l_instances = l_instances
        if len(l_features) > 0:
            query = ""
            for f in l_features:
                query += ", "
                query += f
                query += " [a-z]*\d+.\d*"
            query = query[2:]
            regex_search_term = re.compile(query)
            question = re.sub(regex_search_term, "<instance>", question)
            self.l_exist_features = []
            for f in l_features:
                regex_search_term = re.compile(f)
                result = regex_search_term.findall(question)
                if len(result) == 1:
                    self.l_exist_features.append(f)
                regex_replacement = "<feature>"
                question = re.sub(regex_search_term, regex_replacement, question)
                if f not in self.data['info']["num_features"]:
                    for v in self.data['feature_values'][f]:
                        if v == None:
                            v = "None"
                        regex_search_term = re.compile(str(v))
                        result = regex_search_term.findall(question)
                        if len(result) == 1:
                            self.l_exist_values[f] = v


        self.l_exist_classes = []

        for c in l_classes:
            c = str(c)
            regex_search_term = re.compile(c)
            result = regex_search_term.findall(question)
            if len(result) == 1:
                self.l_exist_classes.append(c)
            regex_replacement = "<class>"
            question = re.sub(c, regex_replacement, question)
        return question

    def request_instance(self):
        print("Hey, I am here agent 125")
        if st.session_state.mode == MODE_INPUT:
            if self.dataset == "mnist":
                msg = "Give me the image's name, in the folder, I already have an example with 7.png"
                # print_log("xagent", msg)
                user_input = state_log("user")
                image = PIL.Image.open(user_input)
                # display(image)
                image_array = np.array(image)
                msg = "This is your input image."
                # print_log("xagent", msg)
                plt.figure(figsize=(2, 2))
                plt.imshow(image_array[:, :, 0])
                plt.show()
                self.current_instance = image_array[:, :, 0].astype('float32') / 255
                self.predicted_class = self.clf.predict(self.current_instance.reshape(1, 28, 28, 1)).argmax()
                st.session_state.mode = MODE_QUESTION
                yield "My prediction for this image is number " + str(self.predicted_class) + "."
            for f, fn in zip(self.data["features"], self.data["feature_names"]):
                self.current_feature = f
                # feature_description = self.data['info']['feature_description'][fn] if fn in  self.data['info']['feature_description'] else ""
                msg = random.choice(l_questions) + str(fn)
                if fn in self.data['info']['feature_description']:
                    msg += f"({self.data['info']['feature_description'][fn]})"
                if f in self.data['map'].keys():
                    if self.dataset == "german-credit" and f == "Job":
                        yield (
                                    msg + "? Please choose one of the following numbers:" + " [0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled]")
                    else:
                        features = ",".join(str(x) for x in self.data['map'][f].values())
                        yield (msg + f"? Please choose one of the following values: [{features}]")
                else:
                    yield (msg + "? Please give me a number")
                    # self.current_instance[f] = float(text)
            st.session_state.mode = MODE_QUESTION
            self.current_feature = None
            string_convert = [str(v) for v in self.current_instance.values]
            instance = ",".join(string_convert)
            display_instance = {}
            for k, v in self.current_instance.items():
                display_instance[k] = [v]
            self.df_display_instance = pandas.DataFrame(display_instance)
            predict = self.clf_display.predict(self.df_display_instance)[0]
            self.predicted_class = predict
            ans = self.data['info']['predict_prompt'][predict] + " \n " + question_msg
            yield "I recorded the information: [" + instance + "]. " + ans

    def collect_instance(self, text):
        f = self.current_feature

        if f is None:
            return None
        # categorical features
        if f in self.data['map'].keys():
            features = ",".join(str(x) for x in self.data['map'][f].values())
            # print(self.data['map'][f].values())
            # print(text)
            while (text not in self.data['map'][f].values()):
                msg = constraints.repeat_cat_features.format(features)
                yield msg
                # print_log("xagent", msg)
                # text = print_log("user")
            if self.data['info']['name'] == 'adult':
                self.current_instance[f] = str(" " + str(text))
            else:
                self.current_instance[f] = str(text)
        else:
            # numerical feature
            while not text.isnumeric():
                yield "Please input a number instead of letters"
            self.current_instance[f] = float(text)
        yield None

    def response(self, text, conversations=[]):
        if "[reset]" in text:
            self.mode = None
            self.dataset = None
            self.current_instance = None
            ans = constraints.welcome_msg_test
            # logging.info(ans)
            return ans
        if "/change instance" in text:
            st.session_state.mode = MODE_INPUT
            self.request_iterator = self.request_instance()
            ans = f"Welcome to {self.dataset} dataset. {self.data['info']['dataset_description']}. I will ask you some questions to collect your information. Are you ready to input information?"
        if st.session_state.mode == MODE_ASK_FOR_FEATURE:
            self.l_exist_feature = []
            self.preprocess_question(text)
            if len(self.l_exist_features) == 0:
                msg = f"please choose one or more features from this list of features: {st.session_state.feature}"
                # print_log("xagent", msg)
                return msg
            st.session_state.mode = MODE_QUESTION
        if st.session_state.mode is None:
            # if text not in ["iris", "adult", "titanic", "german-credit", "yes"]:
            #     return constraints.dataset_error_msg
            # else:
            #     if text != "yes":
            #         self.dataset = text

            self.get_dataset_info(self.dataset)
            self.train_model()
            # for i
            # self.data['X']
            self.current_instance = self.data['X'].iloc[1]
            self.df_display_instance = pandas.DataFrame(self.current_instance).T
            predict = self.clf_display.predict(self.df_display_instance)[0]
            self.predicted_class = predict
            # self.current_instance = self.current_instance.to_dict()
            ans = "The dataset has the following attributes: \n "
            for feature in self.current_instance.keys():
                ans += f"| **{feature}**"
                if feature in self.data['info']['feature_description']:
                    ans += f" ({self.data['info']['feature_description'][feature]})"
            ans += " \n \n We prepared the following sample instance of a customer: \n "
            for feature in self.current_instance.keys():
                ans += f"|{feature}  "
            ans += " \n "
            for feature in self.current_instance.keys():
                ans += "|--"
            ans += " \n "
            for feature in self.current_instance.keys():
                if feature == "Job":
                    ans += f" | {map_job[self.current_instance[feature]]} "
                else:
                    ans += f" | {self.current_instance[feature]}"



            ans += """ \n \n If you like to continue with this instance, please type anything. Otherwise, if you like to input a different instance, please type the **```/change instance```** . You can change the instance anytime by the command."""
            st.session_state.use_llm = False
            st.session_state.mode = MODE_CHOOSE_INSTANCE
            return ans
        elif st.session_state.mode == MODE_CHOOSE_INSTANCE:
            st.session_state.use_llm = False
            ans = f"The prediction for this instance is {self.data['info']['predict_prompt'][self.predicted_class]}. {question_msg}"
            st.session_state.mode = MODE_QUESTION
            return ans

            # st.session_state.exist_feature.append(user_input)
        else:
            st.session_state.use_llm = True
            if st.session_state.mode == MODE_INPUT:
                check_instance = self.collect_instance(text)
                answer = next(check_instance, None)
                if answer != None:
                    return answer
                answer = next(self.request_iterator, None)

            else:
                print("Hey agent 223")
                print(st.session_state.mode)
                print(st.session_state.question)
                self.original_question = text
                question = self.preprocess_question(text)
                if st.session_state.question is None:
                    if st.session_state.mode == MODE_QUESTION:
                        st.session_state.question = self.nlu_model.match(question)
                        print("match")
                        if st.session_state.question is None:
                            print("suggest")
                            st.session_state.mode = MODE_SUGGEST_QUESTION
                    if st.session_state.mode == MODE_SUGGEST_QUESTION:
                        print("suggest in")

                        result = self.nlu_model.suggest_questions(question, self.data["features"], self.data['info']['map_label'][self.predicted_class],
                                                                  self.current_instance,
                                                                  self.data['info']['map_label'].values())
                        if result is None and st.session_state.question is None:
                            result = self.nlu_model.suggest_questions(question, self.data["features"],
                                                                      self.data['info']['map_label'][
                                                                          self.predicted_class],
                                                                      self.current_instance,
                                                                      self.data['info']['map_label'].values())
                            return result
                        if st.session_state.question is None:  # question not yet decided
                            return result
                logging.log(26, f"question = {question}")
                answer = self.answer_question(st.session_state.question)
                print(st.session_state.mode)
                if st.session_state.mode not in [MODE_ASK_FOR_CLS, MODE_ASK_FOR_FEATURE]:
                    print("Did you come here")
                    st.session_state.question = None
                logging.log(26, f"answer = {answer}")
            # logging.info(answer)
            return answer

    def _is_not_blank(self, s):
        return bool(s and not s.isspace())

    def answer_question(self, question, conversations=[]):

        answer = Answers(self.list_node, self.clf, self.clf_display, self.current_instance, question,
                         self.l_exist_classes, self.l_exist_features, self.l_instances, self.l_exist_values, self.data,
                         self.df_display_instance, self.predicted_class, self.dataset_anchor, self.clf_anchor,
                         self.original_question, self.preprocessor)
        return answer.answer(question, conversations)

    def get_dataset_info(self, dataset_name: str):

        self.intro = ""
        with open(files('dataset_info').joinpath(dataset_name + ".json")) as f_in:
            self.data['info'] = json.load(f_in)
        if dataset_name.lower() == "german-credit":
            self.data["X_display"] = self.data["X"] = pd.read_csv('dataset/german-credit/german_credit_data.csv')
            self.data["y_display"] = self.data["y"] = self.data["X_display"]['Risk']
            self.data["X_display"].drop(['Risk'], axis=1, inplace=True)
            #todo: apply text preprocessing to both classes and questions
            self.data["classes"] = ["bad", "good"]
            self.data["cls_mapping"] = {}
            self.data["features"] = self.data["X_display"].columns.tolist()
            self.data["feature_names"] = self.data["features"]
            mapping = {}
            feature_value = {}
            for f in self.data["features"]:
                if f not in self.data['info']["num_features"]:
                    mapping[f] = {str(value): str(value) for value in self.data["X_display"][f].unique()}
                    feature_value[f] = self.data["X_display"][f].unique()
                else:
                    feature_value[f] = 0
            self.data["feature_values"] = feature_value
            self.data["map"] = mapping
            return
        if dataset_name.lower() == "mnist":
            (self.data["X"], self.data["y"]), _ = tf.keras.datasets.mnist.load_data()
            self.data["classes"] = list(range(0, 10))
            self.data["cls_mapping"] = {"prediction?": ["predicted?"]}
            self.data["features"] = []
            return
        if dataset_name.lower() == "adult":
            self.data["X"], self.data["y"] = shap.datasets.adult()
            self.data["X_display"], self.data["y_display"] = shap.datasets.adult(display=True)
            self.data["classes"] = ["False", "True"]
            self.data["cls_mapping"] = {"False": ["<=50K", "less than 50K"],
                                        "K ?": ["K\?"],
                                        "True": [">50K"],
                                        "have": ["get"],
                                        "instance": ["profile", "information"],
                                        "prediction": ["predicted"]}
            self.data["features"] = self.data["X"].columns.tolist()
            self.data["feature_names"] = self.data["features"]

        if dataset_name.lower() == "titanic":
            X_display = pd.read_csv('dataset/titanic/clean_train.csv')
            y_display = X_display['Survived']
            X_display.drop(['Survived'], axis=1, inplace=True)
            self.data["X_display"] = X_display
            self.data["y_display"] = y_display
            self.data["X"] = X_display.copy(deep=True)
            self.data["y"] = y_display.copy(deep=True)
            self.data["classes"] = ["False", "True"]
            self.data["cls_mapping"] = {"False": ["die", "to die"],
                                        "True": ["survive", "to survive"]}
            self.data["features"] = self.data["X_display"].columns.tolist()
            self.data["feature_names"] = ["Class (1st, 2nd, 3rd)", "Gender", "Age", "Fare",
                                          "Port of Embarkation(C = Cherbourg, Q = Queenstown, S = Southampton)",
                                          "Title", "Deck", "Family size"]
            for f in self.data['info']['cat_features']:
                self.data["X_display"][f] = self.data["X_display"][f].astype(
                    pandas.core.dtypes.dtypes.CategoricalDtype(categories=self.data["X_display"][f].unique(),
                                                               ordered=True))
            self.intro = " I will receive the information of a passenger, then predict whether the passenger will survive or not"

        if dataset_name.lower() == "iris":
            iris = sklearn.datasets.load_iris()
            X_display = iris.data
            y_display = iris.target
            features = ['sepal length', 'sepal width', 'petal length', 'petal width']
            self.data["X_display"] = pd.DataFrame(data=X_display, columns=features)
            self.data["y_display"] = pd.Series(y_display)
            self.data["X"] = self.data["X_display"].copy(deep=True)
            self.data["y"] = self.data["y_display"].copy(deep=True)
            self.data["classes"] = [0, 1, 2]
            print(self.data["classes"])
            self.data["cls_mapping"] = {}
            self.data["features"] = features
            self.data["feature_names"] = self.data["features"]
        mapping = {}
        for f in self.data["X_display"].columns:
            if type(self.data["X_display"][f].dtype) == pandas.core.dtypes.dtypes.CategoricalDtype:
                mapping[f] = dict(zip(self.data["X_display"][f].cat.codes, self.data["X_display"][f]))
                self.data["X"][f] = self.data["X_display"][f].cat.codes
        self.data['map'] = mapping

    def get_model(self, dataset_name, name_clf: str = "sklearn.tree._classes.DecisionTreeClassifier",
                  metric: str = "predictive_accuracy"):
        tasks_df = openml.tasks.list_tasks(
            task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe"
        )
        task_id = tasks_df[tasks_df.name == dataset_name].iloc[0]['tid']
        self.clf = sklearn.tree.DecisionTreeClassifier(max_depth=100)
        flow_ids = openml.flows.get_flow_id(name=name_clf)
        evals = openml.evaluations.list_evaluations(
            function=metric, tasks=[task_id], output_format="dataframe"
        )
        run_id = evals[evals['flow_id'].isin(flow_ids)]['run_id'].iloc[-1]
        run_downloaded = openml.runs.get_run(run_id)
        task = openml.tasks.get_task(task_id)
        setup_id = run_downloaded.setup_id

        # after this, we can easily reinstantiate the model
        model_duplicate = openml.setups.initialize_model(setup_id)
        # it will automatically have all the hyperparameters set

        # and run the task again
        self.train_model()

    def _get_relative(self, node_i, ancestor, n_nodes):
        for i in range(self.clf.tree_.node_count):
            if (node_i == self.clf.tree_.children_left[i]) or (node_i == self.clf.tree_.children_right[i]):
                ancestor.append(i)
                self._get_relative(i, ancestor, self.clf.tree_.node_count)

    def model(self):
        x_in = Input(shape=(28, 28, 1))
        x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x_in)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x_out = Dense(10, activation='softmax')(x)

        cnn = Model(inputs=x_in, outputs=x_out)
        cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return cnn

    def train_model(self):
        #         self.clf = sklearn.tree.DecisionTreeClassifier(max_depth=100)
        dataset_folder = 'dataset'
        if self.dataset == "german-credit":
            self.clf = pickle.load(open(os.path.join("models/german-credit", 'rf_german_credit.pkl'), "rb"))
            self.clf_display = self.clf
            self.preprocessor = pickle.load(open(os.path.join("models/german-credit", 'preprocessor.pkl'), "rb"))
            self.dataset_anchor = utils.load_dataset('german-credit', balance=True, dataset_folder=dataset_folder,
                                                     discretize=False)
            self.clf_anchor = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
            self.clf_anchor.fit(self.dataset_anchor.train, self.dataset_anchor.labels_train)
            return
        if self.dataset == "mnist":
            self.data['X'] = self.data['X'].astype('float32') / 255
            self.data['X'] = np.reshape(self.data['X'], self.data['X'].shape + (1,))
            self.data['y'] = to_categorical(self.data['y'])
            self.clf = self.model()
            self.clf = load_model(os.path.join(PATH, 'mnist_cnn.h5'))
            return
        if self.dataset == "adult":
            self.dataset_anchor = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder,
                                                     discretize=False)

            self.clf_anchor = sklearn.ensemble.RandomForestClassifier()
            self.clf_anchor.fit(self.dataset_anchor.train, self.dataset_anchor.labels_train)
        numerical = self.data['info']['num_features']
        categorical = self.data["X_display"].columns.difference(numerical)

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        transformations = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical)])

        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        self.clf_display = Pipeline(steps=[('preprocessor', transformations),
                                           ('classifier', RandomForestClassifier())])
        self.clf = RandomForestClassifier()
        #         self.clf_display = RandomForestClassifier()
        self.clf.fit(self.data["X"], self.data["y"])
        if len(categorical) > 0:
            self.clf_display.fit(self.data["X_display"], self.data["y_display"])
        else:
            self.clf_display = self.clf
