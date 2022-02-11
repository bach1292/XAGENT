import openml

import json
import random
import re
import os
import openml
import pandas as pd
import shap
import sklearn
# from dtreeviz.trees import *
from openml.tasks import TaskType
import numpy as np
# print the JS visualization code to the notebook
# shap.initjs()
import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from Agent.answer import Answers
from Agent import utils
from anchor import anchor_tabular

# import the desired library
from Agent.nlu import NLU

# import sys
# sys.stdout = sys.__stdout__
# sys.stderr = sys.__stderr__
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import PIL
# from IPython.display import display
MODE_INPUT = 0
MODE_QUESTION = 1
# MODE_DATASET = 1
PATH = os.path.dirname(__file__)
class Agent:
    def __init__(self):
        self.dataset = None
        self.current_instance = None
        self.clf = None
        self.predicted_class = None
        self.mode = None
        self.data = {"X": None, "y": None, "features": None, "classes": None}
        self.nlu_model = NLU()
        self.list_node = []
        self.clf_display = None
        self.l_exist_classes = None
        self.l_exist_features = None
        self.l_instances = None
        self.df_display_instance = None
        self.current_feature = None
        self.dataset_anchor = None
        self.clf_anchor = None

    def preprocess_question(self, question):
        for c in self.data['cls_mapping']:
            for v in self.data['cls_mapping'][c]:
                question = question.replace(str(v),str(c))
        l_features = self.data['features']
        l_classes = self.data['classes']
        # extract instance from question
        l_instances = [[], []]
        self.l_instances = l_instances
        #         print("self.l_instances", self.l_instances)
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
                # print(f)
                regex_search_term = re.compile(f)
                result = regex_search_term.findall(question)
                if len(result) == 1:
                    self.l_exist_features.append(f)
                regex_replacement = "<feature>"
                question = re.sub(regex_search_term, regex_replacement, question)
        self.l_exist_classes = []

        for c in l_classes:
            c = str(c)
            regex_search_term = re.compile(c)
            result = regex_search_term.findall(question)
            if len(result) == 1:
                self.l_exist_classes.append(c)
            regex_replacement = "<class>"
            question = re.sub(c, regex_replacement, question)
        # print(question)
        return question
    def request_instance(self):
        if self.mode == MODE_INPUT:
            if self.dataset == "mnist":
                print("Give me the image's name, in the folder, I already have an example with 7.png")
                user_input = input()
                image = PIL.Image.open(user_input)
                # display(image)
                image_array = np.array(image)
                print('\033[1m\033[94m' + 'X-Agent'+': ' + '\033[0m' + "This is your input image.")
                plt.figure(figsize=(2, 2))
                plt.imshow(image_array[:,:,0])
                plt.show()
                self.current_instance = image_array[:,:,0].astype('float32') / 255
                self.predicted_class = self.clf.predict(self.current_instance .reshape(1, 28, 28, 1)).argmax()
                self.mode = MODE_QUESTION
                yield "My prediction for this image is number " + str(self.predicted_class) + "."
            l_questions = ["How about your ", "What is your ", "Give me your "]
            for f, fn in zip(self.data["features"], self.data["feature_names"]):
                self.current_feature = f
                if f in self.data['map'].keys():
                    features = ",".join(str(x) for x in self.data['map'][f].values())
                    yield (random.choice(l_questions) + str(
                        fn) + "? Please choose one of the following values: [" + features + "]")
                else:
                    yield (random.choice(l_questions) + str(fn) + "? Please give me a number")
                    # self.current_instance[f] = float(text)
            self.mode = MODE_QUESTION
            self.current_feature = None
            string_convert = [str(v) for v in self.current_instance.values()]
            instance = ",".join(string_convert)
            display_instance = {}
            for k, v in self.current_instance.items():
                display_instance[k] = [v]
            self.df_display_instance = pandas.DataFrame(display_instance)
            predict = self.clf_display.predict(self.df_display_instance)[0]
            self.predicted_class = predict
            # if predict == False:
            ans = self.data['info']['predict_prompt'][predict]
            # else:
            #     ans = self.data['info']['predict_prompt'][1]
            yield "I recorded the information: [" + instance + "] " + ans
            # yield "instance confirm"
            # return #self.input_collection(text)
    def collect_instance(self, text):
        f = self.current_feature
        print(f)
        if f is None:
            return
        if f in self.data['map'].keys():
            if self.data['info']['name'] == 'adult':
                self.current_instance[f] = str(" " + str(text))
            else:
                self.current_instance[f] = str(text)
        else:
            self.current_instance[f] = float(text)
    def dataset_response(self, text):
        if "dataset" in text:
            self.mode = None
            self.dataset = None
            self.current_instance = None
            return "Please choose a dataset. We only support iris, adult, titanic and mnist at the moment ^^. Please type correctly only one of dataset names that I just listed"
        if self.mode is None:
            if text not in ["iris","adult","titanic","mnist"]:
                return "Please choose one of the datasets: iris, adult, titanic, mnist. Please type correctly only one of dataset names that I just listed"
            else:
                self.dataset = text
                self.mode = MODE_INPUT
                self.get_dataset_info(text)
                print("Wait a min, I need to learn it")
                self.train_model()
                self.current_instance = {}
                self.request_iterator = self.request_instance()
                return "Welcome to " + text + " dataset, are you ready to input the instance?" + self.intro
        else:
            if self.mode == 0:
                self.collect_instance(text)
                answer = next(self.request_iterator, None)
            else:
                question = self.preprocess_question(text)
                #             print(question)
                # print(question)
                question, _ = self.nlu_model.match(question)
                # print(question)
                # todo
                answer = self.answer_question(question)
            return answer

    def _is_not_blank(self, s):
        return bool(s and not s.isspace())

    def answer_question(self, question):

        answer = Answers(self.list_node, self.clf, self.clf_display, self.current_instance, question,
                         self.l_exist_classes, self.l_exist_features, self.l_instances, self.data,
                         self.df_display_instance, self.predicted_class, self.dataset_anchor, self.clf_anchor)
        return answer.answer(question)

    def get_dataset_info(self, dataset_name: str):
        self.intro = ""
        with open("dataset_info/" + dataset_name + ".json") as f_in:
            self.data['info'] = json.load(f_in)
        if dataset_name.lower() == "mnist":
            (self.data["X"], self.data["y"]),_ = tf.keras.datasets.mnist.load_data()
            self.data["classes"] = list(range(0,10))
            self.data["cls_mapping"] = {"prediction?": ["predicted?"]}
            self.data["features"] = []
            return
        if dataset_name.lower() == "adult":
            self.data["X"], self.data["y"] = shap.datasets.adult()
            self.data["X_display"], self.data["y_display"] = shap.datasets.adult(display=True)
            self.data["classes"] = ["False","True"]
            self.data["cls_mapping"] = {"False": ["<=50K", "less than 50K"],
                                        "K ?": ["K\?"],
                                        "True": [">50K"],
                                        "have": ["get"],
                                        "instance":["profile","information"],
                                        "prediction":["predicted"]}
            self.data["features"] = self.data["X"].columns.tolist()
            self.data["feature_names"] = self.data["features"]

        if dataset_name.lower() == "titanic":
            X_display = pd.read_csv('dataset/titanic/clean_train.csv')
            y_display = X_display['Survived']
            X_display.drop(['Survived'], axis=1, inplace=True)
            self.data["X_display"] = X_display
            self.data["y_display"] = y_display
            # print(self.data["y_display"].shape())
            self.data["X"] = X_display.copy(deep=True)
            self.data["y"] = y_display.copy(deep=True)
            self.data["classes"] = ["False","True"]
            self.data["cls_mapping"] = {"False": ["die", "to die"],
                                        "True": ["survive","to survive"]}
            self.data["features"] = self.data["X_display"].columns.tolist()
            self.data["feature_names"] = ["Class (1st, 2nd, 3rd)","Gender","Age","Fare","Port of Embarkation(C = Cherbourg, Q = Queenstown, S = Southampton)","Title","Deck","Family size"]
            for f in self.data['info']['cat_features']:
                # print(f)
                self.data["X_display"][f] = self.data["X_display"][f].astype(
                    pandas.core.dtypes.dtypes.CategoricalDtype(categories=self.data["X_display"][f].unique(),
                                                               ordered=True))
            self.intro = " I will receive the information of a passenger, then predict whether the passenger will survive or not"

        if dataset_name.lower() == "iris":
            iris = sklearn.datasets.load_iris()
            X_display = iris.data
            y_display = iris.target
            features = ['sepal length', 'sepal width', 'petal length', 'petal width']
            self.data["X_display"] = pd.DataFrame(data = X_display, columns=features)
            self.data["y_display"] = pd.Series(y_display)
            # print(self.data["y_display"].shape())
            self.data["X"] = self.data["X_display"].copy(deep=True)
            self.data["y"] = self.data["y_display"].copy(deep=True)
            self.data["classes"] = iris.target_names
            self.data["cls_mapping"] = {}
            self.data["features"] = features
            self.data["feature_names"] = self.data["features"]
        mapping = {}
        for f in self.data["X_display"].columns:
            if type(self.data["X_display"][f].dtype) == pandas.core.dtypes.dtypes.CategoricalDtype:
                mapping[f] = dict(zip(self.data["X_display"][f].cat.codes, self.data["X_display"][f]))
                self.data["X"][f] = self.data["X_display"][f].cat.codes
        self.data['map'] = mapping
        # print(mapping)


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
        #         print(evals)
        #         print(flow_ids)
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
        if self.dataset == "mnist":
            self.data['X'] = self.data['X'].astype('float32') /255
            self.data['X'] = np.reshape(self.data['X'] , self.data['X'] .shape + (1,))
            self.data['y'] = to_categorical(self.data['y'])
            self.clf = self.model()
            self.clf = load_model(os.path.join(PATH,'mnist_cnn.h5'))
            return
        if self.dataset == "adult":
            dataset_folder = 'dataset'
            self.dataset_anchor = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder, discretize=False)

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


