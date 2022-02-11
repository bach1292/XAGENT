import copy
import json

import shap
from dtreeviz.trees import *

# print the JS visualization code to the notebook
shap.initjs()

import dice_ml

import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
from alibi.explainers import CounterfactualProto
from skimage.color import gray2rgb, rgb2gray, label2rgb # since the code wants color images
from anchor import utils
from anchor import anchor_tabular
import os
PATH = os.path.dirname(__file__)
# print('TF version: ', tf.__version__)
# print('Eager execution enabled: ', tf.executing_eagerly()) # False
class Answers:
    def __init__(self, list_node, clf, clf_display, current_instance, question, l_exist_classes, l_exist_features,
                 l_instances, data, df_display_instance, predicted_class, dataset_anchor, clf_anchor):
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

    def extract_relation(self, test_data, clf_list, feature_name):
        relation = []
        for i, clf in enumerate(clf_list):
            relation.append({})
            for j, (v1, v2) in enumerate(zip(test_data, clf[:-1])):
                if v1 != v2:
                    if type(v1) == str:
                        relation[i][feature_name[j]] = "not suitable"
                    else:
                        if v1 < v2:
                            relation[i][feature_name[j]] = "too low"
                        else:
                            relation[i][feature_name[j]] = "too high"
        return relation

    def answer(self, question):
        # print(question)
        def dice_answer(target_class = 0, features = 'all'):
            target = 0
            # print(target_class)
            for i, c in enumerate(self.l_classes):
                if target_class == str(c):
                    target = i
            train_dataset = self.data['X_display'].assign(Label=self.data['y_display'])
            d = dice_ml.Data(dataframe=train_dataset, continuous_features=self.data['info']['num_features'],
                             outcome_name='Label')
            # Using sklearn backend
            m = dice_ml.Model(model=self.clf_display, backend="sklearn")
            # Using method=random for generating CFs
            exp = dice_ml.Dice(d, m, method="random")
            e1 = exp.generate_counterfactuals(self.df_display_instance, total_CFs=1, desired_class=target,
                                              features_to_vary=features)
            # print(e1.to_json())
            # e1.visualize_as_list(show_only_changes=True)
            return e1
        # print(question)
        if question == "unknown":
            return "Can you rephrase your question?"
        df = pd.read_csv(os.path.join(PATH,"Median_4.csv"))
        id_question = df[df['Question'] == question]["Label"].iloc[0]
        if id_question == \
                df[df['Question'] == "How should this Instance change to get a different prediction?"]["Label"].iloc[0]:
            class_Q = self.predicted_class
            # print(self.predicted_class)
            # print(self.l_exist_classes)
            class_P = None
            for c in self.l_exist_classes:
                if c != self.predicted_class:
                    class_P = c
            if self.data['info']['name'] == "mnist":
                x_train = self.data["X"]
                shape = (1,) + self.data["X"].shape[1:]
                gamma = 100.
                theta = 100.
                c_init = 1.
                c_steps = 2
                max_iterations = 1000
                feature_range = (x_train.min(), x_train.max())
                ae = load_model(os.path.join(PATH,'mnist_ae.h5'))
                enc = load_model(os.path.join(PATH,'mnist_enc.h5'), compile=False)
                cf = CounterfactualProto(self.clf, shape, gamma=gamma, theta=theta,
                                         ae_model=ae, enc_model=enc, max_iterations=max_iterations,
                                         feature_range=feature_range, c_init=c_init, c_steps=c_steps)
                cf.fit(x_train)
                X = self.current_instance.reshape((1,) + self.data['X'][1].shape)
                plt.figure(figsize=(2, 2))
                plt.imshow(X.reshape(28, 28));
                # print(type(class_P))
                explanation_2 = cf.explain(X, k=5, k_type='mean', target_class=[int(class_P)])
                if explanation_2.cf is None:
                    return "It is hard to change this instance to " + class_P
                # print(explanation_2.id_proto)
                proto_2 = explanation_2.id_proto
                plt.imshow(explanation_2.cf['X'].reshape(28, 28));
                plt.show()
                return "Here you go! I just modified your image a bit to make it look like number " + str(class_P)

            e1 = dice_answer(class_P)
            json_e1 = e1.to_json()
            js = json.loads(json_e1)
            ans_relation = []
            for j, (v1, v2) in enumerate(zip(js['test_data'][0][0], js['cfs_list'][0][0][:-1])):
                if v1 != v2:
                    if type(v1) == str:
                        s = js['feature_names'][j] + " changes to" + str(v2)
                        # print("you need to change " + js['feature_names'][j])
                    else:
                        if v1 < v2:
                            s = js['feature_names'][j] + " increases to " + str(v2)
                            # print("you need to increase " + js['feature_names'][j])
                        else:
                            s = js['feature_names'][j] + " decreases to " + str(v2)
                    ans_relation.append(s)
            return "Sounds easy! If " + " and ".join(ans_relation) + ", " + self.data['info']['change_ans'][self.l_classes.index(class_P)]

        if id_question == \
                df[df['Question'] == "How should this feature change for this instance to get a different prediction?"][
                    "Label"].iloc[0]:
            #             print(self.l_exist_features)
            if len(self.l_exist_features) == 0:
                print("which features?")
                user_input = input()
            else:
                class_Q = self.predicted_class
                class_P = None
                for c in self.l_exist_classes:
                    if c != self.predicted_class:
                        class_P = c
                # print(self.l_exist_features)
                e1 = dice_answer(class_P, self.l_exist_features)
                json_e1 = e1.to_json()
                js = json.loads(json_e1)
                ans_relation = []
                for j, (v1, v2) in enumerate(zip(js['test_data'][0][0], js['cfs_list'][0][0][:-1])):
                    if v1 != v2:
                        if type(v1) == str:
                            s = js['feature_names'][j] + " should be changed to " + str(v2)
                            # print("you need to change " + js['feature_names'][j])
                        else:
                            if v1 < v2:
                                s = js['feature_names'][j] + " should be increased to " + str(v2)
                                # print("you need to increase " + js['feature_names'][j])
                            else:
                                s = js['feature_names'][j] + " should be decreased to " + str(v2)
                        ans_relation.append(s)
                print("The " +  " and ".join(ans_relation) + ", to get " + self.data['info']['change_ans'][self.l_classes.index(class_P)] )
                return ""
        if id_question == df[df['Question'] == "What would happen if this instance changes to A?"]["Label"].iloc[0]:
            #             print(3)
            temp_instance = None
            for i in range(0, len(self.l_exist_features)):
                f = self.l_exist_features[i]
                index_feature = self.l_features.index(f)
                temp_instance = copy.copy(self.current_instance)
                temp_instance[index_feature] = self.l_instances[0][i]
            return "new instance is: " + str(temp_instance) + " and the predicted class is" + str(
                self.clf.predict([temp_instance]))
        if id_question == df[df['Question'] == "Why is this instance given this prediction?"]["Label"].iloc[0] or id_question == df[df['Question'] == "Give me the reason for this prediction."]["Label"].iloc[0]:

            # print(self.df_display_instance.columns)
            if self.data['info']['name'] == 'mnist':
                from sklearn.pipeline import Pipeline
                class PipeStep(object):
                    """
                    Wrapper for turning functions into pipeline transforms (no-fitting)
                    """

                    def __init__(self, step_func):
                        self._step_func = step_func

                    def fit(self, *args):
                        return self

                    def transform(self, X):
                        return self._step_func(X)
                makegray_step = PipeStep(lambda img_list: np.array([rgb2gray(img).reshape(28,28,1) for img in img_list]))
                flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

                simple_rf_pipeline = Pipeline([
                    ('Make Gray', makegray_step),
                    ('RF', self.clf)
                ])
                background = self.data['X'][np.random.choice(self.data['X'].shape[0], 100, replace=False)]
                img = gray2rgb(self.current_instance.astype(np.uint8))
                e = shap.DeepExplainer(self.clf, background)
                # ...or pass tensors directly
                # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
                # print(self.current_instance.shape)
                shap_values = e.shap_values(self.current_instance.reshape(1, 28, 28, 1))
                # print(len(shap_values))
                # print(shap_values[0].shape)
                shap.image_plot(shap_values[self.predicted_class], -self.current_instance.reshape(1, 28, 28, 1), width=5)
                plt.show()
                return self.data['info']['why_ans']
            explainer = shap.TreeExplainer(self.clf)
            num_instance = []
            for f in self.df_display_instance.columns:
                # print(f)
                if f in self.data['map'].keys():
                    for k, v in self.data['map'][f].items():
                        if str(v) == str(self.df_display_instance[f][0]):
                            # print(v)
                            num_instance.append(k)
                else:
                    num_instance.append(self.df_display_instance[f][0])
            # print(num_instance)
            shap_values = explainer.shap_values(np.array(num_instance))
            shap.force_plot(explainer.expected_value[1], shap_values[1], self.df_display_instance,figsize=(15,3), show = True, matplotlib=True)

#             plt.savefig('static/temp.svg')
            return self.data['info']['why_ans']
        if id_question == \
                df[df['Question'] == "What is the scope of change permitted to still get the same prediction?"][
                    "Label"].iloc[0]:
            if self.data['info']['name'] == 'adult':
                explainer = anchor_tabular.AnchorTabularExplainer(
                    self.dataset_anchor.class_names,
                    self.dataset_anchor.feature_names,
                    self.dataset_anchor.train,
                    self.dataset_anchor.categorical_names)
                instance = list(self.current_instance.values())
                # print(instance)
                for feature in self.dataset_anchor.categorical_features:
                    instance[feature] = self.dataset_anchor.categorical_names[feature].index(instance[feature][1:])

                # print('Prediction: ', explainer.class_names[self.clf_anchor.predict(np.array(instance).reshape(1, -1))[0]])
                exp = explainer.explain_instance(np.array(instance), self.clf_anchor.predict, threshold=0.80)
                print('If you keep these conditions: %s, the prediction will stay the same.' % (' AND '.join(exp.names())))
                # dice_answer(self.predicted_class,self.data['info']['num_features'])
                return ""
            return "Sorry, I don't support this question for this dataset, let try adult data"
        if id_question == df[df['Question'] == "How is this instance not predicted A?"]["Label"].iloc[0]:
            class_Q = self.predicted_class
            class_P = None
            for c in self.l_exist_classes:
                if c != self.predicted_class:
                    class_P = c
            e1 = dice_answer(class_P)
            json_e1 = e1.to_json()
            js = json.loads(json_e1)
            relation = self.extract_relation(js['test_data'][0][0], js['cfs_list'][0], js['feature_names'])
            print("There are multiple reasons for this result, one of them is:")
            print(" and ".join([str(k) + " is " + str(v) for k, v in relation[0].items()])+".")
            # e1.visualize_as_dataframe(show_only_changes=True)
            # print("In the table, I show the way to modify your features to change the class to " + str(
            #     class_P))
            ans_relation = []
            for j, (v1, v2) in enumerate(zip(js['test_data'][0][0], js['cfs_list'][0][0][:-1])):
                if v1 != v2:
                    if type(v1) == str:
                        s = js['feature_names'][j] + " changes to" + str(v2)
                        # print("you need to change " + js['feature_names'][j])
                    else:
                        if v1 < v2:
                            s = js['feature_names'][j] + " increases to at least " + str(v2)
                            # print("you need to increase " + js['feature_names'][j])
                        else:
                            s = js['feature_names'][j] + " decreases to at most " + str(v2)
                    ans_relation.append(s)

            # print(self.l_classes)
            return "If " + " and ".join(
                ans_relation) + ", to get " + self.data['info']['change_ans'][self.l_classes.index(class_P)]
        return "I cannot understand this question, can you rephrase the question or ask another question?"
