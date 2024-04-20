import shap
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import dice_ml
from XAgent.Agent import constraints
from alibi.explainers import CounterfactualProto
from tensorflow.keras.models import Model, load_model
import streamlit as st
from anchor import anchor_tabular

from XAgent.Agent.mode import MODE_QUESTION
from XAgent.Agent.utils import state_log
from XAgent.Agent.utils import ask_for_feature

PATH = os.path.dirname(__file__)
sys.path.append(PATH)
def shap_explainer(self, id_question):
    if self.data['info']['name'] == 'mnist':
        background = self.data["X"][np.random.choice(self.data["X"].shape[0], 100, replace=False)]
        e = shap.DeepExplainer(self.clf, background)
        # ...or pass tensors directly
        shap_values = e.shap_values(self.current_instance.reshape(1, 28, 28, 1))
        shap.image_plot(shap_values[self.predicted_class], -self.current_instance.reshape(1, 28, 28, 1), width=5)
        plt.show()
    elif self.data['info']['name'] == 'german-credit':
        X_transform = self.clf['preprocessor'].transform(self.df_display_instance)
        background = shap.maskers.Independent(self.clf['preprocessor'].transform(self.data['X_display']),
                                              max_samples=100)
        explainer = shap.Explainer(self.clf['classifier'], masker=background, algorithm="tree")
        shap_values = explainer.shap_values(X_transform)
        # shap_values = explainer_values.values
        predicted_cls = 1-self.data["classes"].index(self.predicted_class)
        shap_values_original_input = []
        X = self.data['X_display']
        count_values = [
            len(X[column].unique()) if column in ['Sex', 'Housing', 'Saving accounts', 'Checking account',
                                                  'Purpose'] else 1 for column in X]
        for shap_array in shap_values:
            sub_array = []
            start = 0
            for count_value in count_values:
                sub_array.append(sum(shap_array[0][start:(start + count_value)]))
                start += count_value
            shap_values_original_input.append(np.array(sub_array))
        explainer_values = explainer(X_transform)
        explainer_values.values = shap_values_original_input[predicted_cls].reshape(1,-1)
        # explainer_values.values = shap_values_original_input
        if id_question in constraints.l_feature_questions_ids:
            # ask_for_feature(self)
            if len(self.l_exist_features) == 0:
                return ask_for_feature(self)
            index = [ self.df_display_instance.columns.get_loc(self.l_exist_features[0])]
            # shap.force_plot(explainer.expected_value[predicted_cls][index], shap_values[predicted_cls][index], self.df_display_instance.columns[index],figsize=(15,3), show = True, matplotlib=True)
            # shap.plots.waterfall(explainer_values[0][index])
            shap.force_plot(explainer.expected_value[predicted_cls], shap_values_original_input[predicted_cls][index],
                            self.df_display_instance.columns[index], figsize=(15, 3), show=False, matplotlib=True)
            st.session_state.mode = MODE_QUESTION
        else:
            shap.force_plot(explainer.expected_value[predicted_cls], shap_values_original_input[predicted_cls],
                            self.df_display_instance.columns, figsize=(15, 3), show=False, matplotlib=True)
    else:
        explainer = shap.Explainer(self.clf)
        num_instance = []
        for f in self.df_display_instance.columns:
            if f in self.data['map'].keys():
                for k, v in self.data['map'][f].items():
                    if str(v) == str(self.df_display_instance[f][0]):
                        num_instance.append(k)
            else:
                num_instance.append(self.df_display_instance[f][0])
        predicted_cls = self.data["classes"].index(self.predicted_class)
        shap_values = explainer.shap_values(np.array(num_instance))
        shap.force_plot(explainer.expected_value[predicted_cls], shap_values[predicted_cls],
                        self.df_display_instance.columns, figsize=(15, 3), show=False, matplotlib=True)
    # msg = 'temp.png'
    temp_dir = "static"
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)

    filename = f"{st.session_state.dt_string}.jpg"
    file_path = os.path.join(temp_dir, filename)
    plt.savefig(file_path, bbox_inches='tight')
    # print_log("xagent", f'<img height="100%" width="100%" src="/app/{filename}"/>')
    return file_path


def dice_answer(self, target_class=0, features='all'):
    target = 0
    for i, c in enumerate(self.l_classes):
        if target_class == str(c):
            target = i
    train_dataset = self.data['X_display'].assign(Label=self.data['y_display'])
    # print(train_dataset.head())
    # print(self.df_display_instance)
    d = dice_ml.Data(dataframe=train_dataset, continuous_features=self.data['info']['num_features'],
                     outcome_name='Label')
    # Using sklearn backend
    m = dice_ml.Model(model=self.clf_display, backend="sklearn")
    # Using method=random for generating CFs
    exp = dice_ml.Dice(d, m, method="random")
    e1 = exp.generate_counterfactuals(self.df_display_instance, total_CFs=1, desired_class=target,
                                      features_to_vary=features)
    return e1
def cf_proto(self, class_P):
    x_train = self.data["X"]
    shape = (1,) + self.data["X"].shape[1:]
    gamma = 100.
    theta = 100.
    c_init = 1.
    c_steps = 2
    max_iterations = 1000
    feature_range = (x_train.min(), x_train.max())
    ae = load_model(os.path.join(PATH, 'mnist_ae.h5'))
    enc = load_model(os.path.join(PATH, 'mnist_enc.h5'), compile=False)
    cf = CounterfactualProto(self.clf, shape, gamma=gamma, theta=theta,
                             ae_model=ae, enc_model=enc, max_iterations=max_iterations,
                             feature_range=feature_range, c_init=c_init, c_steps=c_steps)
    cf.fit(x_train)
    X = self.current_instance.reshape((1,) + self.data["X"][1].shape)
    plt.figure(figsize=(2, 2))
    plt.imshow(X.reshape(28, 28));
    explanation_2 = cf.explain(X, k=5, k_type='mean', target_class=[int(class_P)])
    if explanation_2.cf is None:
        return "It is hard to change this instance to " + class_P
    proto_2 = explanation_2.id_proto
    plt.imshow(explanation_2.cf['X'].reshape(28, 28));
    plt.show()
    return "Here you go! I just modified your image a bit to make it look like number " + str(class_P)
def anchor_answer(self):
    if self.data['info']['name'] in ['adult', "german-credit"]:
        explainer = anchor_tabular.AnchorTabularExplainer(
            self.dataset_anchor.class_names,
            self.dataset_anchor.feature_names,
            self.dataset_anchor.train,
            self.dataset_anchor.categorical_names)
        instance = list(self.current_instance.values)
        for feature in self.dataset_anchor.categorical_features:
            # print(self.dataset_anchor.categorical_names[feature])
            if instance[feature] == "nan":
                instance[feature] = ""
            if feature == 2:
                instance[feature] = self.dataset_anchor.categorical_names[feature].index(
                    str(int(instance[feature])))
            else:
                instance[feature] = self.dataset_anchor.categorical_names[feature].index(instance[feature])

        exp = explainer.explain_instance(np.array(instance), self.clf_anchor.predict, threshold=0.80)
        # dice_answer(self.predicted_class,self.data['info']['num_features'])
        return 'If you keep these conditions: %s, the prediction will stay the same.' % (' AND '.join(exp.names()))
    return "Sorry, I don't support this question for this dataset, let try another data"