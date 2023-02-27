import pandas as pd
import os
import sys
# sys.path.append('/homes/bach/XAGENT/XAgent/Agent')
# silence command-line output temporarily
# sys.stdout, sys.stderr = os.devnull, os.devnull
from importlib_resources import files
from simcse import SimCSE
# unsilence command-line output

enabled = True
class NLU:
    def __init__(self):
        self.model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
        self.df = pd.read_csv(files("XAgent").joinpath('Median_4.csv'))
        self.model.build_index(list(self.df['Question']))
    def match(self, question):
        threshold = 0.5
        match_result = self.model.search(question)
        if len(match_result) > 0:
            match_question, score =  self.model.search(question)[0]
            if score < threshold:
                match_question = "unknown"
        else:
            match_question = "unknown"
            score = 0
        return match_question, score
    

# class NLU:
#     def __init__(self):
#         pass
#     def get_dataset_from_openml(dataset_name:str)-> tuple:
#         return openml.datasets.get_dataset(dataset_name,output_format="dataframe", target=dataset.default_target_attribute)
#     def get_model_run(task_id: int = 59, name_clf: str = "sklearn.tree._classes.DecisionTreeClassifier", metric: str= "predictive_accuracy"):
#         clf = sklearn.tree.DecisionTreeClassifier()
#         flow_ids = openml.flows.get_flow_id(name=name_clf)
#         evals = openml.evaluations.list_evaluations(
#             function=metric, tasks=[task_id], output_format="dataframe"
#         )
#         run_id = evals[evals['flow_id'].isin(flow_ids)]['run_id'].iloc[-1]
#         run_downloaded = openml.runs.get_run(run_id)
#         task = openml.tasks.get_task(task_id)
#         setup_id = run_downloaded.setup_id

#         # after this, we can easily reinstantiate the model
#         model_duplicate = openml.setups.initialize_model(setup_id)
#         # it will automatically have all the hyperparameters set
#         return model_duplicate