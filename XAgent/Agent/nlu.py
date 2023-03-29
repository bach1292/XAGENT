import pandas as pd
import os
import sys
import  logging
# sys.path.append('/homes/bach/XAGENT/XAgent/Agent')
# silence command-line output temporarily
# sys.stdout, sys.stderr = os.devnull, os.devnull
from importlib_resources import files
from simcse import SimCSE
# unsilence command-line output

enabled = True
topk = 45
class NLU:
    def __init__(self):
        self.model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
        self.df = pd.read_csv(files("XAgent").joinpath('Median_4.csv'), index_col=0).drop_duplicates()
        self.model.build_index(list(self.df['Question']))
        print(len(self.df))

    def get_list_questions(self, match_results):
        labels = []
        questions = []
        for question,_ in match_results:
            label = self.df.query('Question == @question')['Label'].iloc[0]
            if label not in labels:
                questions.append(question)
                labels.append(label)
        return questions


    def match(self, question, conversation = []):
        threshold = 0.6
        match_results = self.model.search(question, threshold=threshold)
        # print("result", match_results)
        if len(match_results) > 0:
            match_question, score = match_results[0]
            return match_question
        else:
            match_results = self.model.search(question, threshold=0, top_k=topk)
            questions = self.get_list_questions(match_results)
            # print(match_results)
            ans = "I am not sure what you mean. Can you please choose one of the following questions if there is a match, otherwise, choose 6 to get more questions\n"
            print(ans)
            # conversation.append(f"XAgent:{ans}")
            logging.log(25, f"Xagent: {ans}")
            for idx, question in enumerate(questions[:5]):
                msg = f"{idx+1}. {question}"
                print(msg)
                ans += f"{msg}\n"
            msg = "6. See more questions"
            print(msg)
            ans += f"{msg}\n"
            msg = "Please choose the number of the corresponding question"
            print(msg)
            ans += f"{msg}\n"
            conversation.append(f"Xagent: {ans}")
            logging.log(25, f"Xagent: {ans}")
            choice = input('\033[91m\033[1mUser:\033[0m')
            conversation.append(f"User: {choice}")
            logging.log(25, f"User: {choice}")
            while(True):
                if choice.isnumeric():
                    if int(choice) <= topk+1:
                        break
                msg = "It is not a number or not appropiate number. Please choose another number"
                print(f"\033[1m\033[94mX-Agent:\033[0m {msg}")
                logging.log(25, f"Xagent: {msg}")
                conversation.append(f"Xagent: {msg}")
                choice = input('\033[91m\033[1mUser:\033[0m')
                conversation.append(f"User: {choice}")
                logging.log(25, f"User: {choice}")
            # print(choice)
            # print(str(int(choice) == 6))
            ans = ""
            if int(choice) == 6:
                for idx, question in enumerate(questions[5:15], start=5):
                    msg = f"{idx+1}. {question}"
                    print(msg)
                    ans+=f"{msg}\n"
                msg = "16. None of them"
                print(msg)
                ans += f"{msg}\n"
                msg = "Please choose again the number of the corresponding question"
                print(f"\033[1m\033[94mX-Agent:\033[0m {msg}")
                ans += f"{msg}\n"
                conversation.append(f"Xagent: {ans}")
                logging.log(25, f"Xagent: {ans}")
                choice = input('\033[91m\033[1mUser:\033[0m')
                conversation.append(f"User: {choice}")
                logging.log(25, f"User: {choice}")
                if int(choice) == 16:
                    return "unknown"
                while (True):
                    if choice.isnumeric():
                        if int(choice) < 16:
                            break
                    msg = "It is not a number or not appropiate number. Please choose another number"
                    print(f"\033[1m\033[94mX-Agent:\033[0m {msg}")
                    logging.log(25, f"Xagent: {msg}")
                    choice = input('\033[91m\033[1mUser:\033[0m')
                    conversation.append(f"User: {choice}")
                    logging.log(25, f"User: {choice}")
            return questions[int(choice) - 1]
        return "unknown"
    

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