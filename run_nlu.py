import pandas as pd
import argparse
import numpy as np
from simcse import SimCSE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from torch.utils.data.dataset import random_split
class TextClassificationModel(nn.Module):

    def __init__(self, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_class)
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
#         self.linear_relu_stack.weight.data.uniform_(-initrange, initrange)
#         self.linear_relu_stack.bias.data.zero_()

    def forward(self, x):
        return self.linear_relu_stack(x)
class PhraseDataset(Dataset):
    def __init__(self, df, map_label):
        self.df = df
        self.map_label = map_label
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"emb":torch.tensor(self.df['emb'].iloc[idx]),"label":torch.tensor(self.map_label.index(self.df['label'].iloc[idx]))}
        return sample


def train_model(dataloader,model,optimizer,criterion,epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 30
    start_time = time.time()

    for idx, sample in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(sample['emb'])
        label = sample['label']
        loss = criterion(predicted_label, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
def stemming_sentences(text):
    st = PorterStemmer()
    output = []
    for sentence in text:
        output.append(" ".join([st.stem(i) for i in sentence.split()]))
    return output

def evaluate(dataloader,model):
    model.eval()
    total_acc, total_count = 0, 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            predicted_label = model(sample['emb'])
            label = sample['label']
            #             loss = criterion(predicted_label, label)
            predicted_labels.append(predicted_label.argmax(1))
            true_labels.append(label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count, predicted_labels, true_labels
def run(model_name = "simcse-dnn", question_set="all"):
    # print(model_name)
    df = pd.read_csv("Median_4.csv", index_col=0).drop_duplicates()
    if question_set == "xai":
        df = df.loc[df['Label'].isin([5, 6, 8, 11, 12, 15, 17, 20, 64, 67, 68, 69, 71, 73])]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1292)
    if model_name == "simcse-dnn":
        model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
        l_phraseA_embed = model.encode(df['Question'].tolist())
        df['emb'] = [t for t in l_phraseA_embed]
        df_emb = df[['emb', 'Label']].rename(columns={'Label': 'label'})
        map_label = list(set(df_emb['label']))
        # Hyperparameters
        EPOCHS = 50  # epoch
        LR = 6  # learning rate
        BATCH_SIZE = 1  # batch size for training
        y_tests = []
        accuracy = []
        macro_f1 = []
        num_class = len(set(df_emb['label'].tolist()))
        emsize = len(l_phraseA_embed[0])
        y_predicted_labels = []
        for train_, test_ in skf.split(df_emb['emb'], df_emb['label']):
            model = TextClassificationModel(emsize, num_class)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
            df_train = df_emb.iloc[train_]
            df_test = df_emb.iloc[test_]

            dataset_train = PhraseDataset(df_train,map_label)
            dataset_test = PhraseDataset(df_test,map_label)
            dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)
            dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=0)


            for epoch in range(1, EPOCHS + 1):
                train_model(dataloader_train,model,optimizer,criterion,epoch)

            accu_test, y_predicted_label, y_true_labels = evaluate(dataloader_test,model)
            y_tests.append(y_true_labels)
            y_predicted_labels.append(y_predicted_label)
            accuracy.append(accu_test)
            macro_f1.append(metrics.f1_score(y_true_labels, y_predicted_label, average='macro'))
        print("Accuracy= " + str(np.mean(accuracy)) + " +- " + str(np.std(accuracy)))
        print("F1_Macro= " + str(np.mean(macro_f1)) + " +- " + str(np.std(macro_f1)))
    if model_name == "rf":
        clf = Pipeline([('vect', CountVectorizer()),
                        ('tfid', TfidfTransformer()),
                        ('clf', RandomForestClassifier())
                        #                 ('clf', SVC())
                        ])

        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'vect__max_df': [1.0, 0.8],
                      'vect__min_df': [0.1, 0.2, 1],
                      'clf__bootstrap': [True, False],

                      'clf__min_samples_leaf': [1, 2],

                      'clf__n_estimators': [50],
                      }
        accuracy = []
        macro_f1 = []
        micro_f1 = []
        y_scores = []
        y_tests = []
        X = stemming_sentences(df['Question'].tolist())
        y = df['Label'].tolist()
        for train, test in skf.split(X, y):
            X_train = np.array(X)[train]
            y_train = np.array(y)[train]
            X_test = np.array(X)[test]
            y_test = np.array(y)[test]
            y_tests.append(y_test)
            gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
            gs_clf.fit(X_train, y_train)
            pred = gs_clf.predict(X_test)
            y_scores.append(pred)
            print('accuracy', metrics.accuracy_score(y_test, pred))
            macro_f1.append(metrics.f1_score(y_test, pred, average='macro'))
            micro_f1.append(metrics.f1_score(y_test, pred, average='micro'))
            accuracy.append(metrics.accuracy_score(y_test, pred))
        print("Accuracy= " + str(np.mean(accuracy)) + " +- " + str(np.std(accuracy)))
        print("F1_Macro= " + str(np.mean(macro_f1)) + " +- " + str(np.std(macro_f1)))
    if model_name == "svm":
        from sklearn.linear_model import SGDClassifier
        # SGDClassifier with hinge loss gives a linear SVM

        clf = Pipeline([('vect', CountVectorizer()),
                        ('tfid', TfidfTransformer()),
                        #                 ('clf', RandomForestClassifier(max_depth=None, random_state=0))
                        ('clf', SVC())
                        ])

        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'vect__max_df': [1.0, 0.8],
                      'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                      'clf__C': [0.1, 1, 10, 100, 1000],
                      'clf__gamma': [0.1, 1],
                      }

        accuracy = []
        macro_f1 = []
        micro_f1 = []
        y_scores = []
        y_tests = []
        X = stemming_sentences(df['Question'].tolist())
        y = df['Label'].tolist()
        for train, test in skf.split(X, y):
            X_train = np.array(X)[train]
            y_train = np.array(y)[train]
            X_test = np.array(X)[test]
            y_test = np.array(y)[test]
            y_tests.append(y_test)
            gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
            gs_clf.fit(X_train, y_train)
            pred = gs_clf.predict(X_test)
            y_scores.append(pred)
            print('accuracy', metrics.accuracy_score(y_test, pred))
            macro_f1.append(metrics.f1_score(y_test, pred, average='macro'))
            micro_f1.append(metrics.f1_score(y_test, pred, average='micro'))
            accuracy.append(metrics.accuracy_score(y_test, pred))
        print("Accuracy= " + str(np.mean(accuracy)) + " +- " + str(np.std(accuracy)))
        print("F1_Macro= " + str(np.mean(macro_f1)) + " +- " + str(np.std(macro_f1)))
    if model_name == "simcse":
        accuracy = []
        macro_f1 = []
        micro_f1 = []
        y_scores = []
        y_tests = []
        X_tests = []
        X = df['Question'].tolist()
        y = df['Label'].tolist()
        for train, test in skf.split(X, y):
            X_train = np.array(X)[train]
            y_train = np.array(y)[train]
            X_test = np.array(X)[test]
            X_tests.append(X_test)
            y_test = np.array(y)[test]
            y_tests.append(y_test)
            model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
            model.build_index(list(X_train))
            y_scores_pred = [model.search(p1) for p1 in X_test]
            y_scores.append(y_scores_pred)
            y_preds_similarity = [y[X.index(q[0][0])] if len(q) > 1 else 0 for q in y_scores_pred]
            print('accuracy', metrics.accuracy_score(y_test, y_preds_similarity))
            macro_f1.append(metrics.f1_score(y_test, y_preds_similarity, average='macro'))
            micro_f1.append(metrics.f1_score(y_test, y_preds_similarity, average='micro'))
            accuracy.append(metrics.accuracy_score(y_test, y_preds_similarity))
        print("Accuracy= " + str(np.mean(accuracy)) + " +- " + str(np.std(accuracy)))
        print("F1_Macro= " + str(np.mean(macro_f1)) + " +- " + str(np.std(macro_f1)))
        # X_train, X_test, y_train, y_test = train_test_split(list_questions, list_labels,test_size = 0.25, random_state=42, stratify = list_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        "-M",
        type=str,
        default="simcse-dnn",
        help="choose model among 'simcse-dnn', 'simcse', 'rf', 'svm'",
    )
    parser.add_argument(
        "--question_set",
        "-Q",
        type=str,
        default="all",
        help="evaluate on all questions or only xai questions. Choose 'all' or 'xai'",
    )
    args = parser.parse_args()
    run(**vars(args))