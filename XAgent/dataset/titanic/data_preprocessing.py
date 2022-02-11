import pandas as pd
import string
import numpy as np
from scipy.stats import mode
import json
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str(big_string).find(substring) != -1:
            return substring
    return 'Unknown'
def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
if __name__== "__main__" :
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                        'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                        'Don', 'Jonkheer']
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    names = ["clean_train.csv","clean_test.csv"]
    for df,name in zip([train, test],names):
        df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))
        # replacing all titles with mr, mrs, miss, master
        df['Title'] = df.apply(replace_titles, axis=1)
        # Turning cabin number into Dec
        df['Deck'] = df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
        # Creating new family_size column
        df['Family_Size'] = df['SibSp'] + df['Parch']
        classmeans = df.pivot_table('Fare', columns='Pclass', aggfunc='mean')
        df.Fare = df[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )
        meanAge=np.mean(df.Age)
        df.Age=df.Age.fillna(meanAge)
        modeEmbarked = mode(df.Embarked)[0][0]
        df.Embarked = df.Embarked.fillna(modeEmbarked)
        df_out = df.drop(['Cabin','SibSp','Parch','Name', "Ticket", "PassengerId"], axis=1)
        df_out.to_csv(name, index = False)
    info = {}
    info['cat_features'] = ['Pclass','Age','Embarked','Title','Deck']
    info['num_features'] = ['Age', 'Fare', 'Family_Size']
    info['predict_prompt'] = ['Unfortunately, this person will die','Congrats, you will survive']


    with open('info.json', 'w') as fp:
        json.dump(info, fp)
    print("")

