from streamlit.testing.v1 import AppTest
import os, sys
import pandas as pd

PATH = os.path.dirname( __file__)
sys.path.append(os.path.dirname(PATH))

df = pd.read_csv("./tests/test_questions.csv")
list_questions = df['Question'].to_list()
at = AppTest.from_file("app.py", default_timeout=500)
at.run()
at.chat_input[0].set_value("yes").run()
at.chat_input[0].set_value("ok").run()
for q in list_questions:
    try:
        at.chat_input[0].set_value(q).run()
    except KeyError:
        print(KeyError)
assert not at.exception