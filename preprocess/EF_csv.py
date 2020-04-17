from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter

L1_dict = {'cn':8,'de':1,'es':6,'fr':2,'it':3,'jp':4,'kr':5,'sa':0,'tr':7}


if __name__ == "__main__":
    with open('EF201403_selection52.xml') as f:
        soup = BeautifulSoup(f, 'xml')

    text_id = []
    text_level = []
    text_unit = []
    learner_id = []
    L1 = []
    topic = []
    grade = []
    text = []
    writing = soup.writing
    for writing in soup.find_all('writing'):
        if writing.learner['nationality'] not in L1_dict:
            continue
        text_id.append(writing['id'])
        text_level.append(writing['level'])
        text_unit.append(writing['unit'])
        learner_id.append(writing.learner['id'])
        L1.append(L1_dict[writing.learner['nationality']])
        topic.append(writing.topic['id'])
        grade.append(writing.grade)
        text.append(writing.text)

    for i in range(len(text)):
        text[i] = text[i].split('\n')[6]

    df = pd.DataFrame(columns = ['TextFile', 'L1', 'Sentence', 'Level', 'Unit', 'LearnerId', 'Topic','Grade'])
    df['TextFile'] = text_id
    df['L1'] = L1
    df['Sentence'] = text
    df['Level'] = text_level
    df['Unit'] = text_unit
    df['LearnerId'] = learner_id
    df['Topic'] = topic
    df['Grade'] = grade

    df.to_csv('EF.csv')
