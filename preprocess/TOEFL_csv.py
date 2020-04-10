import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
data_path = "ETS_Corpus_of_Non-Native_Written_English/data/text/responses/original"

# dictionary from L1 to label
L1D = {'ARA':0, 'DEU':1, 'FRA':2, 'ITA':3, 'JPN':4, 'KOR':5, 'SPA':6, 'TUR':7, 'ZHO':8, 'HIN':9, 'TEL':10}
PromptD = {'P1':0, 'P2':1, 'P3':2, 'P4':3, 'P5':4, 'P6':5, 'P7':6, 'P8':7}

def get_TOEFL_essay(data_name):
    data_numpy = data_name.values
    sentences = []
    for i in range(len(data_numpy)):
        with open('{0}/{1}'.format(data_path,data_numpy[i][0])) as f:
            sentence = ""
            line_s = f.readlines()
            for line in line_s:
                line = line.replace('\n','')
                line = line.replace('\t','')
                sentence += line
                sentence += ' '
            sentence = sentence[:-1]
            sentences.append(sentence)
    return sentences

def clean_essay(data_name):
    sentences = get_TOEFL_essay(data_name)
    for i in range(len(sentences)):
        pre = ' '
        sen = sentences[i]
        new = ''
        for char in sen:
            if char == '.':
                if pre==' ':
                    new = new[:-1]
            if pre==' ':
                if char == ' ':
                    continue
            if pre=='.' or pre=='!' or pre=='?' or pre==',':
                if char != ' ':
                    new+=' '
                if char == '.':
                    continue
            new += char
            pre = char
        sentences[i] = new
    return sentences

def make_sentences(data_numpy, data_sen):
    texts = []
    prompts = []
    targets = []
    proficiency = []
    sentences = []

    for i in range(len(data_numpy)):
        for sen in sent_tokenize(data_sen[i]):
            # too much tokens
            if len(tokenizer.tokenize(sen)) > 128:
                sen128 = ""
                for sen_detail in sen.split(','):
                    if sen128 == "":
                        sen128 = sen_detail
                    elif len(tokenizer.tokenize(sen128+','+sen_detail)) < 128:
                        sen128 += ',' + sen_detail
                    else:
                        # if there are too much sen128
                        sentences.append(sen128)
                        texts.append(data_numpy[i,0])
                        prompts.append(data_numpy[i,1])
                        targets.append(data_numpy[i,2])
                        proficiency.append(data_numpy[i,3])
                        sen128 = sen_detail

                sentences.append(sen128)
                texts.append(data_numpy[i,0])
                prompts.append(data_numpy[i,1])
                targets.append(data_numpy[i,2])
                proficiency.append(data_numpy[i,3])

            # not too much tokens
            else:
                sentences.append(sen)
                texts.append(data_numpy[i,0])
                prompts.append(data_numpy[i,1])
                targets.append(data_numpy[i,2])
                proficiency.append(data_numpy[i,3])

    df = pd.DataFrame()
    df['TextFile'] = texts
    df['Prompt'] = prompts
    df['L1'] = targets
    df['Proficiency'] = proficiency
    df['Sentence'] = sentences

    return df

if __name__ == "__main__":
    # get dataset
    train_data = pd.read_csv('ETS_Corpus_of_Non-Native_Written_English/data/text/index-training.csv',names=['TextFile','Prompt','L1','Proficiency'])
    dev_data = pd.read_csv('ETS_Corpus_of_Non-Native_Written_English/data/text/index-dev.csv', names=['TextFile','Prompt','L1','Proficiency'])
    test_data = pd.read_csv('ETS_Corpus_of_Non-Native_Written_English/data/text/index-test.csv', names=['TextFile','Prompt','Proficiency'])

    # clean dataset
    train_data['Sentence'] = clean_essay(train_data)
    dev_data['Sentence'] = clean_essay(dev_data)
    test_data['Sentence'] = clean_essay(test_data)

    # preprocessing test dataset
    index_data = pd.read_csv('ETS_Corpus_of_Non-Native_Written_English/data/text/index.csv', names=['TextFile','Prompt','L1','Proficiency'])
    index_text = index_data.TextFile
    index_L1 = index_data.L1
    test_text = test_data.TextFile
    test_dict = {}
    for i in range(len(index_text)):
        test_dict[index_text[i]] = index_L1[i]
    test_L1 = []
    for i in range(len(test_text)):
        test_L1.append(test_dict[test_text[i]])
    test_data['L1'] = test_L1
    test_data = test_data[["TextFile","Prompt","L1","Proficiency","Sentence"]]

    # convert L1 string label to integer
    for i in range(train_data.shape[0]):
        train_data.L1[i] = L1D[train_data.L1[i]]
    for i in range(dev_data.shape[0]):
        dev_data.L1[i] = L1D[dev_data.L1[i]]
    for i in range(test_data.shape[0]):
        test_data.L1[i] = L1D[test_data.L1[i]]

    # convert Prompt string label to integer
    for i in range(train_data.shape[0]):
        train_data.Prompt[i] = PromptD[train_data.Prompt[i]]
    for i in range(dev_data.shape[0]):
        dev_data.Prompt[i] = PromptD[dev_data.Prompt[i]]
    for i in range(test_data.shape[0]):
        test_data.Prompt[i] = PromptD[test_data.Prompt[i]]

    # save dataset
    train_data.to_csv('TOEFL11/train.csv')
    dev_data.to_csv('TOEFL11/dev.csv')
    test_data.to_csv('TOEFL11/test.csv')
    train_dev_data = pd.concat([train_data, dev_data])
    train_dev_data.to_csv('TOEFL11/train_dev.csv')
    train_dev_test_data = pd.concat([train_dev_data, test_data])
    train_dev_test_data.to_csv('TOEFL11/train_dev_test.csv')

    # making sentences dataset
    train_numpy = train_data.to_numpy()[:,:-1]
    dev_numpy = dev_data.to_numpy()[:,:-1]
    test_numpy = test_data.to_numpy()[:,:-1]
    train_sen = train_data.to_numpy()[:,-1]
    dev_sen = dev_data.to_numpy()[:,-1]
    test_sen = test_data.to_numpy()[:,-1]
    train_sentences_df = make_sentences(train_numpy, train_sen)
    dev_sentences_df = make_sentences(dev_numpy, dev_sen)
    test_sentences_df = make_sentences(test_numpy, test_sen)

    # save dataset
    train_sentences_df.to_csv('TOEFL11/train_sentences.csv')
    dev_sentences_df.to_csv('TOEFL11/dev_sentences.csv')
    test_sentences_df.to_csv('TOEFL11/test_sentences.csv')
    train_dev_sentences_df = pd.concat([train_sentences_df, dev_sentences_df])
    train_dev_sentences_df.to_csv('TOEFL11/train_dev_sentences.csv')
    train_dev_test_sentences_df = pd.concat([train_dev_sentences_df, test_sentences_df])
    train_dev_test_sentences_df.to_csv('TOEFL11/train_dev_test_sentences.csv')
