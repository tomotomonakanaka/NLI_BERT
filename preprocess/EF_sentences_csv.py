import pandas as pd
import numpy as np
from TOEFL_csv import make_sentences
test_df = pd.read_csv('EF/EF_converted.csv')
test_data = test_df.to_numpy()
test_numpy = np.array([test_data[:,1], test_data[:,2], test_data[:,3], test_data[:,1]])
test_numpy = test_numpy.T
test_sentences = test_data[:,4]

test_sentences_df = make_sentences(test_numpy, test_sentences)
test_sentences_df.to_csv("EF/EF_test_sentences.csv")
