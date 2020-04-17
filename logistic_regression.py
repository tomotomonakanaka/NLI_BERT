import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

max_iters = [5,7,9,13,16,20,25,50]
penalties = ['l2', 'none']
tols = [0.01,0.005,0.001,0.0005,0.0001]
Cs = [0.5,0.7,0.9,1.0,1.2,1.5,1.8]
# files
TRAIN_ROW_PATH = "TOEFL11/train.csv"
TEST_ROW_PATH = "TOEFL11/test.csv"
train_file_name = 'train_hiddens'
test_file_name = 'test_hiddens'

# load X
train_file = open(train_file_name, 'rb')
X_train = pickle.load(train_file)
train_file.close()
test_file = open(test_file_name, 'rb')
X_test = pickle.load(test_file)
test_file.close()

# load y
train_df_answer = pd.read_csv(TRAIN_ROW_PATH)
y_train = train_df_answer.L1.values
test_df_answer = pd.read_csv(TEST_ROW_PATH)
y_test = test_df_answer.L1.values

clf = LogisticRegression(max_iter=15).fit(X_train, y_train)
print(clf.score(X_test, y_test))
#
# for penalty in penalties:
#     for tol in tols:
#         for C in Cs:
#             for max_iter in max_iters:
#                 clf = LogisticRegression(penalty=penalty, tol=tol, C=C, max_iter=max_iter).fit(X_train, y_train)
#                 print(penalty, tol, C, max_iter, clf.score(X_test, y_test))
