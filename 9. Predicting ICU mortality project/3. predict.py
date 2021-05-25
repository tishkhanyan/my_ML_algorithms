from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import argparse
import pickle
import json

parser = argparse.ArgumentParser(description='Receive the test data')
parser.add_argument('--test', help="file name to read the test data from")
args = parser.parse_args()

test = pd.read_csv(args.test)
test_length = test.shape[0]

X_train = pd.read_csv('X_train.csv')

test['BUN/Creatinine_first'] = test.BUN_first / test.Creatinine_first
test['BUN/Creatinine_last'] = test.BUN_last / test.Creatinine_last

test['BUN/Creatinine_first'] = test.BUN_first / test.Creatinine_first
test['BUN/Creatinine_last'] = test.BUN_last / test.Creatinine_last

test['SAPS-I'] = test['SAPS-I'].map(lambda v: np.nan if v == -1 else v)
test['SOFA'] = test['SOFA'].map(lambda v: np.nan if v == -1 else v)

test.Glucose_first = test.Glucose_first.map(lambda v: 4 if v >= 300 else (3 if v <= 15 else (2 if v <= 40 else 1)))
test.Glucose_last = test.Glucose_last.map(lambda v: 4 if v >= 300 else (3 if v <= 15 else (2 if v <= 40 else 1)))

test.HR_first = test.HR_first.map(lambda v: 1 if 60 <= v <= 100 else(2 if v < 60 else 3))
test.HR_last = test.HR_last.map(lambda v: 1 if 60 <= v <= 100 else(2 if v < 60 else 3))

test['PaO2/FiO2_first'] = test.PaO2_first / test.FiO2_first
test['PaO2/FiO2_last'] = test.PaO2_last / test.FiO2_last
test['PaCO2/FiO2_first'] = test.PaCO2_first / test.FiO2_first
test['PaCO2/FiO2_last'] = test.PaCO2_last / test.FiO2_last

useless = []
for feature in test:
    if feature not in X_train.columns:
        useless.append(feature)
test.drop(useless, axis=1, inplace=True)
test.sort_index(axis=1, inplace=True)

fileObj = open('min_max.obj', 'rb')
min_max_dict = pickle.load(fileObj)
fileObj.close()

for feature in test:
    test[feature] = test[feature].map(lambda v:
                                      (v-min_max_dict[feature][0]) /
                                      (min_max_dict[feature][1]-min_max_dict[feature][0]))

temp = pd.concat([X_train, test])

imputer = KNNImputer()
imputer.fit(temp)

temp = pd.DataFrame(data=imputer.transform(temp), columns=temp.columns)

X_test = temp.tail(test_length)
X_test.reset_index(drop=True, inplace=True)

fileObj = open('model.obj', 'rb')
model = pickle.load(fileObj)
fileObj.close()

output = dict()
output['pred_proba'] = list(np.round(model.predict_proba(X_test)[:, 1], 2))
output['threshold'] = 0.5

with open("pred_proba.json", "w") as outfile:
    json.dump(output, outfile)
