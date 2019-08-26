"""
model 1: age, R1_necrosis, R1_heterogeneous, R1_capsular in multivariable model
model 2: age, R1_necrosis, R1_heterogeneous, R1_capsular, FD_average_1till7, Lac_average_1till7 in multivariable model
"""

import os
import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression

def logistic(train_x, train_y):
    print('<< implement logistic regression classifier ... >>')
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(train_x, train_y)
    return logreg

def main():
    fname = "20190814_Entire_Train and Test_Meningioma.csv"
    data = pd.read_csv(fname)
    print(data.head())
    print(data)

if __name__ == "__main__":
    main()
