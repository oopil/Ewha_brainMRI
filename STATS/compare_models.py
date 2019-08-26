"""
model 1: age, R1_necrosis, R1_heterogeneous, R1_capsular in multivariable model
model 2: age, R1_necrosis, R1_heterogeneous, R1_capsular, FD_average_1till7, Lac_average_1till7 in multivariable model
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

def logistic(train_x, train_y):
    print('<< implement logistic regression classifier ... >>')
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(train_x, train_y)
    return logreg

def main():
    # -------------- model 1 --------------#
    fname = "20190814_Entire_Train and Test_Meningioma.csv"
    data = pd.read_csv(fname)
    tr = data[data.train == 1]
    tr_y = tr.low_or_high
    tr_x = tr[["age", "R1_necrosis", "R1_heterogeneous", "R1_capsular"]]

    ts = data[data.train == 0]
    ts_y = ts.low_or_high
    ts_x = ts[["age", "R1_necrosis", "R1_heterogeneous", "R1_capsular"]]

    # YWP already did it ...

if __name__ == "__main__":
    main()
