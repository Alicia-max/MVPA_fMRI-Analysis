import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut

if __name__ == '__main__':
    X = np.linspace(0, 1, 100)
    y = 2 * X

    mul_rep = []
    for i in range(1, 5):
        one_rep = np.tile(i, 5)
        mul_rep.extend(one_rep)
    stratify = np.tile(mul_rep, 5)

    skf = LeaveOneGroupOut()
    splits = skf.split(X, y, stratify)
    for train_ind_split, val_ind_split in splits:
        train_x_split = X[train_ind_split]
        train_y_split = y[train_ind_split]

        test_x_split = X[val_ind_split]
        test_y_split = y[val_ind_split]
        print("sdfsdf")
