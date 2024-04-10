import pandas as pd
import os


def compare_names():
    # Load the training data
    train_x = pd.read_csv('data/train_x_202108.csv')
    train_y = pd.read_csv('data/train_y_202108.csv')

    # Load the testing data
    test_x = pd.read_csv('data/test_x_202109.csv')
    test_y = pd.read_csv('data/test_y_202109.csv')

    # first column 'ord_no' is id, check whether x and y have the same id
    assert (train_x['ord_no'] == train_y['ord_no']).all()
    assert (test_x['ord_no'] == test_y['ord_no']).all()


if __name__ == '__main__':
    compare_names()
    print('Data names are consistent')
    