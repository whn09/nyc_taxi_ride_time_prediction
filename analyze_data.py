# coding: utf-8
import os
import pickle
import pandas as pd
pd.set_option('display.max_columns', None)


def stat(x):
    return pd.Series(
        [x.count(), x.min(), x.idxmin(), x.quantile(.25), x.median(), x.quantile(.75), x.mean(), x.max(), x.idxmax(),
         x.mad(), x.var(), x.std(), x.skew(), x.kurt()],
        index=['总数', '最小值', '最小值位置', '25%分位数', '中位数', '75%分位数', '均值', '最大值', '最大值位数', '平均绝对偏差', '方差', '标准差', '偏度',
               '峰度'])


def analyze_data(data_X, data_Y):
    data_X.drop(['id'], axis=1, inplace=True)
    data_X_stat = data_X.apply(stat)
    print(data_X_stat)
    # data_Y_stat = data_Y.apply(stat)
    # print(data_Y_stat)


if __name__ == '__main__':
    base_dir = '../dataset/nyc_taxi/'
    data_pickle_file = base_dir + 'data.pickle'

    if os.path.exists(data_pickle_file):
        train_X, train_Y, validate_X, validate_Y, test_X, test_Y = pickle.load(open(data_pickle_file, 'r'))
        train_validate_X = pd.concat((train_X, validate_X), axis=0)
        train_validate_Y = pd.concat((train_Y, validate_Y), axis=0)
        # print('train_validate_X:')
        # print(train_validate_X)
        # print('train_validate_Y:')
        # print(train_validate_Y)
        # train_validate = pd.concat((train_validate_X, train_validate_Y), axis=1, ignore_index=True)  # TODO bug
        # train_validate.to_csv('train_validate.csv', index=False)
        train_validate_X.to_csv('train_validate_X.csv', index=False)
        train_validate_Y.to_csv('train_validate_Y.csv', index=False)
        test_X.to_csv('test_X.csv', index=False)
        analyze_data(train_X, train_Y)
