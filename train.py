# coding: utf-8
import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pickle
from utils import *
from load_data import *


def xgb_train_validate(train_X, train_Y, test_X, test_Y):
    xg_train = xgb.DMatrix(train_X.drop('id', axis=1), label=train_Y)
    xg_test = xgb.DMatrix(test_X.drop('id', axis=1), label=test_Y)
    # setup parameters for xgboost
    param = {}
    # scale weight of positive examples
    param['eta'] = 0.1  # default
    # param['eta'] = 0.02
    param['max_depth'] = 6  # default
    param['silent'] = 1  # default
    param['nthread'] = 4  # default
    param['gamma'] = 1
    param['subsample'] = 0.9
    param['min_child_weight'] = 1
    param['colsample_bytree'] = 0.9
    param['lambda'] = 1
    param['booster'] = 'gbtree'
    param['eval_metric'] = 'rmse'

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    # num_round = 5
    num_round = 200

    # param['objective'] = 'reg:linear'
    param['objective'] = 'reg:gamma'
    bst = xgb.train(param, xg_train, num_round, watchlist)

    # xgb.plot_tree(bst)
    # plt.savefig('xgboost_tree.png')
    # xgb.plot_importance(bst)
    # plt.savefig('xgboost_importance.png')
    imp = bst.get_fscore()
    print(sorted(imp.items(), key=lambda d: d[1], reverse=True))

    # save model to file
    pickle.dump(bst, open("xgboost_bst.model", "wb"))

    pred_prob = bst.predict(xg_test)
    # print('pred_prob:', pred_prob.shape, pred_prob)
    test_rmsle = rmsle(pred_prob, test_Y)
    print('test_rmsle:', test_rmsle)

    pred_prob_train = bst.predict(xg_train)
    # print('pred_prob_train:', pred_prob_train.shape, pred_prob_train)
    train_rmsle = rmsle(pred_prob_train, train_Y)
    print('train_rmsle:', train_rmsle)


def xgb_predict(test_X):
    xg_test = xgb.DMatrix(test_X.drop('id', axis=1))

    # load model from file
    bst = pickle.load(open("xgboost_bst.model", "rb"))

    pred_prob = bst.predict(xg_test)

    test_X['trip_duration'] = pred_prob
    test_X.to_csv('preds_xgboost.csv', sep=',', columns=['id', 'trip_duration'], index=False)


def train_predict(train_X, train_Y, validate_X, validate_Y, test_X, test_Y):
    # offline
    xgb_train_validate(train_X, train_Y, validate_X, validate_Y)

    # online submit
    # xgb_train_validate(pd.concat((train_X, validate_X), axis=0), pd.concat((train_Y, validate_Y), axis=0), validate_X, validate_Y)
    # xgb_predict(test_X)


if __name__ == '__main__':
    base_dir = '../dataset/nyc_taxi/'
    data_pickle_file = base_dir+'data.pickle'

    if os.path.exists(data_pickle_file):
        train_X, train_Y, validate_X, validate_Y, test_X, test_Y = pickle.load(open(data_pickle_file, 'r'))
    else:
        train, validate, test, m, s = load_data(base_dir + 'train.csv', base_dir + 'test.csv')
        weather = load_weather_data(base_dir + 'weather_data_nyc_centralpark_2016.csv')
        train_route, test_route = load_route_data(
            base_dir + 'new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv',
            base_dir + 'new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv',
            base_dir + 'new-york-city-taxi-with-osrm/fastest_routes_test.csv')
        train_X, train_Y, validate_X, validate_Y, test_X, test_Y = prepare_data(train, validate, test, weather, train_route, test_route, m, s)
        pickle.dump((train_X, train_Y, validate_X, validate_Y, test_X, test_Y), open(data_pickle_file, 'w'), protocol=2)

    # train_X = extend_feature(train_X)
    # validate_X = extend_feature(validate_X)
    # test_X = extend_feature(test_X)
    # pickle.dump((train_X, train_Y, validate_X, validate_Y, test_X, test_Y), open(data_pickle_file, 'w'), protocol=2)

    print('train_X:', train_X.shape)
    print(train_X.head())
    print('validate_X:', validate_X.shape)
    print('test_X:', test_X.shape)
    train_predict(train_X, train_Y, validate_X, validate_Y, test_X, test_Y)
