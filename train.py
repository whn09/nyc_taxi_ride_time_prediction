# coding: utf-8
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib
import pickle
from utils import *
from load_data import *


# LR
def lr_train_pred(train_X, train_Y, test_X, test_Y):
    lr = LogisticRegression(penalty='l2')
    lr.fit(train_X.astype(np.float), train_Y.astype(np.float))
    pred_label = lr.predict(test_X.astype(np.float))
    roc(test_Y.astype(np.float), pred_label, 'lr_roc.png')
    ks(test_Y.astype(np.float), pred_label, 'lr_ks.png')
    pred_label_train = lr.predict(train_X.astype(np.float))
    roc(train_Y.astype(np.float), pred_label_train, 'lr_train_roc.png')
    ks(train_Y.astype(np.float), pred_label_train, 'lr_train_ks.png')
    return pred_label, pred_label_train


# 随机森林
def rfc_train_pred(train_X, train_Y, test_X, test_Y):
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(train_X.astype(np.float), train_Y.astype(np.float))
    pred_label = rfc.predict(test_X.astype(np.float))
    roc(test_Y.astype(np.float), pred_label, 'rfc_roc.png')
    ks(test_Y.astype(np.float), pred_label, 'rfc_ks.png')
    pred_label_train = rfc.predict(train_X.astype(np.float))
    roc(train_Y.astype(np.float), pred_label_train, 'rfc_train_roc.png')
    ks(train_Y.astype(np.float), pred_label_train, 'rfc_train_ks.png')
    return pred_label, pred_label_train


# XGBoost
def xgb_train_pred(train_X, train_Y, test_X, test_Y):
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    # setup parameters for xgboost
    param = {}
    # scale weight of positive examples
    param['eta'] = 0.1  # default
    # param['eta'] = 0.02
    param['max_depth'] = 6  # default
    param['silent'] = 1  # default
    param['nthread'] = 4  # default
    ##param['num_class'] = 6 # default
    param['num_class'] = 2
    # param['gamma'] = 1
    # param['subsample'] = 0.6

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    # num_round = 5
    num_round = 500

    # do the same thing again, but output probabilities
    param['objective'] = 'multi:softprob'
    # param['objective'] = 'multi:softmax'
    # param['objective'] = 'binary:logistic'
    bst = xgb.train(param, xg_train, num_round, watchlist)

    xgb.plot_tree(bst)
    plt.savefig('xgboost_tree.png')
    xgb.plot_importance(bst)
    plt.savefig('xgboost_importance.png')

    # save model to file
    pickle.dump(bst, open("xgboost_bst.model", "wb"))
    # load model from file
    bst = pickle.load(open("xgboost_bst.model", "rb"))

    imp = bst.get_fscore()
    print(sorted(imp.items(), key=lambda d: d[1], reverse=True))

    # Note: this convention has been changed since xgboost-unity
    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], param['num_class'])
    roc(test_Y, pred_prob[:, 1], 'xgboost_roc.png')
    ks(test_Y, pred_prob[:, 1], 'xgboost_ks.png')
    pred_label = np.argmax(pred_prob, axis=1)
    error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
    print('Test error = {}'.format(error_rate))
    print(pred_prob.shape)
    test_rmsle = rmsle(pred_prob[:, 1], test_Y)
    print('test_rmsle:', test_rmsle)

    pred_prob_train = bst.predict(xg_train).reshape(train_Y.shape[0], param['num_class'])
    roc(train_Y, pred_prob_train[:, 1], 'xgboost_train_roc.png')
    ks(train_Y, pred_prob_train[:, 1], 'xgboost_train_ks.png')
    pred_label_train = np.argmax(pred_prob_train, axis=1)
    error_rate = np.sum(pred_label_train != train_Y) / train_Y.shape[0]
    print('Train error = {}'.format(error_rate))
    print(pred_prob_train.shape)
    train_rmsle = rmsle(pred_prob_train[:, 1], train_Y)
    print('train_rmsle:', train_rmsle)

    return pred_label, pred_label_train


def train(train_X, train_Y, validate_X, validate_Y, test_X, test_Y):
    xgb_train_pred(train_X, train_Y, validate_X, validate_Y)


if __name__ == '__main__':
    base_dir = '../dataset/nyc_taxi/'
    train, validate, test = load_data(base_dir + 'train.csv', base_dir + 'test.csv')
    train_X, train_Y, validate_X, validate_Y, test_X, test_Y = prepare_data(train, validate, test)
    train(train_X, train_Y, validate_X, validate_Y, test_X, test_Y)
