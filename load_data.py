# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

'''
id - a unique identifier for each trip
vendor_id - a code indicating the provider associated with the trip record
pickup_datetime - date and time when the meter was engaged
dropoff_datetime - date and time when the meter was disengaged
passenger_count - the number of passengers in the vehicle (driver entered value)
pickup_longitude - the longitude where the meter was engaged
pickup_latitude - the latitude where the meter was engaged
dropoff_longitude - the longitude where the meter was disengaged
dropoff_latitude - the latitude where the meter was disengaged
store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
trip_duration - duration of the trip in seconds
'''


def load_data(train_file, test_file):
    train_columns = 'id,vendor_id,pickup_datetime,dropoff_datetime,passenger_count,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,store_and_fwd_flag,trip_duration'
    test_columns = 'id,vendor_id,pickup_datetime,passenger_count,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,store_and_fwd_flag,trip_duration'
    train = pd.read_csv(train_file, sep=',', names=train_columns.split(','), skiprows=1)
    # print('train:', train.shape)
    # print(train.head())
    test = pd.read_csv(test_file, sep=',', names=test_columns.split(','), skiprows=1)
    # print('test:', test.shape)
    # print(test.head())
    train, validate = train_test_split(train, test_size=0.1)

    return train, validate, test


def prepare_data_one(data):
    data_Y = data['trip_duration']
    data_X = data.drop(['id', 'trip_duration'], axis=1)
    if 'dropoff_datetime' in data_X.keys():
        data_X = data.drop(['dropoff_datetime'], axis=1)
    return data_X, data_Y


def prepare_data(train, validate, test):
    train_X, train_Y = prepare_data_one(train)
    validate_X, validate_Y = prepare_data_one(validate)
    test_X, test_Y = prepare_data_one(test)
    return train_X, train_Y, validate_X, validate_Y, test_X, test_Y


if __name__ == '__main__':
    base_dir = '../dataset/nyc_taxi/'
    train, validate, test = load_data(base_dir+'train.csv', base_dir+'test.csv')
    print('train:', train.shape)
    print(train.head())
    print('validate:', validate.shape)
    print(validate.head())
    print('test:', test.shape)
    print(test.head())

    train_X, train_Y, validate_X, validate_Y, test_X, test_Y = prepare_data(train, validate, test)
    print('train_X:', train_X.shape)
    print(train_X.head())
    print('train_Y:', train_Y.shape)
    print(train_Y.head())
    print('validate_X:', validate_X.shape)
    print(validate_X.head())
    print('validate_Y:', validate_Y.shape)
    print(validate_Y.head())
    print('test_X:', test_X.shape)
    print(test_X.head())
    print('test_Y:', test_Y.shape)
    print(test_Y.head())
