# coding: utf-8
import datetime
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


def get_weather_date(x):
    date = datetime.datetime.strptime(x, '%d-%m-%Y')
    return date.strftime('%Y-%m-%d')


def load_weather_data(weather_file):
    weather_columns = 'date,maximum_temperature,minimum_temperature,average_temperature,precipitation,snow_fall,snow_depth'
    weather = pd.read_csv(weather_file, sep=',', names=weather_columns.split(','), skiprows=1)
    weather['pickup_date'] = weather['date'].map(lambda x: get_weather_date(x))
    weather.drop(['date'], axis=1, inplace=True)
    weather.replace('T', 0.0, inplace=True)
    weather = weather.apply(pd.to_numeric, errors='ignore')
    return weather


def load_route_data(train_route_file1, train_route_file2, test_route_file):
    route_columns = 'id,starting_street,end_street,total_distance,total_travel_time,number_of_steps,street_for_each_step,distance_per_step,travel_time_per_step,step_maneuvers,step_direction,step_location_list'
    train_route1 = pd.read_csv(train_route_file1, sep=',', names=route_columns.split(','), skiprows=1)
    train_route2 = pd.read_csv(train_route_file2, sep=',', names=route_columns.split(','), skiprows=1)
    train_route = pd.concat((train_route1, train_route2), axis=0)
    test_route = pd.read_csv(test_route_file, sep=',', names=route_columns.split(','), skiprows=1)

    drop_columns = ['starting_street', 'end_street', 'street_for_each_step', 'distance_per_step',
                    'travel_time_per_step', 'step_maneuvers', 'step_direction', 'step_location_list']
    train_route.drop(drop_columns, axis=1, inplace=True)
    test_route.drop(drop_columns, axis=1, inplace=True)

    # train_route = train_route.apply(pd.to_numeric, errors='ignore')
    # test_route = test_route.apply(pd.to_numeric, errors='ignore')

    return train_route, test_route


def get_store_and_fwd_flag_int(x):
    if x == 'N':
        return 0
    return 1


def get_pickup_date(x):
    return x[:10]


def get_pickup_datetime_year(x):
    pickup_datetime = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return pickup_datetime.year


def get_pickup_datetime_month(x):
    pickup_datetime = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return pickup_datetime.month


def get_pickup_datetime_day(x):
    pickup_datetime = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return pickup_datetime.day


def get_pickup_datetime_weekday(x):
    pickup_datetime = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return pickup_datetime.weekday()  # 0-6


def get_pickup_datetime_hour(x):
    pickup_datetime = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return pickup_datetime.hour


def prepare_data_one(data, weather, route):
    data_Y = data['trip_duration']
    data_X = data.drop(['trip_duration'], axis=1)

    data_X['store_and_fwd_flag_int'] = data_X['store_and_fwd_flag'].map(lambda x: get_store_and_fwd_flag_int(x))
    data_X['pickup_date'] = data_X['pickup_datetime'].map(lambda x: get_pickup_date(x))
    # data_X['pickup_datetime_year'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_year(x))
    data_X['pickup_datetime_month'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_month(x))
    data_X['pickup_datetime_day'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_day(x))
    data_X['pickup_datetime_weekday'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_weekday(x))
    data_X['pickup_datetime_hour'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_hour(x))

    data_X = pd.merge(data_X, weather, how='left', on=['pickup_date'])
    data_X = pd.merge(data_X, route, how='left', on=['id'])

    # TODO may need to deal with NaN

    data_X.drop(['store_and_fwd_flag', 'pickup_datetime', 'pickup_date'], axis=1, inplace=True)
    if 'dropoff_datetime' in data_X.keys():
        data_X = data_X.drop(['dropoff_datetime'], axis=1)

    return data_X, data_Y


def prepare_data(train, validate, test, weather, train_route, test_route):
    train_X, train_Y = prepare_data_one(train, weather, train_route)
    validate_X, validate_Y = prepare_data_one(validate, weather, train_route)
    test_X, test_Y = prepare_data_one(test, weather, test_route)
    return train_X, train_Y, validate_X, validate_Y, test_X, test_Y


def main():
    base_dir = '../dataset/nyc_taxi/'
    train, validate, test = load_data(base_dir + 'train.csv', base_dir + 'test.csv')
    print('train:', train.shape)
    print(train.head())
    print('validate:', validate.shape)
    print(validate.head())
    print('test:', test.shape)
    print(test.head())

    weather = load_weather_data(base_dir + 'weather_data_nyc_centralpark_2016.csv')
    print('weather:', weather.shape)
    print(weather.head())

    train_route, test_route = load_route_data(base_dir + 'new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv',
                                              base_dir + 'new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv',
                                              base_dir + 'new-york-city-taxi-with-osrm/fastest_routes_test.csv')
    print('train_route:', train_route.shape)
    print(train_route.head())
    print('test_route:', test_route.shape)
    print(test_route.head())

    train_X, train_Y, validate_X, validate_Y, test_X, test_Y = prepare_data(train, validate, test, weather, train_route,
                                                                            test_route)
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


if __name__ == '__main__':
    main()
