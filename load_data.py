# coding: utf-8
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

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

    m = np.mean(train['trip_duration'])
    s = np.std(train['trip_duration'])

    train, validate = train_test_split(train, test_size=0.1)

    return train, validate, test, m, s


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
    # pickup_datetime = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    # return pickup_datetime.month
    return x.month


def get_pickup_datetime_day(x):
    # pickup_datetime = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    # return pickup_datetime.day
    return x.day


def get_pickup_datetime_weekday(x):
    # pickup_datetime = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    # return pickup_datetime.weekday()  # 0-6
    return x.weekday()


def get_pickup_datetime_hour(x):
    # pickup_datetime = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    # return pickup_datetime.hour
    return x.hour


def get_euclid_distance(x):
    pickup_longitude = x['pickup_longitude']
    pickup_latitude = x['pickup_latitude']
    dropoff_longitude = x['dropoff_longitude']
    dropoff_latitude = x['dropoff_latitude']
    euclid_distance = haversine(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    return euclid_distance


def get_manhattan_distance(x):
    pickup_longitude = x['pickup_longitude']
    pickup_latitude = x['pickup_latitude']
    dropoff_longitude = x['dropoff_longitude']
    dropoff_latitude = x['dropoff_latitude']
    manhattan_distance = haversine(pickup_longitude, pickup_latitude, pickup_longitude, dropoff_latitude) + haversine(pickup_longitude, dropoff_latitude, dropoff_longitude, dropoff_latitude)
    return manhattan_distance


def get_direction(x):
    pickup_longitude = x['pickup_longitude']
    pickup_latitude = x['pickup_latitude']
    dropoff_longitude = x['dropoff_longitude']
    dropoff_latitude = x['dropoff_latitude']
    direction = bearing_array(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    return direction


def get_direction_8(x):
    directions = [45*i-22.5 for i in [-3, -2, -1, 0, 1, 2, 3, 4]]
    direction_8 = 0
    for i in range(len(directions)):
        if directions[i-1] <= x < directions[i]:
            direction_8 = i
            break
    return direction_8


def prepare_data_one(data, weather, route, m, s, stage):
    start = time.time()
    print('data.shape:', data.shape)

    if stage != 'test':
        data = data[data['trip_duration'] <= m + 2 * s]
        data = data[data['trip_duration'] >= m - 2 * s]
        print('filter trip_duration data.shape:', data.shape, time.time()-start)

        city_long_border = (-74.03, -73.75)
        city_lat_border = (40.63, 40.85)
        data = data[data['pickup_longitude'] <= city_long_border[1]]
        data = data[data['pickup_longitude'] >= city_long_border[0]]
        data = data[data['pickup_latitude'] <= city_lat_border[1]]
        data = data[data['pickup_latitude'] >= city_lat_border[0]]
        data = data[data['dropoff_longitude'] <= city_long_border[1]]
        data = data[data['dropoff_longitude'] >= city_long_border[0]]
        data = data[data['dropoff_latitude'] <= city_lat_border[1]]
        data = data[data['dropoff_latitude'] >= city_lat_border[0]]
        print('filter pickup and dropoff data.shape:', data.shape, time.time()-start)

    data_Y = data['trip_duration']
    data_X = data

    data_X['store_and_fwd_flag_int'] = data_X['store_and_fwd_flag'].map(lambda x: get_store_and_fwd_flag_int(x))
    print('store_and_fwd_flag_int done', time.time()-start)
    data_X['pickup_date'] = data_X['pickup_datetime'].map(lambda x: get_pickup_date(x))
    print('pickup_date done', time.time()-start)
    data_X['pickup_datetime'] = pd.to_datetime(data_X['pickup_datetime'])
    print('pickup_datetime done', time.time() - start)
    # data_X['pickup_datetime_year'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_year(x))
    # print('pickup_datetime_year done', time.time()-start)
    data_X['pickup_datetime_month'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_month(x))
    print('pickup_datetime_month done', time.time()-start)
    data_X['pickup_datetime_day'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_day(x))
    print('pickup_datetime_day done', time.time()-start)
    data_X['pickup_datetime_weekday'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_weekday(x))
    print('pickup_datetime_weekday done', time.time()-start)
    data_X['pickup_datetime_hour'] = data_X['pickup_datetime'].map(lambda x: get_pickup_datetime_hour(x))
    print('pickup_datetime_hour done', time.time()-start)
    data_X['is_weekday'] = data_X['pickup_datetime_weekday'].map(lambda x: x < 5 and 1 or 0)
    print('is_weekday done', time.time()-start)
    data_X['is_weekend'] = data_X['pickup_datetime_weekday'].map(lambda x: x >= 5 and 1 or 0)
    print('is_weekend done', time.time()-start)
    data_X['is_morning_peak'] = data_X['pickup_datetime_hour'].map(lambda x: 7 <= x <= 9 and 1 or 0)
    print('is_morning_peak done', time.time()-start)
    data_X['is_evening_peak'] = data_X['pickup_datetime_hour'].map(lambda x: 17 <= x <= 19 and 1 or 0)
    print('is_evening_peak done', time.time()-start)

    data_X['euclid_distance'] = data_X.apply(lambda x: get_euclid_distance(x), axis=1)
    print('euclid_distance done', time.time()-start)
    data_X['manhattan_distance'] = data_X.apply(lambda x: get_manhattan_distance(x), axis=1)
    print('manhattan_distance done', time.time()-start)
    data_X['direction'] = data_X.apply(lambda x: get_direction(x), axis=1)
    print('direction done', time.time()-start)
    data_X['direction_8'] = data_X['direction'].map(lambda x: get_direction_8(x))
    print('direction_8 done', time.time()-start)

    data_X['pickup_longitude_grid'] = data_X['pickup_longitude'].map(lambda x: round(x, 3))
    print('pickup_longitude_grid done', time.time()-start)
    data_X['pickup_latitude_grid'] = data_X['pickup_longitude'].map(lambda x: round(x, 3))
    print('pickup_latitude_grid done', time.time()-start)
    data_X['dropoff_longitude_grid'] = data_X['dropoff_longitude'].map(lambda x: round(x, 3))
    print('dropoff_longitude_grid done', time.time()-start)
    data_X['dropoff_latitude_grid'] = data_X['dropoff_latitude'].map(lambda x: round(x, 3))
    print('dropoff_latitude_grid done', time.time()-start)

    data_X = pd.merge(data_X, weather, how='left', on=['pickup_date'])
    print('weather merge done', time.time()-start)
    data_X = pd.merge(data_X, route, how='left', on=['id'])
    print('route merge done', time.time()-start)

    data_X.drop(['store_and_fwd_flag', 'pickup_datetime', 'pickup_date', 'trip_duration'], axis=1, inplace=True)
    if stage != 'test':  # if 'dropoff_datetime' in data_X.keys():
        data_X = data_X.drop(['dropoff_datetime'], axis=1)
    print('drop done', time.time()-start)

    return data_X, data_Y


def extend_feature(data_X):
    data_X.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1, inplace=True)
    return data_X


def prepare_data(train, validate, test, weather, train_route, test_route, m, s):
    print('prepare_data train')
    train_X, train_Y = prepare_data_one(train, weather, train_route, m, s, stage='train')
    print('prepare_data validate')
    validate_X, validate_Y = prepare_data_one(validate, weather, train_route, m, s, stage='validate')
    print('prepare_data test')
    test_X, test_Y = prepare_data_one(test, weather, test_route, m, s, stage='test')
    return train_X, train_Y, validate_X, validate_Y, test_X, test_Y


def main():
    base_dir = '../dataset/nyc_taxi/'
    train, validate, test, m, s = load_data(base_dir + 'train.csv', base_dir + 'test.csv')
    print('train:', train.shape)
    print(train.head())
    print('validate:', validate.shape)
    print(validate.head())
    print('test:', test.shape)
    print(test.head())
    print('m:', m)
    print('s:', s)

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
                                                                            test_route, m, s)
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

    # print('train precipitation:', train_X['precipitation'].unique())
    # print('train snow_fall:', train_X['snow_fall'].unique())
    # print('train snow_depth:', train_X['snow_depth'].unique())
    # print('train total_distance:', train_X['total_distance'].unique())
    # print('train total_travel_time:', train_X['total_travel_time'].unique())
    # print('train number_of_steps:', train_X['number_of_steps'].unique())


if __name__ == '__main__':
    main()
