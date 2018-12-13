# coding: utf-8
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
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
    train = train[train['trip_duration'] <= m + 2 * s]
    train = train[train['trip_duration'] >= m - 2 * s]
    print('filter trip_duration train.shape:', train.shape)

    city_long_border = (-74.03, -73.75)
    city_lat_border = (40.63, 40.85)
    train = train[train['pickup_longitude'] <= city_long_border[1]]
    train = train[train['pickup_longitude'] >= city_long_border[0]]
    train = train[train['pickup_latitude'] <= city_lat_border[1]]
    train = train[train['pickup_latitude'] >= city_lat_border[0]]
    train = train[train['dropoff_longitude'] <= city_long_border[1]]
    train = train[train['dropoff_longitude'] >= city_long_border[0]]
    train = train[train['dropoff_latitude'] <= city_lat_border[1]]
    train = train[train['dropoff_latitude'] >= city_lat_border[0]]
    print('filter pickup and dropoff train.shape:', train.shape)

    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                        train[['dropoff_latitude', 'dropoff_longitude']].values))
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

    return train, test, kmeans


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
    euclid_distance = haversine_array(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    return euclid_distance


def get_manhattan_distance(x):
    pickup_longitude = x['pickup_longitude']
    pickup_latitude = x['pickup_latitude']
    dropoff_longitude = x['dropoff_longitude']
    dropoff_latitude = x['dropoff_latitude']
    manhattan_distance = haversine_array(pickup_longitude, pickup_latitude, pickup_longitude, dropoff_latitude) + haversine_array(pickup_longitude, dropoff_latitude, dropoff_longitude, dropoff_latitude)
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


def prepare_data_one(data, weather, route, kmeans, stage):
    start = time.time()
    print('data.shape:', data.shape)

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

    # data_X['euclid_distance'] = data_X.apply(lambda x: get_euclid_distance(x), axis=1)
    data_X['euclid_distance'] = haversine_array(data_X['pickup_longitude'].values, data_X['pickup_latitude'].values, data_X['dropoff_longitude'].values, data_X['dropoff_latitude'].values)
    print('euclid_distance done', time.time()-start)
    # data_X['manhattan_distance'] = data_X.apply(lambda x: get_manhattan_distance(x), axis=1)
    data_X['manhattan_distance'] = haversine_array(data_X['pickup_longitude'].values, data_X['pickup_latitude'].values, data_X['pickup_longitude'].values, data_X['dropoff_latitude'].values) + haversine_array(data_X['dropoff_longitude'].values, data_X['dropoff_latitude'].values, data_X['pickup_longitude'].values, data_X['dropoff_latitude'].values)
    print('manhattan_distance done', time.time()-start)
    # data_X['direction'] = data_X.apply(lambda x: get_direction(x), axis=1)
    data_X['direction'] = bearing_array(data_X['pickup_longitude'].values, data_X['pickup_latitude'].values, data_X['dropoff_longitude'].values, data_X['dropoff_latitude'].values)
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

    data_X['pickup_cluster'] = kmeans.predict(data_X[['pickup_latitude', 'pickup_longitude']])
    print('pickup_cluster done', time.time() - start)
    data_X['dropoff_cluster'] = kmeans.predict(data_X[['dropoff_latitude', 'dropoff_longitude']])
    print('dropoff_cluster done', time.time() - start)

    # vendor_id_dummy = pd.get_dummies(data_X['vendor_id'], prefix='vi', prefix_sep='_')
    # print('vendor_id_dummy done', vendor_id_dummy.shape, time.time()-start)
    # passenger_count_dummy = pd.get_dummies(data_X['passenger_count'], prefix='pc', prefix_sep='_')
    # if stage == 'test':
    #     passenger_count_dummy = passenger_count_dummy.drop('pc_9', axis=1)  # filter test abnormal value
    # print('passenger_count_dummy done', passenger_count_dummy.shape, time.time()-start)
    # store_and_fwd_flag_int_dummy = pd.get_dummies(data_X['store_and_fwd_flag_int'], prefix='sf', prefix_sep='_')
    # print('store_and_fwd_flag_int_dummy done', store_and_fwd_flag_int_dummy.shape, time.time()-start)
    # pickup_cluster_dummy = pd.get_dummies(data_X['pickup_cluster'], prefix='pic', prefix_sep='_')
    # print('pickup_cluster_dummy done', pickup_cluster_dummy.shape, time.time()-start)
    # dropoff_cluster_dummy = pd.get_dummies(data_X['dropoff_cluster'], prefix='drc', prefix_sep='_')
    # print('dropoff_cluster_dummy done', dropoff_cluster_dummy.shape, time.time()-start)
    # pickup_datetime_month_dummy = pd.get_dummies(data_X['pickup_datetime_month'], prefix='pm', prefix_sep='_')
    # print('pickup_datetime_month_dummy done', pickup_datetime_month_dummy.shape, time.time()-start)
    # pickup_datetime_day_dummy = pd.get_dummies(data_X['pickup_datetime_day'], prefix='pd', prefix_sep='_')
    # print('pickup_datetime_day_dummy done', pickup_datetime_day_dummy.shape, time.time()-start)
    # pickup_datetime_hour_dummy = pd.get_dummies(data_X['pickup_datetime_hour'], prefix='ph', prefix_sep='_')
    # print('pickup_datetime_hour_dummy done', pickup_datetime_hour_dummy.shape, time.time()-start)
    # pickup_datetime_weekday_dummy = pd.get_dummies(data_X['pickup_datetime_weekday'], prefix='pwd', prefix_sep='_')
    # print('pickup_datetime_weekday_dummy done', pickup_datetime_weekday_dummy.shape, time.time()-start)
    # direction_8_dummy = pd.get_dummies(data_X['direction_8'], prefix='d8', prefix_sep='_')
    # print('direction_8_dummy done', direction_8_dummy.shape, time.time() - start)
    #
    # data_X = pd.concat([data_X,
    #                     vendor_id_dummy,
    #                     passenger_count_dummy,
    #                     store_and_fwd_flag_int_dummy,
    #                     pickup_cluster_dummy,
    #                     dropoff_cluster_dummy,
    #                     pickup_datetime_month_dummy,
    #                     pickup_datetime_day_dummy,
    #                     pickup_datetime_hour_dummy,
    #                     pickup_datetime_weekday_dummy,
    #                     direction_8_dummy], axis=1)
    # print('concat dummy done', time.time()-start)

    data_X = pd.merge(data_X, weather, how='left', on=['pickup_date'])
    print('weather merge done', time.time()-start)
    data_X = pd.merge(data_X, route, how='left', on=['id'])
    print('route merge done', time.time()-start)

    data_X.drop(['store_and_fwd_flag', 'pickup_datetime', 'pickup_date', 'trip_duration'], axis=1, inplace=True)
    if stage != 'test':  # if 'dropoff_datetime' in data_X.keys():
        data_X = data_X.drop(['dropoff_datetime'], axis=1)
    print('drop done', time.time() - start)

    if stage != 'test':
        train_X, validate_X, train_Y, validate_Y = train_test_split(data_X, data_Y, test_size=0.1, random_state=2018)
        return train_X, train_Y, validate_X, validate_Y

    return data_X, data_Y


def extend_feature(data_X):
    data_X.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1, inplace=True)
    data_X.drop(['pickup_longitude_grid', 'pickup_latitude_grid', 'dropoff_longitude_grid', 'dropoff_latitude_grid'], axis=1, inplace=True)
    data_X.drop(['vendor_id', 'passenger_count', 'store_and_fwd_flag_int', 'pickup_cluster', 'dropoff_cluster', 'pickup_datetime_month', 'pickup_datetime_day', 'pickup_datetime_hour', 'pickup_datetime_weekday', 'direction_8'], axis=1, inplace=True)
    return data_X


def prepare_data(train, test, weather, train_route, test_route, kmeans):
    print('prepare_data train')
    train_X, train_Y, validate_X, validate_Y = prepare_data_one(train, weather, train_route, kmeans, stage='train')
    print('prepare_data test')
    test_X, test_Y = prepare_data_one(test, weather, test_route, kmeans, stage='test')
    return train_X, train_Y, validate_X, validate_Y, test_X, test_Y


def main():
    base_dir = '../dataset/nyc_taxi/'
    train, test, kmeans = load_data(base_dir + 'train.csv', base_dir + 'test.csv')
    print('train:', train.shape)
    print(train.head())
    print('test:', test.shape)
    print(test.head())
    print('kmeans:', kmeans)

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

    train_X, train_Y, validate_X, validate_Y, test_X, test_Y = prepare_data(train, test, weather, train_route,
                                                                            test_route, kmeans)
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
