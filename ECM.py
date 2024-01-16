# code to implement Electric Circuit Model (ECM)

# import dependency
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import my_package as mpb
import subprocess
import os
from scipy.sparse import linalg
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import process


# preprocess raw GPS data and calculate velocity and population for each node
# input_file_path: raw GPS data path
# output_file_folder: folder to store output of velocity and population
# unit of resolution: km (smallest resolution: 0.5)

# define project file path
PROJECT_PREFIX = 'Your project file path'

def preprocess_data(input_file_path, output_file_folder, resolution=0.5, show=False):
    if (os.path.exists(input_file_path) == False) | (os.path.exists(output_file_folder) == False):
        print('file does not exist')
        raise
    # load data
    data = pd.read_csv(input_file_path)
    data_len = len(data)
    # file save name
    save_name = input_file_path[-12:-4]
    # normalization
    # population of Japan: 124490000
    # remove people without velocity
    data = data[data['speed'] != 0]
    data = data[data['speed'].isnull() == False]
    # transfrom hour and minute to second
    data['second'] = ((data['hour'] * 60) + data['minute']) * 60
    # transform time to km/h and decompose to speed_x and speed_y
    data['speed'] = data['speed'] * 3.6
    data = data[data['speed'] <= 320]
    # divide by the total population in 2020 in Japan
    #data['speed'] = data['speed'] / normalize_population
    # calculate the x-y component of velocity of a user
    data['course_x'] = data['course'].apply(lambda x : math.sin(np.round(math.radians(x), 2)))
    data['course_y'] = data['course'].apply(lambda x : math.cos(np.round(math.radians(x), 2)))
    data['speed_x'] = data['course_x'] * data['speed']
    data['speed_y'] = data['course_y'] * data['speed']
    # data.sort_values(by = ['mesh_500', 'dailyid', 'second'], inplace=True)
    # remove blank values
    data = data[data['speed_x'].isnull() == False]
    #data = data[(data['speed_x'] != 0) | (data['speed_y'] != 0)]
    # convert the meshID to unified id_x, id_y, and id_str (basic size: 0.5 km) 
    data['id_x' + '_' + str(int(resolution * 1000))] = mpb.longitude_to_id_x(data['longitude'], resolution=resolution)
    data['id_y' + '_' + str(int(resolution * 1000))] = mpb.latitude_to_id_y(data['latitude'], resolution=resolution)
    data['id' + '_' + str(int(resolution * 1000))] = data['id' + '_x_' + str(int(resolution * 1000))].astype(str) + ',' + data['id' + '_y_' + str(int(resolution * 1000))].astype(str)
    # traverse each time peroid
    # 5 a.m. = 18000 second, 24 p.m. = 86400 second, 30 minute = 1800 second
    for index, second in enumerate(range(0, 86400, 1800)):
        hour = int(second / 3600)
        minute = int((second % 3600) / 60)
        # filter out the data in necessary time interval
        sub_data = data[(data['second'] >= second) & (data['second'] <= second + 1800)]
        # initialize dataframe to store result
        preprocess_result = pd.DataFrame()
        preprocess_result['id' + '_' + str(int(resolution * 1000))] = sub_data['id' + '_' + str(int(resolution * 1000))].unique()
        # calculate the mean velocity of a grid
        user_count = sub_data['dailyid'].value_counts().reset_index()
        user_count.columns = ['dailyid', 'count']
        sub_data = pd.merge(sub_data, user_count, on='dailyid', how='left')
        # calculate the population in a grid
        population = sub_data.groupby('id' + '_' + str(int(resolution * 1000)))['dailyid'].nunique().reset_index()
        population.columns = ['id' + '_' + str(int(resolution * 1000)), 'population']
        # calculate speed x in a grid
        speed_x_mean = sub_data.groupby(['id' + '_' + str(int(resolution * 1000)), 'dailyid'])['speed_x'].mean().reset_index()
        speed_x_mean.columns = ['id' + '_' + str(int(resolution * 1000)), 'dailyid', 'speed_x_mean']
        n_apperance = sub_data.groupby(['id' + '_' + str(int(resolution * 1000)), 'dailyid'])['speed_x'].nunique().reset_index()
        n_apperance.columns = ['id' + '_' + str(int(resolution * 1000)), 'dailyid', 'n_apperance']
        speed_x_mean = pd.merge(speed_x_mean, n_apperance, on=['id' + '_' + str(int(resolution * 1000)), 'dailyid'], how='left')
        speed_x_mean = pd.merge(speed_x_mean, user_count, on='dailyid', how='left')
        speed_x_mean['speed_x_mean_idv'] = speed_x_mean['speed_x_mean'] * (speed_x_mean['n_apperance'] / speed_x_mean['count'])
        speed_x_sum = speed_x_mean.groupby(['id' + '_' + str(int(resolution * 1000))])['speed_x_mean_idv'].sum().reset_index()
        speed_x_sum.columns = ['id' + '_' + str(int(resolution * 1000)), 'speed_x_sum']
        # calculate speed y in a grid
        speed_y_mean = sub_data.groupby(['id' + '_' + str(int(resolution * 1000)), 'dailyid'])['speed_y'].mean().reset_index()
        speed_y_mean.columns = ['id' + '_' + str(int(resolution * 1000)), 'dailyid', 'speed_y_mean']
        n_apperance = sub_data.groupby(['id' + '_' + str(int(resolution * 1000)), 'dailyid'])['speed_y'].nunique().reset_index()
        n_apperance.columns = ['id' + '_' + str(int(resolution * 1000)), 'dailyid', 'n_apperance']
        speed_y_mean = pd.merge(speed_y_mean, n_apperance, on=['id' + '_' + str(int(resolution * 1000)), 'dailyid'], how='left')
        speed_y_mean = pd.merge(speed_y_mean, user_count, on='dailyid', how='left')
        speed_y_mean['speed_y_mean_idv'] = speed_y_mean['speed_y_mean'] * (speed_y_mean['n_apperance'] / speed_y_mean['count'])
        speed_y_sum = speed_y_mean.groupby(['id' + '_' + str(int(resolution * 1000))])['speed_y_mean_idv'].sum().reset_index()
        speed_y_sum.columns = ['id' + '_' + str(int(resolution * 1000)), 'speed_y_sum']
        # merge the mean velocity
        preprocess_result = pd.merge(preprocess_result, speed_x_sum, on=['id' + '_' + str(int(resolution * 1000))], how='left')
        preprocess_result = pd.merge(preprocess_result, speed_y_sum, on=['id' + '_' + str(int(resolution * 1000))], how='left')
        preprocess_result = pd.merge(preprocess_result, population, on=['id' + '_' + str(int(resolution * 1000))], how='left')
        preprocess_result.to_csv(output_file_folder + save_name + '-' + str(hour) + 
                                 '-' + str(minute) + '-preprocessed.csv', index=False)
        
# get preprocess task dict
# weekday: weekday data in dependency file
# unit of resolution: km (available value: 0.5, 1, 2, 4, 8, and 16)
# city: 'Tokyo', 'Osaka', 'Fukuoka', 'Nagoya', or 'All'
# year: 2022
# mode: FTO (four to one) or NTO (nine to one)
def get_preprocess_task(weekday_list, resolution_list, city_list, year_list, mode):
    task = {'input_file_path': [], 'output_file_folder': [], 'resolution': [],
           'city': [], 'year': [], 'mode': []}
    for weekday in list(weekday_list):
        for resolution in resolution_list:
            for city in city_list:
                for year in year_list:
                    # create input file path
                    if city == 'All':
                        prefix = '/home/gps_data/' + str(year) + '/' + str(year) + str(weekday[8:10]) + '/'
                    else:
                        prefix = '/home/zhong/EC_to_Gravity/data/raw_material/'+ str(city) + '/' + str(year) + '/'
                    input_file_path = prefix + weekday
                    if (os.path.exists(input_file_path) == False):
                        raise Exception('input file path: ' + input_file_path + ' do not exist')
                    # create output file path
                    output_file_folder = PROJECT_PREFIX + 'data/renormalization_' + mode + '/' + str(city) + '/' + str(year) + '/' + str(int(resolution * 1000)) + '/velocity_population/'
                    if (os.path.exists(output_file_folder) == False):
                        print('output file folder: ' + output_file_folder + ' do not exist')
                        raise
                    task['input_file_path'].append(input_file_path)
                    task['output_file_folder'].append(output_file_folder)
                    task['resolution'].append(resolution)
                    task['city'].append(city)
                    task['year'].append(year)
                    task['mode'].append(mode)
    return task

# commence the multiprocess parallel computation
# preprocess_task: result of dict from get_preprocess_task
# max_workers: how much CPU used in parallel computation
def do_preprocess_task(preprocess_task, max_workers):
    future_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as process_pool:
        for i in range(len(preprocess_task[list(preprocess_task.keys())[0]])):
            future = process_pool.submit(preprocess_data, preprocess_task['input_file_path'][i], 
                                         preprocess_task['output_file_folder'][i], 
                                         preprocess_task['resolution'][i], 
                                         preprocess_task['city'][i])
            future_list.append(future)
    return future_list

# resoution is only for ID setting
def cal_current(input_file_path, resolution=0.5):
    # load data
    data = pd.read_csv(input_file_path)
    ID = 'id_' + str(int(resolution * 1000))
    # parse string
    path_prefix = input_file_path.split('/')[-1]
    date = path_prefix.split('-')[0]
    hour = int(path_prefix.split('-')[1])
    minute = int(path_prefix.split('-')[2])
    second = hour * 3600 + minute * 60
    time = int(date + str(second))
    # copy dataframe
    temp_up = data.copy()
    temp_right = data.copy()
    # calculate right and up node ID
    id_up = data[ID].apply(lambda x:x.split(',')[0] + ',' + str(int(x.split(',')[1]) + 1))
    id_right = data[ID].apply(lambda x:str(int(x.split(',')[0]) + 1) + ',' + x.split(',')[1])
    # merge dataframe
    data = pd.concat([data, id_up], axis=1)
    data = pd.concat([data, id_right], axis=1)
    data.columns = [ID, 'speed_x_sum', 'speed_y_sum', 'population', 'id_up', 'id_right']
    temp_right.columns = ['id_right', 'speed_x_sum', 'speed_y_sum', 'population']
    temp_up.columns = ['id_up', 'speed_x_sum', 'speed_y_sum', 'population']
    current_right = pd.merge(data, temp_right, on='id_right', how='left')
    current_up = pd.merge(data, temp_up, on='id_up', how='left')
    # calculate right current
    current_right['current'] = ((current_right['speed_x_sum_x'] + 
                                current_right['speed_x_sum_y'])  / 2)
    current_right = current_right[[ID, 'current']]
    current_right['direction'] = 'right'
    current_right = current_right[current_right['current'].isnull() == False]
    # calculate up current
    current_up['current'] = ((current_up['speed_y_sum_x']  + 
                             current_up['speed_y_sum_y']) / 2)
    current_up = current_up[[ID, 'current']]
    current_up['direction'] = 'up'
    current_up = current_up[current_up['current'].isnull() == False]
    current = pd.concat([current_right, current_up])
    # add time information
    normalize_population = 2957390 / 124490000
    current['current'] = current['current'] / normalize_population
    current['second'] = second
    current['date'] = date
    current['date'] = current['date'].astype(str)
    return current

# calculate current time series
def cal_CTS(weekday_list, city, year, mode, resolution=0.5):
    # check file integrity
    for weekday in weekday_list:
        for index, second in enumerate(range(18000, 86400, 1800)):
            hour = int(second / 3600)
            minute = int((second % 3600) / 60)
            # create input file path
            prefix = PROJECT_PREFIX + 'data/renormalization_' + str(mode) + '/'+ str(city) + '/' + str(year) + '/' + str(int(resolution * 1000)) + '/velocity_population/'
            suffix = str(weekday) + '-' + str(hour) + '-' + str(minute) + '-preprocessed.csv'
            input_file_path = prefix + suffix
            # check path
            mpb.path_exist(input_file_path)
    output_file_path = prefix[:prefix.find(prefix.split('/')[-2])] + 'current/current_time_series-' + str(int(int(weekday) / 100)) + '.csv'
    # calculate current time series
    current_time_series = pd.DataFrame() 
    for weekday in weekday_list:
        for index, second in enumerate(range(18000, 86400, 1800)):
            hour = int(second / 3600)
            minute = int((second % 3600) / 60)
            # create input file path
            prefix = PROJECT_PREFIX + 'data/renormalization_' + str(mode) + '/'+ str(city) + '/' + str(year) + '/' + str(int(resolution * 1000)) + '/velocity_population/'
            suffix = str(weekday) + '-' + str(hour) + '-' + str(minute) + '-preprocessed.csv'
            input_file_path = prefix + suffix
            # calculate current time series
            current = cal_current(input_file_path, resolution=resolution)
            current_time_series = pd.concat([current_time_series, current])
    current_time_series.to_csv(output_file_path, index=False)

# get multi process task
def get_current_task(month_list, resolution_list, city_list, year_list, mode):
    weekday = pd.read_csv('/home/zhong/Bridge/dependency/weekday_data/weekday.csv')
    weekday['weekday'] = weekday['weekday'].apply(lambda x:x[4:-4])
    task = {'month': [], 'weekday': [], 'resolution': [], 'city': [], 'year': [], 'mode': []}
    for month in month_list:
        for resolution in resolution_list:
            for city in city_list:
                for year in year_list:
                    task['weekday'].append(weekday[weekday['month'] == month]['weekday'].values)
                    task['month'].append(month)
                    task['resolution'].append(resolution)
                    task['city'].append(city)
                    task['year'].append(year)
                    task['mode'].append(mode)
    return task

# run cal_CTS on multi process
def do_current_task(current_task, max_workers):
    future_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as process_pool:
        for i in range(len(current_task[list(current_task.keys())[0]])):
            future = process_pool.submit(cal_CTS, 
                                         current_task['weekday'][i], 
                                         current_task['resolution'][i], 
                                         current_task['city'][i], 
                                         current_task['year'][i], 
                                         current_task['mode'][i])
            future_list.append(future)
    return future_list

# calculate resistance
def cal_resistance(month_list, city, year, mode, time_peroid=[18000, 86400], resolution=0.5):
    CTS = pd.DataFrame()
    for month in month_list:
        input_file_path = (PROJECT_PREFIX + 'data/renormalization_' + str(mode) + 
                     '/' + str(city) + '/' + str(year) + '/' + str(int(resolution * 1000)) + 
                     '/current/current_time_series-' + str(year) + str(month) + '.csv')
        mpb.path_exist(input_file_path)
    for month in month_list:
        input_file_path = (PROJECT_PREFIX + 'data/renormalization_' + str(mode) + 
                     '/' + str(city) + '/' + str(year) + '/' + str(int(resolution * 1000)) + 
                     '/current/current_time_series-' + str(year) + str(month) + '.csv')
        temp = pd.read_csv(input_file_path)
        CTS = pd.concat([CTS, temp])
    # add geometry information
    CTS = mpb.add_geometry(CTS, resolution=resolution)
    # limit the map range
    if city == 'Tokyo':
        CTS = CTS[(CTS['longitude_' + str(int(resolution * 1000))] >= 139.15) & 
                  (CTS['longitude_' + str(int(resolution * 1000))] <= 141) & 
                  (CTS['latitude_' + str(int(resolution * 1000))] >= 34.8) & 
                  (CTS['latitude_' + str(int(resolution * 1000))] <= 36.3)]
    del CTS['longitude_' + str(int(resolution * 1000))]
    del CTS['latitude_' + str(int(resolution * 1000))]
    # limit time peroid
    CTS = CTS[(CTS['second'] >= time_peroid[0]) & (CTS['second'] <= time_peroid[1])]
    # calculate resistance and voltage
    CTS['abs_current'] = np.abs(CTS['current'])
    resistance = CTS.groupby(['id_' + str(int(resolution * 1000)), 'direction'])['abs_current'].sum().reset_index()
    resistance.columns.values[-1] = 'conductivity'
    resistance['conductivity'] /= (30 * len(month_list) * int((time_peroid[1] - time_peroid[0]) / 1800))
    resistance['resistance'] = 1 / resistance['conductivity']
    CTS = pd.merge(CTS, resistance, on=[resistance.columns[0], 'direction'], how='left')
    CTS['E'] = CTS['current'] * CTS['resistance']
    # save result to csv
    resistance_output_file_path = (PROJECT_PREFIX + 'data/renormalization_' + str(mode) + 
                     '/' + str(city) + '/' + str(year) + '/' + str(int(resolution * 1000)) + 
                     '/resistance/resistance_' + month_list[0] + '-' + month_list[-1] +  '.csv')
    voltage_output_file_path = (PROJECT_PREFIX + 'data/renormalization_' + str(mode) + 
                     '/' + str(city) + '/' + str(year) + '/' + str(int(resolution * 1000)) + 
                     '/resistance/voltage_' + month_list[0] + '-' + month_list[-1] +  '.csv')
    resistance.to_csv(resistance_output_file_path, index=False)
    CTS.to_csv(voltage_output_file_path, index=False)
