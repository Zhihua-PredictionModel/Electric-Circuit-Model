import pandas as pd
import numpy as np
import os

# for the map created by Koike Hajime
# available parameter size: 0.5, 1, 2, 4, 8, 16(km)
# function to transfrom longitude to relative coordinate x
def longitude_to_id_x(longitude, resolution):
    if resolution == 0.5:
        min_longitude = 122.934560
        min_latitude = 20.426987
    elif resolution == 1:
        min_longitude = 122.937307
        min_latitude = 20.429237
    elif resolution == 2:
        min_longitude = 122.942800
        min_latitude = 20.433737
    elif resolution == 4:
        min_longitude = 122.953787
        min_latitude = 20.442737
    elif resolution == 8:
        min_longitude = 122.975761
        min_latitude = 20.460738
    elif resolution == 16:
        min_longitude = 123.019709
        min_latitude = 20.496737
    else:
        print('resolution is not available')
        raise
    longitude_width = resolution / ((40000 / 360) * np.cos((35 / 180) * np.pi)) 
    latitude_width = resolution / (40000 / 360)
    id_x = ((longitude - min_longitude) / longitude_width + 0.5).astype(int)
    #id_y = int((latitude - min_latitude) / latitude_width + 0.5)
    return id_x

# function to transfrom latitude to relative coordinate y
def latitude_to_id_y(latitude, resolution):
    if resolution == 0.5:
        min_longitude = 122.934560
        min_latitude = 20.426987
    elif resolution == 1:
        min_longitude = 122.937307
        min_latitude = 20.429237
    elif resolution == 2:
        min_longitude = 122.942800
        min_latitude = 20.433737
    elif resolution == 4:
        min_longitude = 122.953787
        min_latitude = 20.442737
    elif resolution == 8:
        min_longitude = 122.975761
        min_latitude = 20.460738
    elif resolution == 16:
        min_longitude = 123.019709
        min_latitude = 20.496737
    else:
        print('resolution is not available')
        raise
    longtitude_width = resolution / ((40000 / 360) * np.cos((35 / 180) * np.pi)) 
    latitude_width = resolution / (40000 / 360)
    #id_x = int((longitude - min_longitude) / longtitude_width + 0.5)
    id_y = ((latitude - min_latitude) / latitude_width + 0.5).astype(int)
    return id_y

# function to transform relative coordinate x to longitude
def id_x_to_longitude(id_x, resolution):
    if resolution == 0.5:
        min_longitude = 122.934560
        min_latitude = 20.426987
    elif resolution == 1:
        min_longitude = 122.937307
        min_latitude = 20.429237
    elif resolution == 2:
        min_longitude = 122.942800
        min_latitude = 20.433737
    elif resolution == 4:
        min_longitude = 122.953787
        min_latitude = 20.442737
    elif resolution == 8:
        min_longitude = 122.975761
        min_latitude = 20.460738
    elif resolution == 16:
        min_longitude = 123.019709
        min_latitude = 20.496737
    else:
        print('resolution is not available')
        raise
    longitude_width = resolution / ((40000 / 360) * np.cos((35 / 180) * np.pi)) 
    latitude_width = resolution / (40000 / 360)
    longitude = id_x * longitude_width + min_longitude
    return longitude

# function to transform relative coordinate y to latitude
def id_y_to_latitude(id_y, resolution):
    if resolution == 0.5:
        min_longitude = 122.934560
        min_latitude = 20.426987
    elif resolution == 1:
        min_longitude = 122.937307
        min_latitude = 20.429237
    elif resolution == 2:
        min_longitude = 122.942800
        min_latitude = 20.433737
    elif resolution == 4:
        min_longitude = 122.953787
        min_latitude = 20.442737
    elif resolution == 8:
        min_longitude = 122.975761
        min_latitude = 20.460738
    elif resolution == 16:
        min_longitude = 123.019709
        min_latitude = 20.496737
    else:
        print('resolution is not available')
        raise
    longitude_width = resolution / ((40000 / 360) * np.cos((35 / 180) * np.pi)) 
    latitude_width = resolution / (40000 / 360)
    latitude = id_y * latitude_width + min_latitude
    return latitude

# add longitude and latitude to dataframe when id is given    
def add_geometry(data, resolution, duplicate=False):
    data = data.copy()
    resolution_index = str(int(resolution * 1000))
    data['id_x_' + resolution_index] = data['id_' + resolution_index].apply(lambda x:x.split(',')[0]).astype(int)
    data['id_y_' + resolution_index] = data['id_' + resolution_index].apply(lambda x:x.split(',')[1]).astype(int)
    data['longitude_' + resolution_index] = id_x_to_longitude(data['id_x_' + resolution_index], resolution=resolution)
    data['latitude_' + resolution_index] = id_y_to_latitude(data['id_y_' + resolution_index], resolution=resolution)
    if duplicate != False:
        data['longitude'] = data['longitude_' + resolution_index]
        data['latitude'] = data['latitude_' + resolution_index]
    return data

# convert longitude and latitude to id according to resolution 
def lon_lat_to_id(longitude, latitude, resolution):
    id_x = longitude_to_id_x(np.array(longitude), resolution=resolution)
    id_y = latitude_to_id_y(np.array(latitude), resolution=resolution)
    return str(id_x) + ',' + str(id_y)
        
# get four direction node ID of current node
def right_node(current_id, split_symbol=','):
    return str(int(current_id.split(split_symbol)[0]) + 1) + split_symbol + current_id.split(split_symbol)[1]

def left_node(current_id, split_symbol=','):
    return str(int(current_id.split(split_symbol)[0]) - 1) + split_symbol + current_id.split(split_symbol)[1]

def up_node(current_id, split_symbol=','):
    return current_id.split(split_symbol)[0] + split_symbol + str(int(current_id.split(split_symbol)[1]) + 1)

def down_node(current_id, split_symbol=','):
    return current_id.split(split_symbol)[0] + split_symbol + str(int(current_id.split(split_symbol)[1]) - 1)
