# -*- coding: utf-8 -*-
"""
## Andrew Greensweight
## CWID: 20009891
## agreensw@stevens.edu
## Course: CPE646 - Pattern Recognition and Classification
## Assignment: Project
"""

'''Data acquisition: 
This code is to interface with the SteamSpy API which is
a Steam stats-gathering service. The documentation is found here:
    https://steamspy.com/api.php
    
    Return format:
  * appid - Steam Application ID. If it's 999999, then data for this applicati-
  on is hidden on developer's request, sorry.
  * name - game's name
  * developer - comma separated list of the developers of the game
  * publisher - comma separated list of the publishers of the game
  * score_rank - score rank of the game based on user reviews
  * owners - owners of this application on Steam as a range.
  * average_forever - average playtime since March 2009. In minutes.
  * average_2weeks - average playtime in the last two weeks. In minutes.
  * median_forever - median playtime since March 2009. In minutes.
  * median_2weeks - median playtime in the last two weeks. In minutes.
  * ccu - peak CCU yesterday.
  * price - current US price in cents.
  * initialprice - original US price in cents.
  * discount - current discount in percents.
  * tags - game's tags with votes in JSON array.
  * languages - list of supported languages.
  * genre - list of genres.'''
  
import csv
import json
import os
import time
import numpy as np
import pandas as pd
import requests

pd.set_option("display.max_columns", 100) #tables show all columns

def get_request(url, parameters=None):

    """Return json-formatted response of get request with optional parameters.
url --> string
parameters: {'parameter': 'value'}
    
Returns
json_data
json-formatted response 
"""
    try:
        response = requests.get(url=url, params=parameters)
    except requests.exceptions.SSLError as s: 
        print('SSL Error: ', s)
        
        for i in range(5,0,-1):
            print('\rWaiting... ({})'.format(i), end='')
            time.sleep(1)
        print('Retrying...' + ' '*10)
        
        return get_request(url,parameters)
    
    if response:
        return response.json()
    else:
        print('No response, waiting 10 seconds...')
        time.sleep(10)
        print('Retrying...')
        return get_request(url,parameters)

url = "https://steamspy.com/api.php"      
parameters = {"request": "all"}

''' Request all from SteamSpy and parse into pandas data frame'''
json_data = get_request(url,parameters)
steam_spy = pd.DataFrame.from_dict(json_data, orient='index')

game_list = steam_spy[['appid', 'name']].sort_values('appid').reset_index(drop=True)

outname = 'game_list.csv' #makes destination for csv file wherever the python code is executed
outdir = '.Cpe646_Project'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
fullname = os.path.join(outdir, outname)

if not os.path.exists(fullname):
    game_list.to_csv(fullname, index=False)

game_list = pd.read_csv(fullname)
print('Verify API request for Appids is working:\n',game_list.head())

print('------------\n')

def get_game_data(start, stop, parser, pause):
    '''Return list of game data generated from parser, which is a function to 
    handle the requests
    '''
    game_data = []
    #iterate thru each row of the game_list
    for idx, row in game_list[start:stop].iterrows():
        print('Current index: {} |'.format(idx), end='\r')
        
        gameid = row['appid']
        name = row['name']
        #retrieve game data for a row handled by parser and add to list game data
        
        data = parser(gameid, name)
        game_data.append(data)
        #to prevent too many API requests
        time.sleep(pause) 
        
    return game_data

def process_batches(parser, game_list, dl_path, data_filename, index_filename,
                    columns, begin=0, end=-1, batchsize=100, pause=1):
    
    
    '''Process game data in batches then write directly to a file
    parser = function to format request
    game_list = dataframe of appid (gameid) and name
    dl_path = where to store data
    data_filename = name of file we will save game app data
    index_filename = file to store highest index written
    columns = columns for file
    begin -> starting index
    end -> index to finish
    batchsize -> number of games to write in each batch 
    pause -> time to wait to prevent to many API requests
    
    returns -> none
    
    inspiration:
     https://stackoverflow.com/questions/16982569/making-multiple-api-calls-in-parallel-using-python-ipython
     &
     https://towardsdatascience.com/json-and-apis-with-python-fba329ef6ef0
    '''
    print("Starting index: {}".format(begin))
    
    if end == -1 :
        end = len(game_list) + 1
        
    batches = np.arange(begin, end, batchsize)
    batches = np.append(batches, end)
    
    games_written = 0
    
    for i in range(len(batches) - 1):
        start = batches[i]
        stop = batches[i+1]
        game_data = get_game_data(start,stop,parser,pause)
        
        relative_path = os.path.join(dl_path, data_filename)
            
        #writing game_data to file
        with open(relative_path, 'a', newline='', encoding='utf-8') as f:
            write = csv.DictWriter(f,fieldnames=columns, extrasaction='ignore')
            
            for j in range(3,0,-1):
                print("\r preparing to write.. ({})".format(j),end='')
                time.sleep(0.5)
            
            write.writerows(game_data)
            print('\rExtracted lines {}-{} to the following file {}'.format(start, stop -1, data_filename))
            
        games_written += len(game_data)
        
        index_path = os.path.join(dl_path, index_filename)
        
        #document last index
        with open(index_path, 'w') as f:
            index = stop
            print(index, file=f)
            
def reset_idx(dl_path, idx_filename):
    """Reset index in file to 0. This can restart download process"""
    relative_path = os.path.join(dl_path, idx_filename)
    
    with open(relative_path, 'w') as f:
        print(0, file=f)
        

def get_idx(dl_path, idx_filename):
    """Get the index from a file and return 0 if file not found
    Ensures if there is an error in the download, the last index from the
    file can be read and we can continue from there with each batch of data
    requested
    """
    try:
        relative_path = os.path.join(dl_path, idx_filename)

        with open(relative_path, 'r') as f:
            index = int(f.readline())
    
    except FileNotFoundError:
        index = 0
        
    return index


def prepare_data_file(dl_path, filename, index, columns):
    """Prepares the csv for storing the data. If the index retrieved is 0 we 
    are either starting from scratch or starting over. In both situations
    we will want a blank csvCreate file and write headers if 
    index is 0."""
    if index == 0:
        relative_path = os.path.join(dl_path, filename)

        with open(relative_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

def parse_steamspy_request(appid,name):
    '''The data returned and maximum polling rate of the API is documented here:
        https://steamspy.com/api.php -this will make use of the proccesing
        the game data in batches
        '''
    url = 'https://steamspy.com/api.php'
    parameters = {"request": "appdetails", "appid": appid}
    
    json_data = get_request(url, parameters)
    return json_data

outdir = '.Cpe646_Project'
if not os.path.exists(outdir):
    os.mkdir(outdir)

dl_path = outdir
steamspy_data = 'raw_steamspy_data.csv'
steamspy_idx = 'steamspy_index.txt'
rel_path_steamspy = os.path.join(dl_path,steamspy_data)

steamspy_columns = ['appid', 'name', 'developer', 'publisher', 'score_rank', 'positive',
    'negative', 'userscore', 'owners', 'average_forever', 'average_2weeks',
    'median_forever', 'median_2weeks', 'price', 'initialprice', 'discount',
    'languages', 'genre', 'ccu', 'tags']

reset_idx(dl_path, steamspy_idx)
index = get_idx(dl_path, steamspy_idx)

#start fresh if index = 0 
prepare_data_file(dl_path, steamspy_data, index, steamspy_columns)


process_batches( parser=parse_steamspy_request,
    game_list=game_list,
    dl_path=dl_path, 
    data_filename=steamspy_data,
    index_filename=steamspy_idx,
    columns=steamspy_columns,
    begin=index,
    end=1001,
    batchsize=20,
    pause=0.2 )

print('Check if steamspy data downloaded correctly:\n', pd.read_csv(rel_path_steamspy).head())

        