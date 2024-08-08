# -*- coding: utf-8 -*-
"""
## Andrew Greensweight
## CWID: 20009891
## agreensw@stevens.edu
## Course: CPE646 - Pattern Recognition and Classification
## Assignment: Project
"""

from ast import literal_eval 
import itertools
import os
import numpy as np
import pandas as pd

pd.set_option('display.max_columns',100)

'''This python code assumes the steamspy data has been downloaded. This can be
accomplished by running data_acquisition.py or by dragging the folder titled
.Cpe646_Project to the directory where this current file is run.
'''

raw_steamspy_data = pd.read_csv('.Cpe646_Project/raw_steamspy_data.csv')
#print(raw_steamspy_data.head())

#print(raw_steamspy_data.isnull().sum()) -> 1000 (every entry) had a null for the score_rank
'''Removing Score Rank because each row had a null value'''

'''Check if any game titles are labeled 'none' '''
#print(raw_steamspy_data[raw_steamspy_data['name'] == 'none'])-> not applicable, all games had a title

'''Check if any developers are missing'''
#print(raw_steamspy_data[raw_steamspy_data['developer'].isnull()])-> one entry
'''    appid                              name developer publisher  score_rank  \
386  247120  Portal 2 Sixense Perceptual Pack       NaN   Sixense         NaN   

     positive  negative  userscore                  owners  average_forever  \
386       283       237          0  1,000,000 .. 2,000,000              309   

     average_2weeks  median_forever  median_2weeks  price  initialprice  \
386               0               2              0      0             0   

     discount languages genre  ccu  \
386         0   English   NaN    1   

                                                  tags  
386  {'Adventure': 5002, 'Free to Play': 85, 'Actio...  '''

#print(raw_steamspy_data[raw_steamspy_data['genre'].isnull()])
'''Gives appid 241930, 247120, 250820, 900883 with no genre. THese game titles will be removed as well'''

'''Remove columns score_rank since it is null for every entry and userscore since it is 0'''
#raw_steamspy_data.drop(['score_rank','userscore'], axis=1)
#tags = raw_steamspy_data['tags']


#print(tags.apply(parse_tags).head())-> grabs first five tags with the most votes for each game

#owners = raw_steamspy_data['owners']
#print(owners.head())
#print(owners.str.replace(',', '').str.replace(' .. ','-').head())

outdir = '.Cpe646_Project'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
def process_tags(df, export=False):
    if export: 
        
        tag_data = df[['appid', 'tags']].copy()
        
        def parse_export_tags(x):
            x = literal_eval(x)

            if isinstance(x, dict):
                return x
            elif isinstance(x, list):
                return {}
            else:
                raise TypeError('Something other than dict or list found')

        tag_data['tags'] = tag_data['tags'].apply(parse_export_tags)

        cols = set(itertools.chain(*tag_data['tags']))

        for col in sorted(cols):
            col_name = col.lower().replace(' ', '_').replace('-', '_').replace("'", "")

            tag_data[col_name] = tag_data['tags'].apply(lambda x: x[col] if col in x.keys() else 0)

        tag_data = tag_data.drop('tags', axis=1)

        tag_data.to_csv('.Cpe646_Project/steamspy_tag_data.csv', index=False)
        print("Exported tag data to '.Cpe646_Project/steamspy_tag_data.csv'")
        
        
    def parse_tags(x):
        x = literal_eval(x)
        
        if isinstance(x, dict):
            return ';'.join(list(x.keys())[:10])
        else:
            return np.nan
    
    df['tags'] = df['tags'].apply(parse_tags)
    
    # rows with null tags seem to be superseded by newer release, so remove
    df = df[df['tags'].notnull()]
    
    return df


def process(df):
    df = df.copy()
    
    # handle missing values
    df = df[df['developer'].notnull()]
    df = df[df['genre'].notnull()]
    
    # remove unwanted columns
    df = df.drop(['score_rank', 'userscore', 'discount', 'initialprice', 'price', 
                  'average_2weeks', 'median_2weeks'
                  ], axis=1)
    
    # keep top 5 tags, exporting full tag data to file
    df = process_tags(df, export=True)
    
    # reformat owners column
    df['owners'] = df['owners'].str.replace(',', '').str.replace(' .. ', '-')
    
    return df


steamspy_data = process(raw_steamspy_data)
#print(steamspy_data.head())
#print(steamspy_data.isnull().sum())-> now there are no nulls in any field
'''rename columns to make it easier to understand'''
steam_clean = steamspy_data.rename({
    'tags': 'steamspy_tags',
    'positive': 'positive_ratings',
    'negative': 'negative_ratings',
    #'average_2weeks': 'average_playtime_in_last_2weeks', dropping because lots of 0s
    #'median_2weeks': 'median_playtime_in_last_2weeks', dropping because lots of 0s
    'ccu': 'Peak_Concurrent_Users_yesterday',
    'average_forever': 'average_playtime',
    'median_forever': 'median_playtime'
}, axis=1)

steam_clean.to_csv('.Cpe646_Project/steamspy_data_clean.csv')

#print(steam_clean.head())