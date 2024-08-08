# -*- coding: utf-8 -*-
"""
## Andrew Greensweight
## CWID: 20009891
## agreensw@stevens.edu
## Course: CPE646 - Pattern Recognition and Classification
## Assignment: Project
"""

import numpy as np
import pandas as pd
import math
import os
from colorama import Fore, Back, Style
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from random import randint

pd.set_option("display.max_columns", 100)
outdir = '.Cpe646_Project'
if not os.path.exists(outdir):
    os.mkdir(outdir)
#df = pd.read_csv('.Cpe646_Project/steamspy_data_clean.csv')

def remove_non_english(df):
    # keep only rows marked as supporting english
    df = df[df['languages'].str.contains('English')].copy()
    #keep rows which don't contain 3 or more non-ascii characters in succession
    df = df[~df['name'].str.contains('[^\u0001-\u007F]{3,}')]
    # remove english column, now redundant
    df = df.drop('languages', axis=1)
    
    return df

def steamDB_calc_rating(row):
    '''Calculate rating score based on SteamDB method found on this link:
      https://steamdb.info/blog/steamdb-rating/  
      '''
    pos = row['positive_ratings']
    neg = row['negative_ratings']
    
    total_reviews = pos + neg
    average = pos/total_reviews
    score = average - (average*0.5) * 2**(-math.log10(total_reviews + 1))
    return score

def remove_non_games(df):
    #animation tools sold on Steam
    df = df[df['genre'].str.contains('Animation') == False].copy()
    #utilities such as anti-cheat software
    df = df[df['genre'].str.contains('Utilities') == False].copy()
    return df

def remove_zero_playtime(df):
    '''if average playtime or median playtime is zero, remove game entry from list
    '''
    df = df[df['average_playtime'] != 0]
    df = df[df['median_playtime'] != 0]
    return df

def format_genre(df):
    df['genre'] = df['genre'].str.replace(', ', ';').copy()
    return df

def combine_genre_and_tags(row): #df1['combined'] = df1.apply(function, axis=1)
    return row['genre'] + ";" + row['steamspy_tags']

def format_combined(df):
    df['combined'] = df['combined'].str.replace(', ', ';')
    return df

def clean_combined_feature(row): #df1['combined'] = df1.apply(function, axis=1)
    return ' '.join(set(row['combined'].split(';')))

def pre_process():
    '''Pre-process the Steamspy dataset'''
    df = pd.read_csv('.Cpe646_Project/steamspy_data_clean.csv')
    #print('Before preprocess, this is the size {} {}'.format(df.shape[0], df.shape[1]))
    
    #remove non-english games
    df = remove_non_english(df)
    
    #remove non-games
    df = remove_non_games(df)
    
    #remove zero avg or median playtime
    df = remove_zero_playtime(df)
    
    #use mid-point of range of owners for each game
    df['owners'] = df['owners'].str.split('-')
    df['owners'] = df['owners'].apply(lambda x: (int(x[0]) + int(x[1])) // 2)
    
    #format genre column
    df = format_genre(df)
    
    #combine genre and tags
    df['combined'] = df.apply(combine_genre_and_tags, axis=1)
    
    #format new combined column
    df = format_combined(df)
    
    #clean up combined column
    df['combined'] = df.apply(clean_combined_feature, axis=1)
    
    return df
    
def get_title_from_index(df,index):
    return df[df.index == index]["name"].values[0]
def get_index_from_title(df,name):
    return df[df.name == name]['appid'].values[0]

data = pre_process()
tfidf_vectorizer = TfidfVectorizer(strip_accents = 'unicode',stop_words='english')
##Convert series to list format
tfidf = tfidf_vectorizer.fit_transform(list(data['combined']))
##Array mapping from feature integer indices to feature name
#print(tfidf_vectorizer.get_feature_names_out())
##Tf-idf-weighted document-term matrix.
#print(tfidf)

'''Create list of game names and dictionaries for recommended games and similarity scores'''

list_game_name = list(data['name'])
recommended_games_dic = {}
similarities_dic = {}
for index in range(data['combined'].shape[0]): #combined->genre + tags, shape 0 across rows
    #Iterate items in DataFrame
    cosine_similarities = linear_kernel(tfidf[index],tfidf).flatten()
    #-> for each row->tfidf[index], calculates the cosine similarity between the specific row against the entire matrix
    ####The tfidf is a long list for each of the games. It calculates the tfidf for all words in combined column
    
    #argsort returns an array of indices
    related_docs_indices = (-cosine_similarities).argsort()[1:11]

    recommended_games_dic.update({list_game_name[index]:[list_game_name[i] for i in related_docs_indices]})
    similarities_dic.update({list_game_name[index]: [cosine_similarities[i] for i in related_docs_indices]})


df_sim = pd.DataFrame(similarities_dic)
df_sim.reset_index(inplace=True)
df_tfidf = pd.DataFrame(recommended_games_dic)
df_tfidf.reset_index(inplace=True)

'''Uncomment these two lines to generate csv files to see each list of similar games and their similarity values'''
#df_tfidf.to_csv('.Cpe646_Project/df_tfidf.csv')
#df_sim.to_csv('.Cpe646_Project/df_sim.csv')

'''User Interface'''
while True:
    game_user_likes = input(Fore.YELLOW + Style.BRIGHT+"Enter a game title" + Back.MAGENTA + " (Your response is case sensitive)." + Style.RESET_ALL + Fore.YELLOW + Style.BRIGHT + 
                            "\nHere are three example game titles to try:\n"+Style.RESET_ALL+"{}, {}, {}\n>".format(
        df_tfidf.columns[randint(1,len(df_tfidf.columns))], df_tfidf.columns[randint(1,len(df_tfidf.columns))], df_tfidf.columns[randint(1,len(df_tfidf.columns))]))
    
    try:
        filtered_recommends = list(df_tfidf[game_user_likes])
        filtered_similarities = list(df_sim[game_user_likes])
        i = 0
        print("\n")
        print(Fore.YELLOW + Style.BRIGHT + "Top 10 similar games to "+game_user_likes+" are:\n")
        print("Game Title:" + '\t'*3 + "Similarity Score (from 0.0 to 1.0:")
        print("----------------------------------------------")
        for game in filtered_recommends:
            print(Style.RESET_ALL + game + '\t'*3 + '{}'.format(filtered_similarities[i]))
            i += 1
    except KeyError:
        print(Fore.RED + Style.BRIGHT + "\nError! This is not a valid game title in the available dataset. Try again." + Style.RESET_ALL)

        


''' ########################################################################
    #################### Scratch Work Below ################################
    ######################################################################## '''

'''      
Attempt #1 for recommender system using count vectorizer
Input: "Left 4 Dead"
Output:
 Top 10 similar games to Left 4 Dead are:

 Mafia III: Definitive Edition
 Wizard of Legend
 Warhammer 40,000: Inquisitor - Martyr
 NBA 2K18
 Fallout Shelter
 Battlefield V
 Beholder
 For The King
 Steep
 FINAL FANTASY XV WINDOWS EDITION
 Wolfenstein: Youngblood   
 
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['combined'])
cosine_sim = cosine_similarity(count_matrix)
game_user_likes = "Left 4 Dead"
game_index = get_index_from_title(game_user_likes)
similar_games = list(enumerate(cosine_sim[game_index]))
sorted_similar_games = sorted(similar_games, key=lambda x:x[1], reverse=True)[1:]
i=0
print("Top 10 similar games to "+game_user_likes+" are:\n")
for element in sorted_similar_games:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>10:
        break
     
'''

'''Check for missing values in processed data'''
#print('Verify no missing values:')
#print(data.isnull().sum().value_counts())
#print('After preprocess, this is the size {} {}'.format(data.shape[0], data.shape[1]))
#print(data.head())

#df1 = df.copy()
#df1 = format_genre(df1)
#df1['combined'] = df1.apply(combine_genre_and_tags, axis=1)
#df1 = format_combined(df1)
#df1['combined'] = df1.apply(clean_combined_feature, axis=1)
#print(df1["combined"])'''
''' testing combining genre and steamspy_tags and also formatting it
to remove dulplicates of strings genre in tags column 
i.e. genre: 'Action' and also tag "Action"     '''
#df1['genre'] = df1['genre'].str.replace(', ', ';')
#df1['combined'] = df1.apply(demo, axis=1)
#print(df1["combined"].head(10))
#df1['combined'] = df1['combined'].str.replace(', ', ';')
#print("formatted combined: \n", df1["combined"].head(10))
#df1['combined'] = df1.apply(demo2, axis=1)
#print("did this work?: \n",df1["combined"])
#a = df1['combined'].iloc[22]

#print(df1['genre'])
#df1['new_column'] = df1['genre'] + ';' + df1['steamspy_tags']
#print("Combined column: \n", df1['new_column'])
#df1['new_column'] = df1['new_column'].unique()
#print("Combine column with unique?: \n",df1['new_column'])
#lst = df1['new_column'].astype(str).values.tolist()    
#print(lst)


'''testing removing zero playtime'''
#print(df1.shape[0], df1.shape[1])
#df1 = df1[df1['average_playtime'] != 0]
#df1 = df1[df1['median_playtime'] != 0]
#print("df1 shape after removing zero playtime games: {} {}".format(df1.shape[0], df1.shape[1]))

'''testing steamDB_calc_rating'''
#df1['total_ratings'] = df1['positive_ratings'] + df1['negative_ratings']
#df1['rating_ratio'] = df1['positive_ratings'] / df1['total_ratings']
#df1['rating'] = df1.apply(steamDB_calc_rating, axis=1)
#print(df1.head())


'''testing removing non games based on genre containing 'Animation' '''
#print(df1.shape[0], df1.shape[1])
#df1 = df1[df1['genre'].str.contains('Animation') == False]
#print("df1 shape after removing non-games: {} {}".format(df1.shape[0], df1.shape[1]))

#print(df1['genre'].str.contains('Animation').sum())

'''testing combining genre with steamspy tags'''
#df1['new_column'] = df1['genre'].split(';') + ';' + df1['steamspy_tags']
#print(df1['new_column'])
#print(set(df1.genre.unique()))



