#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess, json
from os import walk
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from kmeans import Kmeans
from rocks import ROCKS



def simple_menu(text,options_dict):
    loop = True
    while loop:
        print('\n' + text)
        choice = str(input())
        for (key,response) in options_dict.items():
            if choice.upper() == key.upper():
                loop = False
                if callable(response):
                    response()
                else:
                    return response
        if loop:
            print(f"Invalid input: {choice}")


def input_menu(data_to_collect):
    '''data_to_collect is a dict with {variable_name:question}
    The function returns a dict {variable_name:value}'''
    data_to_return = {}
    loop = True
    while loop:
        for (variable_name,question) in data_to_collect.items():
            print('\n' + question)
            answer = input()
            data_to_return[variable_name] = answer
        print("\nHere is the information you entered:")
        for (variable_name,answer) in data_to_return.items():
            print(f"\t{variable_name} : {answer}")
        if len(data_to_collect) > 1:
            print("Confirm ? [Y/n]")
            choice = input()
            loop = (choice.upper() != 'Y')
        else:
            loop = False
    return data_to_return
        
        

# Start here
def main():
    print("Welcome to Aixon users clustering utility!\n")
    simple_menu("""What do you want to do ?
    D: download a new robot
    F: format data of a downloaded robot (select user id and Keywords/Interests)
    P: pre-process formatted data
    C: apply clustering to a pre-processed robot """,
        {'D':downloadMenu, 'F':selectRobot, 'P':selectData, 'C':selectPreprocessedData})



def launch_download(API_key,url,filename):
    filename = f'download_node_{url.split("/")[-1]}.json'
    subprocess.run(["curl", "-k", "-v", url, "-H", "x-api-key: "+API_key, 
                    "-H", "file-format: json", "-o", filename])


def downloadMenu():
    print("You will now be asked to enter your API key and the download URL of "+\
          "the robot you want to analyse. Be aware that downloading the data can take a long time")
    params = input_menu({'API_key':'Input your API key (you can find it in your aixon account settings).',
        'url':'Input the download URL of the robot (you can find it in the robot page, in "robot destination" click "Copy API link").'})
    API_key, url = params['API_key'], params['url']
    # Download the file
    filename = f"download_node_{url.split('/')[-1]}.json"
    launch_download( API_key, url, filename)
    print("The robot is now downloaded, next step is pre-processing.")
    formatMenu(filename)



def selectRobot():
    directory = input_menu({'directory':'Enter the directory where files are located.'})['directory']
    f, robots = [], []
    for (dirpath, dirnames, filenames) in walk(directory):
        f.extend(filenames); break
    for filename in f:
        if filename[:14] == 'download_node_' and filename[-5:] == '.json':
            robots.append(filename[14:-5])
    print('\nThe following robots are available in that folder:')
    for k,r in enumerate(robots):
        print(f'{k} : {r}')
    r = input_menu({'robot':f'Chose the robot you want to format (0-{len(robots)-1}).'})['robot']
    formatMenu('download_node_'+robots[int(r)]+'.json')


def formatMenu(filename):
    '''Select type of user id and Keywords/Interests to keep'''
    # take a sample line to check which id(s) is present
    with open(filename,'r') as f:
        line = json.loads(f.readline())
    potential_ids = []
    for potential_id in ['customuid','emailsha256','idfa','dmp_id','appier_cookie_id']:
        if line[potential_id] != [] and line[potential_id] != "":
            potential_ids.append(potential_id)
    if len(potential_ids) == 1:
        id_chosen = potential_ids[0]
        print(f'\nOnly {id_chosen} is available to identify users.')
    else:
        options_dict = {str(k):potential_id for k,potential_id in enumerate(potential_ids)}
        for k,potential_id in options_dict.items():
            print(f'{k} : \t{potential_id}')
        id_chosen = simple_menu('The previous ids are available, chose one',options_dict)
    potential_KIs = ["Out of network keywords", "In network keywords", "Preset Interest", "My Interest", "Custom keywords"]
    options_dict = {str(k):potential_ki for k,potential_ki in enumerate(potential_KIs)}
    for k,potential_ki in options_dict.items():
        print(f'{k} : \t{potential_ki}')
    KI_type_chosen = simple_menu('The previous Keywords/Interests are available, chose one', options_dict)
    # actual reformatting
    filename_output = f'data_{filename[14:-5]}-{id_chosen}-{KI_type_chosen}.tsv'
    with open(filename,'r') as fr:
        with open(filename_output,'w') as fw:
            for l in fr:
                line = json.loads(l)
                if len(line[KI_type_chosen]) > 0:
                    #TODO special cases for subdicts [0] or ['interests']
                    fw.write('\n'+line[id_chosen][0]+'\t'+','.join([ir['interest'] for ir in line[KI_type_chosen]]))
    preprocessMenu(filename_output)
    


def selectData():
    '''Select the file of formatted data to pre-process'''
    directory = input_menu({'directory':'Enter the directory where files are located.'})['directory']
    f, datasets = [], []
    for (dirpath, dirnames, filenames) in walk(directory):
        f.extend(filenames); break
    for filename in f:
        if filename[:5] == 'data_' and filename[-4:] == '.tsv':
            datasets.append(filename[5:-4])
    print('The following datasets are available in that folder:')
    for k,r in enumerate(datasets):
        print(f'{k} : {r}')
    r = input_menu({'data':f'Chose the dataset you want to pre-process (0-{len(datasets)-1}).'})['data']
    preprocessMenu('data_'+datasets[int(r)]+'.tsv')



def preprocessMenu(filename):
    '''preprocess the file given in argument.
    Preprocessed data is composed of a file with users data with format
    user id \t interest1, interest2, ... (same format as before preprocessing)
    AND a file with the list of interests to consider'''
    print('''    You will now see the distribution of the popularity of interests and the distribution
    of the number of interests per user. Then you will be asked how you want to truncate the data.
    Press enter to continue.''')
    input()
    ki_count, distribution_interests_users = defaultdict(int), defaultdict(int)
    with open(filename,'r') as fr:
        for line in fr:
            try:
                ki_list = line.strip().split('\t')[1].split(',') # list of k/i's of the user
                distribution_interests_users[len(ki_list)] += 1
                for ki in ki_list:
                    ki_count[ki] += 1
            except: pass
    ki_count, distribution_interests_users = dict(ki_count), dict(distribution_interests_users)
    # graph 1: Y number of users X number of ki/user
    plt.figure(2)
    x,y = np.arange(1+max(distribution_interests_users.keys())), []
    for k in x:
        try: y.append(distribution_interests_users[k])
        except: y.append(0)
    plt.fill_between(x, y, color="chartreuse", alpha=0.5)
    plt.plot(x, y, color="yellowgreen")
    plt.xlabel('number of interests or keywords per user')
    plt.ylabel('number of users')
    plt.ylim(bottom=0)
    plt.show()
    bounds = input_menu({
        'user_ki_lowerbound':'Eliminate users with number of Keywords/Interests strictly lower than ...',
        'user_ki_upperbound':'Eliminate users with number of Keywords/Interests strictly higher than ...'})
    ukilow,ukihigh = int(bounds['user_ki_lowerbound']), int(bounds['user_ki_upperbound'])
    # graph 2: Y number of ki's X popularity
    plt.figure(1)
    # x,y = np.arange(len(ki_count)), [k for k in reversed(sorted(ki_count.values()))] # shit happened here
    ki_distrib_by_pop = defaultdict(int)
    for k in ki_count.values():
        ki_distrib_by_pop[k] += 1
    x,y = np.arange(max(ki_count.values())), []
    for k in x:
        y.append(ki_distrib_by_pop[k])
    plt.fill_between(x, y, color="mediumblue", alpha=0.6)
    plt.plot(x, y, color="darkblue")
    plt.xlabel('popularity')
    plt.ylabel('number of interests or keywords')
    plt.ylim(bottom=0)
    plt.show()
    bounds = input_menu({'interest_pop_lowerbound':'Eliminate interests with popularity strictly lower than ...',
        'interest_pop_upperbound':'Eliminate interests with popularity strictly higher than ...'
        })
    kipoplow,kipophigh = int(bounds['interest_pop_lowerbound']), int(bounds['interest_pop_upperbound'])
    outfilename = f'{filename[5:-4]}_uki{ukilow}-{ukihigh}_kipop{kipoplow}-{kipophigh}'
    with open(filename,'r') as fr:
        with open( 'prepData_' + outfilename + '.tsv', 'w') as fw:
            for line in fr:
                try:
                    ki_list = line.strip().split('\t')[1].split(',')
                    if len(ki_list) >= ukilow and len(ki_list) <= ukihigh :
                        fw.write(line)
                except: pass
    with open( 'prepDataInterests_' + outfilename + '.txt', 'w') as fw:
        for (ki,count) in ki_count.items():
            if count >= kipoplow and count <= kipophigh:
                fw.write( ki + '\n' )
    clusterMenu(outfilename)


def selectPreprocessedData():
    '''Select the file of pre-processed data to cluster'''
    directory = input_menu({'directory':'Enter the directory where files are located.'})['directory']
    f, datasets = [], []
    for (dirpath, dirnames, filenames) in walk(directory):
        f.extend(filenames); break
    for filename in f:
        if filename[:9] == 'prepData_' and filename[-4:] == '.tsv':
            datasets.append(filename[9:-4])
    print('The following preprocessed datasets are available in that folder:')
    for k,r in enumerate(datasets):
        print(f'{k} : {r}')
    r = input_menu({'data':f'Chose the dataset you want to cluster (0-{len(datasets)-1}).'})['data']
    clusterMenu(datasets[int(r)])
    




def clusterMenu(filename):
    '''Apply clustering to the file given in argument'''
    file_data, file_interests = 'prepData_'+filename+'.tsv', 'prepDataInterests_'+filename+'.txt'
    algo_chosen = simple_menu('''What clustering algorithm do you want to use ?
    K : K-means
    R : ROCKS (not available yet)''', {'K':'k-means','R':'rocks'})
    if algo_chosen == 'k-means':
        n_clusters = input_menu({'n_clusters':'How many clusters do you want ?'})['n_clusters']
        # now start clustering
        kmeans_handler = Kmeans(file_data, file_interests)
        clusters = kmeans_handler.k_means( n_clusters=int(n_clusters), n_iter=40 )
        # save data if needed
        print('''Please enter the names of the files in which to save clusters data (leave blank for not saving). The extension ".csv" will be added automaticaly.''')
        exportfilenames = input_menu({'info':'Enter file name for clusters summary data',
                                      'clusters':'Enter file name to save clusters in.'})
        kmeans_handler.write_clusters_info( clusters, exportfilenames['info']+'.csv' )
        kmeans_handler.export_clusters( clusters, exportfilenames['clusters']+'.csv' )
    elif algo_chosen == 'rocks':
        n_clusters = input_menu({'n_clusters':'How many clusters do you want ?'})['n_clusters']
        rocks_instance = ROCKS(file_data, file_interests, 0.5)
        clusters = rocks_instance.rocks(sample_size=100, n_clusters=8)
        
        
        
        
        
        
        
        

    # loop = True
    # while loop:
    #     # restart clustering if user wants to
    #     loop = simple_menu('Do you want to continue working on this data ? (Y/N)',{'Y':True,'N':False})


main()
