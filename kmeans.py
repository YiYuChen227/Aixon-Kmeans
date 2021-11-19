#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections  import defaultdict
from copy import deepcopy
import sys, os, random
import numpy as np



        
def distance(a,b):
    """ Euclidean distance between points a and b"""
    return np.linalg.norm(a-b)


def distance_from_nearest_center(centers,point):
    """ Used for Kmeans++ initialisation 
    Returns the square distance between point and the nearest center
    """
    return np.min(np.sum((centers-point)**2,1))


def get_centroid(points):
    """ Returns the centroid of the list points"""
    return np.mean(points, axis=0)


def load_file(filename):
    """ Loads interests from a 1-column file to a list
    """
    names = []
    with open(filename,'r') as f:
        _ = f.readline()
        for line in f:
            try:
                name = names.append( line.strip() )
            except:
                print('Error reading line:' + line)
    return names


def convert_file_2columns(filename_in, filename_interests):
    ''' Loads data from a 2 columns file 
            dmp_id \t interest1 , interest2 etc.
        and interests file
        Returns a list of interests and a numpy matrix of 
        the interests of each dmp_id
        Needs a few seconds to load
    '''
    list_interests = load_file(filename_interests)
    with open(filename_in,'r') as f:
        print('Starting to load users')
        numpy_mat, list_users = [], []
        for line in f:
            [dmp_id,interests] = line.strip().split('\t')
            interests = interests.split(',')
            numpy_row = np.zeros(len(list_interests))
            for (i,interest) in enumerate(list_interests):
                if interest in interests:
                    numpy_row[i] = 1
            numpy_mat.append(numpy_row)
            list_users.append(dmp_id)
    numpy_mat = np.array(numpy_mat)
    return list_users, list_interests, numpy_mat


class Kmeans:
    def __init__(self, filename_data, filename_interests):
        list_users, list_interests, data = convert_file_2columns(filename_data, filename_interests)
        self.list_users = list_users
        self.list_interests = list_interests
        self.data = data
        
    
    def k_means(self, n_clusters=8, n_iter=12):
        '''
        clusters is a list of indices of the data rows in the cluster
        '''
        # start k-means
        while True:
            # had to add a loop to restart when there are empty clusters,
            # which happens quite often due to k-means not being well adapted for dummy variables (0/1 for interests).
            clusters = [[] for i in range(n_clusters)]
            #centroids = [random.choice(self.data) for i in range(n_clusters)] 
            centroids = self.init_kmeansplusplus(n_clusters)
            for iteration in range(n_iter):
                # 1- reassign each object to the cluster which centroid is the closest
                clusters = [[] for i in range(n_clusters)]
                for index, point in enumerate(self.data):
                    nearest_cluster = np.argmin([distance(point, c) for c in centroids])
                    clusters[nearest_cluster].append(index)
                # 2- update the centroids
                for c in range(n_clusters):
                    centroids[c] = get_centroid([ self.data[index] for index in clusters[c] ])
                # 3- print the size of clusters
                print(f'Iteration {iteration}:')
                for c in range(n_clusters):
                    print(f'\tcluster {c}: {len(clusters[c])} users')
                if min([len(c) for c in clusters]) <= 0:
                    print('Empty cluster, restarting')
                    break # restart if there is an empty cluster
                # if max([len(c) for c in clusters]) >= 0.5 * len(self.data):
                #     print(f'A cluster is too big: {max([len(c) for c in clusters])}'); break
            if iteration >= n_iter - 1: break # exit loop if last iteration
        return clusters
    
    
    def init_kmeansplusplus(self, n_clusters):
        '''Initialisation algorithm Kmeans++ to chose initial cluster centers
        Since there are few clusters but many points, we don't need to go through
        all points, about 100 should be enough to get good starting positions.
        '''
        centers = np.asarray([random.choice(self.data)]) # first center chosen at uniform random
        while len(centers) < n_clusters:
            # select next center
            potential_centers = [random.choice(self.data) for k in range(100)]
            distance_to_centers = [distance_from_nearest_center(centers,point) for point in potential_centers]
            next_center = random.choices(potential_centers,distance_to_centers)
            centers = np.append(centers,next_center,axis=0)
        return centers
    
    
    def write_clusters_info(self, clusters,filename):
        ''' To write results in a file (absolute numbers and percentages)
            1 row/cluster + total, 1 column/interest
            cells contain the sum (or percentage) of users with this interest in this cluster
        '''
        with open(filename,'w') as f:
            f.write('Interests ditribution across clusters (absolute numbers)\n')
            f.write('cluster,total,'+','.join(self.list_interests)+'\n')
            for i, cluster in enumerate(clusters):
                numpy_cluster = np.zeros(len(self.data))
                for index in cluster: numpy_cluster[index] = 1
                machin = np.dot(numpy_cluster,self.data)
                line = str(i)+','+str(len(cluster))
                for num in machin: 
                    line += ',' + str(num)
                f.write(line+'\n')
            f.write(f'total,{len(self.data)},'+','.join([str(n) for n in np.sum(self.data,axis=0)])+'\n')
            f.write('Interests ditribution across clusters (percentage)\n')
            f.write('cluster,size,'+','.join(self.list_interests)+'\n')
            for i, cluster in enumerate(clusters):
                numpy_cluster = np.zeros(len(self.data))
                for index in cluster: numpy_cluster[index] = 1
                machin = np.dot(numpy_cluster,self.data)
                line = str(i)+','+str(len(cluster))
                for num in machin: 
                    line += ',' + str(round(num/len(cluster)*100,1))
                f.write(line+'\n')
            f.write(f'total,{len(self.data)},'+','.join([str(round(n/len(self.data)*100,1)) for n in np.sum(self.data,axis=0)])+'\n')
    
    
    
    def export_clusters(self, clusters,filename):
        ''' Export the lists of users in each cluster to a CSV file
        '''
        with open(filename,'w') as f:
            for i, cluster in enumerate(clusters):
                line = f'cluster {i},'
                for user_index in cluster:
                    line += str(self.list_users[user_index]) + ','
                f.write(line[:-1]+'\n')








