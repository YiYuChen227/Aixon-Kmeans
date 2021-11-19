#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
ROCK clustering algorithm
example implementation at https://pyclustering.github.io/docs/0.10.1/html/d8/dec/rock_8py_source.html
'''
from collections  import defaultdict, deque
from copy import deepcopy
import sys, os, random
import numpy as np
import operator



        
def distance(a,b):
    """Euclidean distance between points a and b"""
    return np.linalg.norm(a-b)


def get_centroid(points):
    """Returns the centroid of the list points"""
    return np.mean(points, axis=0)


def Jaccard_similarity(a,b):
    '''Returns the Jaccard similarity between 2 points a and b
    a, b being numpy arrays of booleans'''
    return sum(a&b)/sum(a|b)


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
        Different from the same functions for k-means: here the matrix is boolean
    '''
    list_interests = load_file(filename_interests)
    with open(filename_in,'r') as f:
        print('Starting to load users')
        numpy_mat, list_users = [], []
        for line in f:
            [dmp_id,interests] = line.strip().split('\t')
            interests = interests.split(',')
            numpy_row = np.zeros(len(list_interests),dtype=bool)
            for (i,interest) in enumerate(list_interests):
                if interest in interests:
                    numpy_row[i] = True
            numpy_mat.append(numpy_row)
            list_users.append(dmp_id)
    numpy_mat = np.array(numpy_mat)
    return list_users, list_interests, numpy_mat


class ROCKS:
    def __init__(self, filename_data, filename_interests, threshold, sample_size=100):
        self.threshold = threshold
        self.dnorm = 1.0 + 2.0 * ( (1.0 - self.threshold) / (1.0 + self.threshold) ) # degree normalisation
        list_users, list_interests, data = convert_file_2columns(filename_data, filename_interests)
        self.list_users = list_users
        self.list_interests = list_interests
        # Divide between sample data and main data
        choice = np.random.choice(data.shape[0], sample_size, replace=False)
        sample = np.zeros(data.shape[0], dtype=bool); sample[choice] = True
        self.data          = data[sample]
        self.sampled_data  = data[~sample]
        self.sampled_links = None
        
    def is_neighbour(self, a, b):
        return Jaccard_similarity(a,b) >= self.threshold
    
    def adjacency_matrix(self, data):
        print('Building adjacency matrix')
        n = len(data)
        A = np.zeros((n, n), dtype=bool) # Adjacency matrix
        for i in range(n):
            for j in range(i+1,n):
                if self.is_neighbour(data[i], data[j]):
                    A[i,j], A[j,i] = True, True
        return A
    
    
    def n_links(self, C1, C2):
        ''' Number of links between 2 clusters '''
        number_links = 0
        for p1 in C1:
            for p2 in C2:
                number_links += self.sampled_links[p1,p2]
        return number_links
    
    
    def goodness(self, C1, C2):
        denominator = (len(C1)+len(C2))**self.dnorm - len(C1)**self.dnorm - len(C2)**self.dnorm
        return self.n_links(C1, C2) / denominator
    
    
    def rocks(self, n_clusters=8, n_iter=12):
        '''
        clusters is a list of indices of the data rows in the cluster
        sample_size is the number of points used for initial clustering
        '''
        # initializations
        A = self.adjacency_matrix(self.sampled_data)
        A = np.array(A,dtype=int) # convert A to numbers to get links matrix
        self.sampled_links = A.dot(A); del A   # links gives links between points
        clusters = [[index] for index in range(len(self.sampled_data))] # clusters are lists of indexes of data points
        while len(clusters) > n_clusters:
            [c1,c2] = self.find_pair_clusters(clusters)
            if [c1,c2] != [-1,-1]:
                clusters[c1] += clusters[c2]
                _ = clusters.pop(c2)
            else: 
                break # totally separate clusters
        # now clusters is the list of lists of data points
        clusters_data = []
        for cluster in clusters:
            clusters_data.append( self.sampled_data[clusters,:] )
        return np.array(clusters_data)
    # TODO labeling data on disk
    
    
    def find_pair_clusters(self, clusters):
        maximum_goodness, cluster_indexes = 0.0, [-1, -1]
        for i in range(0, len(clusters)):
            for j in range(i + 1, len(clusters)):
                g = self.goodness(clusters[i], clusters[j])
                if g > maximum_goodness:
                    maximum_goodness = g
                    cluster_indexes = [i, j]
        return cluster_indexes
    
    
    def label_data(self, clusters):
        '''
        Label remaining points from data to assign them to the clusters
        obtained with sampled data
        for each cluster, normalize number of neighbors by dividing 
        with (len(C)+1)**((1.0-self.threshold)/(1.0+self.threshold))
        '''
        # remaining points, assigned to the clusters
        new_clusters_data = np.array([[] for k in range(len(clusters))]) 
        # normalisation factor of each cluster (expected number of links given the cluster size)
        norm = [(len(C)+1)**((1.0-self.threshold)/(1.0+self.threshold)) for C in clusters]
        #TODO replace clusters by cluster samples ?
        for p_index,point in enumerate(self.data):
            links_to_clusters = []
            for i in range(len(clusters)):
                links_to_clusters.append( get_links(point,clusters[i]) / norm[i] )
            new_clusters_data[np.argmax(links_to_clusters)].append(p_index)
        return new_clusters_data
            
            
        
        
        
        
        
        
        
        
    
    
    












