#!/usr/bin/env python3
"""Implementation of K means and K means ++ algorithm"""
import sys

import random
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
#import matplotlib.pyplot as plt

def preprocessing_churn_data(data):
    """Pre processing data"""
    #print(data.info())
    #print(data.head())
    dataframe_churn = pd.DataFrame(data)
    """Pre processing data
    dataframe_churn = pd.DataFrame(data=dataframe_churn, \
        columns=['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', \
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', \
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', \
        'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])"""
    dataframe_churn.drop(dataframe_churn.columns[[0]], axis=1, inplace=True)
    categorical_cols = {1,3,4,6,7,8,9,10,11,12,13,14,15,16,17}
    for num in categorical_cols:
        dataframe_churn[num] = preprocessing.LabelEncoder().fit_transform(dataframe_churn[num])
        #dataframe_churn[num] = dataframe_churn[num].astype('category')
        #dataframe_churn[num] = dataframe_churn[num].cat.codes
    dataframe_churn[19] = pd.to_numeric(dataframe_churn[19], errors="coerce")
    dataframe_churn.dropna(how='any', inplace=True)
    return dataframe_churn

def normalize_data(data):
    """Nomralize data to value between 0 and 1"""
    minmax_processed = preprocessing.MinMaxScaler().fit_transform(data)
    normalized_df = pd.DataFrame(minmax_processed)
    return normalized_df

def euclidian_dist(a, b, axis=1):
    """Euclidian distance between 2 points in Euclidian space"""
    return np.linalg.norm(a - b, axis=axis)

def initial_centroids(data, k):
    """Random initial centroids for k means"""
    centroids = random.sample(list(data), k)
    return centroids

def has_converged(centroids, centroids_old):
    """Check if k means has converged"""
    return False
    #return set([tuple(a) for a in centroids]) == set([tuple(a) for a in centroids_old])

def distance_from_centers(X, centroids):
    """D2 Squared distance for points from centers"""
    dist_squared = np.array([min([np.linalg.norm(x-center)**2 for center in centroids]) for x in X])
    return dist_squared

def next_center(X, dist_squared):
    """Select next center based on highest probability"""
    prob = dist_squared/dist_squared.sum()
    cumprob = prob.cumsum()
    r = random.random()
    index = np.where(cumprob >= r)[0][0]
    return X[index]

def kpp_centers(X, k):
    """Initial Centers for k plus plus clustering"""
    centroids = random.sample(list(X), 1)
    for i in range(1, k):
        dist = distance_from_centers(X, centroids)
        centroids.append(next_center(X, dist))
    return centroids

def k_means(data, k, init="random"):
    """function implementing k-means algorithm"""
    #inputs dataframe, number of clusters and type of kmeans
    normalized_df = normalize_data(data)
    x = normalized_df.values
    X = np.array(list(zip(x)))
    clusters = np.zeros(len(X))
    #Initial cluster centers
    if init == "random":
        centroids = initial_centroids(X, k)
    else:
        centroids = kpp_centers(X, k)
    #k Means algorithm. Iterate until convergence
    max_iterations = 100
    while max_iterations >= 0:
        for i in range(len(X)):
            distances = [euclidian_dist(X[i], center) for center in centroids]
            cluster = np.argmin(distances)
            clusters[i] = cluster

        centroids_old = deepcopy(centroids)
        #new centroids are mean of all data points in new cluster
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            centroids[i] = np.mean(points, axis=0)
        #centroids = new_centers(k, X, clusters)
        if has_converged(centroids, centroids_old):
            print("breaking...... Convergence")
            #break
        max_iterations = max_iterations - 1
    return clusters


def pca_analysis(data):
    """Performing principal component analysis for our data"""
    pca_data = normalize_data(data)
    sklearn_pca = PCA(n_components=2)
    principal_components = sklearn_pca.fit_transform(pca_data)
    principal_df = pd.DataFrame(data=principal_components, \
        columns=['Principal Component 1', 'Principal Component 2'])
    #plt.figure(figsize=(10, 8))
    #plt.scatter(principalDF[:, 0], principalDF[:, 1], c=y)
    #k_means(principalDF,2)
    #k_means(principalDF,2,"kpp")

def optimal_k(data):
    """Find optimal number of k clusters using Silhoutte score analysis"""
    normalized_df = normalize_data(data)
    matrix = normalized_df.as_matrix()
    k_scores = []
    for n_clusters in range(2, 30):
        clusters = k_means(data, n_clusters, type_kmeans)
        silhouette_avg = silhouette_score(matrix, clusters)
        y.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    #Plot the average Silhoutte score for each num
    plt.figure(figsize=(12, 8))
    plt.plot(range(2, 30), k_scores)
    plt.xlabel('No of Clusters')
    plt.ylabel('Silhouette_avg')
    plt.title('Silhoutte Score for different clusters')

    return k_scores

def main():
    """main function"""
    try:
        data = pd.read_csv(sys.argv[1], header=None)
        k = int(sys.argv[2])
        type_kmeans = sys.argv[3]
    except IndexError:
        print("Expected 3 inputs : filepath , number of clusters and type of kmeans. \
            \nProgram exiting....")
        sys.exit(1)

    #K - means clustering
    if len(data.columns) >= 15:
        #cluster.csv dataset
        procesed_churn_data = preprocessing_churn_data(data)
        procesed_churn_data_df = pd.DataFrame(procesed_churn_data)
        clusters = k_means(procesed_churn_data_df, 2, type_kmeans)
    else:
        #wine.csv dataset
        clusters = k_means(data, k, type_kmeans)

    #k_scores = optimal_k(data)
    """
    #Perform PCA
    #pca_analysis(data)
    """
    output = pd.DataFrame(clusters)
    output.to_csv('output.csv', sep=',', index=False, header=None)

if __name__ == '__main__':
    main()
