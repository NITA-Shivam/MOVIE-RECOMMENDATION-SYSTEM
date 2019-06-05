#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:52:03 2019

@author: shivam
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

import scipy.sparse as sp
from scipy.sparse.linalg import svds

import os
os.getcwd()
header = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv(r'u_pro',sep ='\t',names = header)

df.head()

df.info

n_users= df['user_id'].unique().shape[0]
n_items = df['item_id'].unique().shape[0]
print ('There are',n_users,'users')
print ('There are ', n_items,'movies')


ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
ratings

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],size=10,replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
   # to test if testing and training set are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test
train, test = train_test_split(ratings)

# Similarity_1
def similarity_1(ratings, kind, epsilon=1e-9):
# epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
user_similarity_1 = similarity_1(train, kind='user')
item_similarity_1 = similarity_1(train, kind='item')


# Similarity_2
user_similarity_2 = 1-pairwise_distances(train, metric='correlation')
item_similarity_2 = 1-pairwise_distances(train.T, metric='correlation')
item_similarity_2[np.isnan(item_similarity_2)] = 0

print(np.isnan(user_similarity_1).sum())
print(np.isnan(item_similarity_1).sum())
print(np.isnan(user_similarity_2).sum())
print(np.isnan(item_similarity_2).sum())


def predict(ratings, similarity, type):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
# axis =1 does row wise computation
#You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_predict_1 = predict(train, user_similarity_1, 'user')
item_predict_1 = predict(train, item_similarity_1, 'item')

user_predict_2 = predict(train, user_similarity_2, 'user')
item_predict_2 = predict(train, item_similarity_2, 'item')


print(np.isnan(user_predict_1).sum())
print(np.isnan(item_predict_1).sum())
print(np.isnan(user_predict_2).sum())
print(np.isnan(item_predict_2).sum())


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

print ('User base rmse with cosine similarities=',get_mse(user_predict_1, test))
print ('Item based rmse with cosine similarities=',get_mse(item_predict_1, test))

print ('User base rmse with pearson correlation=',get_mse(user_predict_2, test))
print ('Item based rmse with pearson correlation=',get_mse(item_predict_2, test))

# k_array = [30,35,40,50]
# After applying to several possible values of k and visualizing the result on a graph of rmse vs
#gives the least error. Thus, fixing k to be 50 gives :
k_array = [50]

user_test_mse_k_1 = []
item_test_mse_k_1= []

user_test_mse_k_2 = []
item_test_mse_k_2= []

def select_k(item_similarity_k, user_similarity_k,k) :
    for i in range(int(item_similarity_k.shape[0])) :
        max_k = -np.sort(-item_similarity_k, axis=1)[i,:k]
        for j in range(int(item_similarity_k.shape[1])) :
             if item_similarity_k[i,j] not in max_k :
                  item_similarity_k[i,j] = 0
    for i in range(int(user_similarity_k.shape[0])) :
        max_k = -np.sort(-user_similarity_k, axis=1)[i,:k]
        for j in range(int(user_similarity_k.shape[1])) :
            if user_similarity_k[i,j] not in max_k :
                user_similarity_k[i,j] = 0
    return user_similarity_k, item_similarity_k

for k in k_array :
    user_similarity_k_1 = similarity_1(train, kind='user')
    item_similarity_k_1 = similarity_1(train, kind='item')

    user_similarity_k_2 = 1-pairwise_distances(train, metric='correlation')
    item_similarity_k_2 = 1-pairwise_distances(train.T, metric='correlation')
    item_similarity_k_2[np.isnan(item_similarity_2)] = 0

    user_similarity_k_1, item_similarity_k_1 = select_k(item_similarity_k_1, user_similarity_k_1,k)
    user_similarity_k_2, item_similarity_k_2 = select_k(item_similarity_k_2, user_similarity_k_2,k)

    user_predict_k_1 = predict(train, user_similarity_k_1, 'user')
    item_predict_k_1 = predict(train, item_similarity_k_1, 'item')

    user_predict_k_2 = predict(train, user_similarity_k_2, 'user')
    item_predict_k_2 = predict(train, item_similarity_k_2, 'item')

    user_test_mse_k_1 += [get_mse(user_predict_k_1, test)]
    item_test_mse_k_1 += [get_mse(item_predict_k_1, test)]
    
    user_test_mse_k_2 += [get_mse(user_predict_k_2, test)]
    item_test_mse_k_2 += [get_mse(item_predict_k_2, test)]
print ('---------------Errors on using cosine similarity----------------- ')
# print 'For k in the order 30 , 35, 40, 50'# print 'For k in the order 30 , 35, 40, 50'
print ('User based mse for the test set = ' , user_test_mse_k_1)
print ('Item based mse for the test set = ', item_test_mse_k_1)
print ('---------------Errors on using pearson correlation----------------- ')
# print 'For k in the order 30 , 35, 40, 50'
print ('User based mse for the test set = ' , user_test_mse_k_2)
print ('Item based mse for the test set = ', item_test_mse_k_2)

## To visualize for various k's
#pal = sns.color_palette("Set2", 2)
#plt.figure(figsize=(15,8))
#plt.subplot(211)
#plt.plot(k_array, user_test_mse_k_1, c=pal[0], label='User-based test', linewidth=5)
#plt.plot(k_array, item_test_mse_k_1, c=pal[1], label='Item-based test', linewidth=5)
#plt.legend(loc='best', fontsize=20)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#plt.xlabel('k', fontsize=30)
#plt.ylabel('MSE', fontsize=30)
#plt.subplot(212)
#plt.plot(k_array, user_test_mse_k_2, c=pal[0], label='User-based test', linewidth=5)
#plt.plot(k_array, item_test_mse_k_2, c=pal[1], label='Item-based test', linewidth=5)
#plt.legend(loc='best', fontsize=20)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#plt.xlabel('k', fontsize=30)
#plt.ylabel('MSE', fontsize=30)


sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print ('The sparsity level of MovieLens100K is ' + str(sparsity*100) + '%')

# Training and predicting for different values of
j_values = [5,10,15,20,30,50,70,100,200]
err_mat = []

for j in j_values:
    u, s, vt = svds(train, k = j)
    s_diag_matrix=np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    err_mat += [get_mse(X_pred, test)]
    
    err_mat
    
pal = sns.color_palette("Set2", 2)
plt.figure(figsize=(15,8))
plt.plot(j_values, err_mat, c=pal[0], label='Error', alpha=0.5, linewidth=5)
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('k', fontsize=30)
plt.ylabel('MSE', fontsize=30)

# Fix k=20
u, s, vt = svds(train, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
model_mse= get_mse(X_pred, test)
print ('The mse for model based filtering where svd was used is', model_mse)

u, s, vt = svds(train, k = 20)
s_diag_matrix=np.diag(s)
x_model = np.dot(np.dot(u, s_diag_matrix), vt)
x_model_mse= get_mse(x_model, test)
x_model_mse

x_memory = predict(train, user_similarity_k_1, 'user')
x_memory_score = get_mse(x_memory, test)
x_memory_score

x_combined = (x_memory+x_model)/2
x_combined_score = get_mse(x_combined , test)
x_combined_score

ratings_df = pd.DataFrame(columns = ['Memory','Model','Actual'])
ratings_df['Memory'] = x_memory.flatten()
ratings_df['Model'] = x_model.flatten()
ratings_df['Actual'] = ratings.flatten()
ratings_df = ratings_df.loc[ratings_df.Actual > 0]
print (ratings_df.head())
print (ratings_df.shape)


from sklearn import linear_model
model = linear_model.LinearRegression()
train_data, test_data = cv.train_test_split(ratings_df, test_size=0.25)
# Fitting on train data.
X = train_data.drop(['Actual'], axis=1)
y= train_data['Actual']
X_test = test_data.drop(['Actual'], axis=1)
y_test = test_data['Actual']
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

fitted= model.fit(X,y)
predicted = model.predict(X_test)
print(rmse(predicted, y_test))

# Learning the weights of the corresponding models.
print (model.intercept_, model.coef_)


x_combined = (0.277 * x_memory) + (0.158 * x_model) + 2.774
x_combined_score = get_mse(x_combined , test)
x_combined_score

item_columns = [ 'Movie id','Movie title','Release date','Video release date',' IMDb URL',' Unknown','Action'
,'Adventure','Animation',' Children',' Comedy ',' Crime ', 'Documentary' ,' Drama',' Fantasy' ,
' Film-Noir',' Horror' ,' Musical ',' Mystery ',' Romance ',' Sci-Fi', 'Thriller',' War ',' Western']

import os
os.getcwd()
item= pd.read_csv(r'u.item',  error_bad_lines=False, sep='|',names=item_columns, encoding='latin-1')

def get_movie(movie_id) :
    movie_title = item.loc[movie_id-1 , 'Movie title']
    return movie_title
def get_topk_movies(final_predictions, original_ratings,k) :
# Removing movies those were already watched by the user.
      for i in range(original_ratings.shape[0]) :
        for j in range(original_ratings.shape[1]):
             if original_ratings[i,j] > 0 :
                  final_predictions[i,j] = 0

      recommend= np.zeros((943,k))
      recommend_df = pd.DataFrame(recommend)
# Ordering the remaining movies and getting the corresponding movie names.
      for i in range(final_predictions.shape[0]) :
          max_k = np.argsort(-final_predictions, axis=1)[i,:k]
          recommend[i] = max_k + 1
          for j in range(recommend.shape[1]) :
              recommend_df.loc[i,j] = get_movie(recommend[i,j])
      recommend_df.columns = range(1,k+1)
      recommend_df.index = range(1,944)
      return recommend_df
  
    final_recommendations = get_topk_movies(x_combined, ratings, 10)
    
    print(final_recommendations)
    
    final_recommendations.head()