# Recommender-System-methods

1. Matrix Factorization

Define a set of Users (U), items (D) such that the Matrix R is of size (|U| X |D|) and includes all the ratings given by users. The goal is to discover K latent features. Given with the input of two matrics matrices P (|U| X k) and Q (|D| X k), it would generate the product result R.

R = P X Transpose(Q)

Matrix P represents the association between a user and the features while matrix Q represents the association between an item and the features.

We can get the prediction of a rating of an item by the calculation of the dot product of the two vectors corresponding to u_i and d_j.


2. Neural Network based Matrix Factorization
