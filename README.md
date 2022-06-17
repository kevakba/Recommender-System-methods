# Recommender System methods:

This repo contains the explanation of some of the recommender system methods present in the literature. I have tried to implement them using python and tested their performance. Enjoy the reading.

## 1. Collaborative Filtering
Ref: https://dl.acm.org/doi/abs/10.1145/371920.372071 <br>
Since the collaborative filtering problem can be viewed as a generalization of the classification/regression modeling problem, neighborhood-based methods can be viewed as generalizations of nearest neighbor classifiers in the machine learning literature.

**A. User-Based Neighborhood Models**

![image](https://user-images.githubusercontent.com/61937357/135264158-055ad2fd-84d7-4747-aa8c-df15ec137c55.png)

![image](https://user-images.githubusercontent.com/61937357/135264824-130791f8-4a1f-4d8b-8312-6fc006a7bda3.png)

![image](https://user-images.githubusercontent.com/61937357/135265005-89a6af61-5d7f-43fe-9144-02f0b86af5b1.png)

The mean-centering process enables a much better relative prediction with respect to the ratings that have already been observed.

In general, the Pearson correlation coefficient is preferable to the raw cosine because of the bias adjustment effect of mean-centering. The Pearson correlation coefficient is much more discriminative and the sign of the coefficient provides information about similarity and dissimilarity.

When the two users have only a small number of ratings in common, the similarity function should be reduced with a discount factor to de-emphasize the importance of that user pair. The same can be calculated as under:

![image](https://user-images.githubusercontent.com/61937357/135265575-824acee3-da76-4677-8ac2-4b34bbb6a3de.png)

Here beta is an integer value of choice.

Some movies may be very popular and they may repeatedly occur as commonly rated items by different users. Such ratings can sometimes worsen the quality of the recommendations because they tend to be less discriminative across different users. Inverse document frequency can be used. If m_j is the number of ratings of item j, and m is the total number of users, then the weight w_j of the item j is set to the following:

![image](https://user-images.githubusercontent.com/61937357/135265978-d65a2019-8e8e-4caf-be54-89c271f2dca0.png)

The Pearson correlation coefficient can be modified to include the weights as follows:

![image](https://user-images.githubusercontent.com/61937357/135266266-ff61a471-6b92-48bb-8339-bb38f5077449.png)

**B. Item-Based Collaborative Filtering**

Let U_i be the indices of the set of users who have specified ratings for item i. Then, the adjusted cosine similarity between the items (columns) i and j is defined as follows:

![image](https://user-images.githubusercontent.com/61937357/135266846-a55d1250-9033-4f9b-9838-40b209aded31.png)

Here S is itemswise mean-centred rating. Although the Pearson correlation can also be used on the columns in the case of the item-based method, the adjusted cosine generally provides superior results.

Consider the case in which the rating of target item t for user u needs to be determined. The first step is to determine the top-k most similar items to item t based on the aforementioned adjusted cosine similarity. Let the top-k matching items to item t, for which the user u has specified ratings, be denoted by Q_t (u).

![image](https://user-images.githubusercontent.com/61937357/135267428-ad907eca-a012-4535-af6e-5ffbed772503.png)

### 1.1 Dimensionality Reduction and Neighborhood Methods:

The reduced representation will either compress the item dimensionality or the user dimensionality into latent factors. This reduced representation can be used to alleviate the sparsity problem for neighborhood-based models. Depending on which dimension has been compressed into latent factors, the reduced representation can be used for either user-based neighborhood algorithms or item-based neighborhood algorithms.

R = (m-user X n-item) matrix
R_f = R filled with row-wise mean for NaN values

Simularity matrix (S) = R_f.T X R_f = P X E X P.T

P = (n X n) matrix with columns as eigen vectors of S <br>
E = (n X n) diagonal matrix with diagonal elements as eigen values of S <br>
P_d = (n X d) matrix with only d dominant eigen vectors of of S <br>
R_f X P_d = (m X d) matrix with each users having latent representation <br>



## 2. Matrix Factorization

Define a set of Users (U), items (D) such that the Matrix R is of size (|U| X |D|) and includes all the ratings given by users. The goal is to discover K latent features. Given with the input of two matrics matrices P (|U| X k) and Q (|D| X k), it would generate the product result R.

R = P X Transpose(Q)

Matrix P represents the association between a user and the features while matrix Q represents the association between an item and the features.

We can get the prediction of a rating of an item by the calculation of the dot product of the two vectors corresponding to u_i and d_j.

![image](https://user-images.githubusercontent.com/61937357/135051677-a75083b6-3f2e-454d-b1ad-e5a7dbb2533a.png)

To get two entities of both P and Q, we need to initialize the two matrices and calculate the difference of the product named as matrix M. Next, we minimize the difference through gradient descent, aiming at finding a local minimum of the difference.

![image](https://user-images.githubusercontent.com/61937357/135052074-7326262c-1cc6-45e8-8280-c68f30efefe7.png)

To minimize the error, the gradient is able to minimize the error, and therefore we differentiate the above equation with respect to these two variables separately.

![image](https://user-images.githubusercontent.com/61937357/135052192-091925f8-dc22-48a0-a000-b547c8400307.png)

From the gradient, the mathematic formula can be updated for both p_ik and q_kj. Alpha is the step to reach the minimum while the gradient is calculated, and Aplha is usually set with a small value.

![image](https://user-images.githubusercontent.com/61937357/135052513-2a142db0-c10d-4e19-b335-a5ed8fbc4608.png)

From the above equation, p_ik and q_kj can both be updated through iterations until the error converges to its minimum.

![image](https://user-images.githubusercontent.com/61937357/135052713-26e778dd-402b-4533-bbba-0aa133b37a73.png)

Implementation for the above method is [here](https://github.com/kevakba/Recommender-System-methods/blob/main/Recommender_System_%E2%80%94_Matrix_Factorization.ipynb).


## 3. Neural Network based Matrix Factorization

Following is the framework for the model:

![image](https://user-images.githubusercontent.com/61937357/135059723-f1dc0d07-9fb4-4a89-b5ab-a3c88620108f.png)

It mainly contains two units viz. Generalized Matrix Factorization (GMF) & Multi-Layer Perceptron (MLP).

say p_u = User embedded latent vector & q_u = Item embedded latent vector

**A. Generalized Matrix Factorization (GMF)**

MF can be interpreted as a framework to mimick colaborative filtering. It basically takes-in the embedded vectors of user and item, perform element-wise multiplication and pass the resultant vector into an activation function (linear or non-linear). Hence it helps in capturing the interactions between user and item latent features, just like in collaborative filtering method.

![image](https://user-images.githubusercontent.com/61937357/135062389-eefddc79-d04e-4dab-9507-77e7da4f8990.png)

**B. Multi-Layer Perceptron (MLP)**

MLP takes in the embedded vectors of user and item, concatenate them and pass the resultant vector into a feedforward neural network. In this sense, we can
endow the model a large level of flexibility and non-linearity to learn the interactions between user and item.

![image](https://user-images.githubusercontent.com/61937357/135062470-fe633d87-7509-40fb-97c6-ca3bb9380957.png)

The above two vectors from GMF and MLP are then concatenated and passed into a final Output layer for prediction. Output layer may be linear for regression task or with non-linear activation function (like sigmoid) for classification task.

Please refer this [paper](https://arxiv.org/abs/1708.05031) for more details.


## 4. Decision and Regression Trees:

- Let's suppose we have Rating matrix (R) of size (mXn).
- To predict all the ratings of any item column i, we use (n-1) columns as predictor and apply decision tree regressor or classifier, depending upon the avilable data.
- Before applying DT model, we use these (n-1) columns and reduce them into (mXd) matrix with SVD, where d<<(n-1), inorder to remove the sparcity.
- We do above steps for all the columns and hence we would have (n) decision trees for predicting each and every ratings in our table. 


## 5. Naive Bayes Collaborative Filtering:

![image](https://user-images.githubusercontent.com/61937357/135510703-e770da6f-c5bb-4e4f-89ff-89a14e3bbfac.png)



