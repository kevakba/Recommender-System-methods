# Recommender System methods

**1. Matrix Factorization**

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


**2. Neural Network based Matrix Factorization**
