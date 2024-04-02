###### Your ID ######
# ID1: 205734049
# ID2: 208522094
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    # see https://en.wikipedia.org/wiki/Feature_scaling for more information about the calculation

    x_mean = np.mean(X)
    x_max = np.max(X)
    x_min = np.min(X)

    y_mean = np.mean(y)
    y_max = np.max(y)
    y_min = np.min(y)

    # Avoid division by zero
    if x_max == x_min:
        raise ValueError("Cannot perform mean normalization: x_max equals x_min")
    if y_max == y_min:
        raise ValueError("Cannot perform mean normalization: y_max equals y_min")

    X = (X - x_mean) / (x_max - x_min)
    y = (y - y_mean) / (y_max - y_min)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################

    # We use Reshape on input array to ensure it's two-dimensional
    # Since we are going to insert a column of 1's, we need to make sure that the input is a 2D array.
    # the -1 in the reshape function means that we want numpy to figure out what the value should be.
    # if X.ndim == 1:
    #     X = X.reshape(-1, 1)
    #
    # # use np.insert to insert a column of 1's into X at index 0.
    # # axis=1 means we insert a column.
    # X = np.insert(X, 0, 1, axis=1)
    N = X.shape[0]
    X = np.c_[np.ones((N, 1)), X]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################

    # Compute hypothesis using matrix multiplication (dot product):
    h = np.dot(X, theta)

    # Compute the squared error (hypothesis - y)^2, in vectorized form:
    error = h - y
    squared_error = error ** 2

    # Compute the cost function J:
    J = np.sum(squared_error) / (2 * len(y))
    # print(J)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    # iterate num_iters times to update theta:
    # for each iteration, compute the hypothesis, the error, the gradient, and update theta.
    for i in range(num_iters):
        # Compute hypothesis using matrix multiplication (dot product):
        h = np.dot(X, theta)

        # Compute the error (hypothesis - y), vectorized form:
        error = h - y

        # after deriving the cost function, we get the cost function J = (1/m) * sum((h - y) * X)
        # Compute using vectorized form: (the dot product includes the summing)
        gradient = np.dot(error, X) / len(y)

        # Update theta:
        theta = theta - alpha * gradient

        # Compute the cost function J:
        J = compute_cost(X, y, theta)
        J_history.append(J)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    # we are looking for the inverse matrix X^-1 so we can multiply it by y
    # more information on lec.1 time 2:12:00, 2:26:00 , slide 73-74
    # by slide 73-74: theta = pinv(X) * y = (X^T * X)^-1 * X^T * y

    X_pinv = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    pinv_theta = np.dot(X_pinv, y) # M_mxn+1 * M_mx1 = M_n+1x1,
    # (the optimal parameters vector of the model)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    for i in range(num_iters):
        # Compute hypothesis using matrix multiplication (dot product):
        h = np.dot(X, theta)

        # Compute the error (hypothesis - y), vectorized form:
        error = h - y

        # after deriving the cost function, we get the cost function J = (1/m) * sum((h - y) * X)
        # Compute using vectorized form: (the dot product includes the summing)
        gradient = np.dot(error, X) / len(y)

        # Update theta:
        theta = theta - alpha * gradient

        # Compute the cost function J:
        J = compute_cost(X, y, theta)
        J_history.append(J)

        # check the delta of the cost function:
        # if the difference between the last two cost functions is smaller than 1e-8
        # the i>0 condition is to avoid the first iteration
        if i > 0 and abs(J_history[-1] - J_history[-2]) < 1e-8:
            break
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    # steps:
    # for each alpha:
    # 1. find the optimal theta using efficient_gradient_descent
    # 2. check the validation loss using the optimal theta
    # 3. record the validation loss in the dictionary
    # return the dictionary
    for alpha in alphas:
        # initialize theta to random values
        np.random.seed(42)
        current_theta = np.random.random(size=X_train.shape[1]) # we use size= for the amount of features

        # find the optimal theta using efficient_gradient_descent
        # (we neglect the returned J_history)
        current_theta , _ = efficient_gradient_descent(X_train, y_train, current_theta, alpha, iterations)

        # check the validation loss using the optimal theta and record it in the dictionary
        alpha_dict[alpha] = compute_cost(X_val, y_val, current_theta)
    print("finished all alphas")
    print(alpha_dict)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly