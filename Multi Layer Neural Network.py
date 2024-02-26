import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

#initalize weights accounting for a bias    
def initialize_weights(input_size, output_size, seed):
    np.random.seed(seed)
    weights = np.random.randn(output_size, input_size+1)
    return weights

#forward propagtion and adds a 1 to inputs
def forward_propagation(inputs, weights):
    n_samples = inputs.shape[1]
    inputs = np.concatenate([np.ones((1, n_samples)), inputs], axis = 0)

    z = np.dot(weights, inputs)
    activations = sigmoid(z)

    return activations
    
def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def gradient_descent(weights, error_function, h = 1e-5):
    gradient = np.zeros_like(weights)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
           original_weights = weights[i, j]
           
           #f(x + h)
           weights[i, j] = original_weights + h
           error_positive = error_function(weights)
           
           #f(x - h)
           weights[i, j] = original_weights - h
           error_negative = error_function(weights)
           
           #(f(x + h) - f(x - h)) / 2h
           gradient[i,j] = (error_positive - error_negative) / (2 * h)
           weights[i, j] = original_weights
    return gradient

def update_weights(weights, gradient, learning_rate):
    weights -= learning_rate * gradient

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
     #initalize weights
    np.random.seed(seed)
    weights = []
    input_dim = X_train.shape[0]

    for i in range(len(layers)):
        w = initialize_weights(input_dim, layers[i], seed)
        weights.append(w)
        input_dim = layers[i]

    #training loop
    test_errors = [] #store errors after each epoch
    for epoch in range(epochs):
        activations = [X_train]

        #forward propagation computes activations 
        for w in weights:
            activations.append(forward_propagation(activations[-1], w))
        
        #Backward propagation and weight update
        for i in reversed(range(len(weights))):
            if i == len(weights)-1:
                error_function = lambda w, i=i: mse(Y_train, forward_propagation(activations[i], w))
            else:
                def error_for_layer(w):
                    act = forward_propagation(activations[i], w)
                    for l in range(i+1, len(weights)):
                        act = forward_propagation(act, weights[l])
                    return mse(Y_train, act)
                error_function = error_for_layer            
            gradient = gradient_descent(weights[i], error_function, h)
            update_weights(weights[i], gradient, alpha)

        #compute test error without updating weights
        test_activations = [X_test]
        for w in weights:
            test_activations.append(forward_propagation(test_activations[-1], w))
        test_errors.append(mse(Y_test, test_activations[-1]))

    #forward propagation when epochs is 0 on test inputs
    if epochs == 0:
        test_activations = [X_test]
        for w in weights:
            test_activations.append(forward_propagation(test_activations[-1], w))

    return weights, test_errors, test_activations[-1]
    # (f(x + h)-f(x - h))/2*h
    # Reseed the random number generator when initializing weights for each layer.
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()
