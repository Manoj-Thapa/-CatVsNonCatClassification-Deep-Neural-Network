import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dnn_functions import *


def preprocessing():
    """

            Common steps for pre-processing a new dataset are:

            -> Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, …)
            -> Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
            -> “Standardise” the data

            Many software bugs in deep learning come from having matrix/vector dimensions that don’t fit.
            If you can keep your matrix/vector dimensions straight you will go a long way toward eliminating
            many bugs.

    """

    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    return train_set_x, train_set_y, test_set_x, test_set_y


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """

            Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

            Arguments:
            X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
            layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps

            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.

    """

    np.random.seed(1)
    costs = []  # keep track of cost
    iterations = []  # keep track of iterations

    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations+1):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            iterations.append(i)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    # plot the cost
    plt.plot(iterations, costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    """ 

        You can test this algorithm, simply giving your path name of your image 

    """

    # Change this to the name of your image file
    my_image = "VillageCat.jpg"
    my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)

    # Preprocess the image to fit into the algorithm.

    fname = "Images/" + my_image
    image = np.array(Image.open(fname).resize((64, 64)))
    image = image / 255
    final_image = image.reshape(1, 64 * 64 * 3).T
    print('\nAccuracy of your image: ')
    my_predicted_image = predict(final_image, my_label_y, parameters)

    plt.imshow(image)
    plt.show()

    if my_predicted_image[0][0] == 1:
        print('\nYour 4-layer model predicts a "Cat" picture.')
    else:
        print('\nYour 4-layer model predicts a "Non-Cat" picture.')

    return parameters


train_x, train_y, test_x, test_y = preprocessing()

layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model

parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True)

print('\nAccuracy on Train and Test sets\n')
print('Train Accuracy: ')

pred_train = predict(train_x, train_y, parameters)

print('\nTest Accuracy: ')

pred_test = predict(test_x, test_y, parameters)
