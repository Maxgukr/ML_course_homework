
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from dnn_app_utils_v2 import *
import matplotlib.pyplot as plt


def L_layer_model(X, Y, layers_dims, learning_rate=0.0105, num_iterations=3000, print_cost=False):  # lr was 0.009

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 10 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def linear_svm_model(train_x, y, test_x):
    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(train_x, y) for clf in models)

    res=[]
    for clf in models:
        res.append(clf.predict(test_x))

    return np.array(res)


def MLPClassification(layer_dims, X_train, y_train, testx, learning_rate, iteration_num, alpha):
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(4, 3), random_state=1, activation='logistic',max_iter=500)
    clf.fit(X_train, y_train)
    res = clf.predict(testx)
    return res


def main():
    layers_dims = [4, 3, 3]
    dataload = load_iris()
    data = dataload.data
    label = dataload.target

    # shuffle the train data and labels
    indices = np.random.permutation(np.array(data).shape[0])
    train_data = data[indices]
    labels = np.array(label)[indices]

    data_scaled = scale(train_data) # scaled the training data

    # transform label
    y = np.zeros((3, data_scaled.shape[0]))
    for i in range(data_scaled.shape[0]):
        y[labels[i]][i] = 1

    train_x = data_scaled[0:100,:]
    train_y = y[:, 0:100]
    validation_x = data_scaled[101:120,:]
    validation_y = y[:,101:120]
    test_x = data_scaled[121:-1,:]
    test_y = y[:, 121:-1]

    parameters = L_layer_model(train_x.T, train_y, layers_dims, num_iterations=1200, print_cost=True)

    print("predicted accuracy in validation data:")
    pred_train = predict(validation_x.T, labels[101:120], parameters, validation_y)
    print("predicted accuracy in test data:")
    pred_test = predict(test_x.T, labels[121:-1], parameters, test_y)


if __name__=="__main__":
    main()