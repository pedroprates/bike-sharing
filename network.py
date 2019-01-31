import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

        # Adjust learning rate during training
        self.adjust_learning_rate = lambda x: self.lr * x

    def train(self, features, targets):
        """ Train the network on batch of features and targets.
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        """
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros_like(self.weights_input_to_hidden)
        delta_weights_h_o = np.zeros_like(self.weights_hidden_to_output)

        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        """ Implement forward pass here
         
            Arguments
            ---------
            X: features batch

        """
        # Forward pass
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, x, y, delta_weights_i_h, delta_weights_h_o):
        """ Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            hidden_outputs: output from hidden layer on the forward pass
            x: input batch
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        """
        # Backward pass
        error = y - final_outputs

        hidden_error = hidden_outputs[:, None] * (1 - hidden_outputs[:, None]) * self.weights_hidden_to_output

        output_error_term = error
        
        hidden_error_term = output_error_term * hidden_error
        
        # Weight step (input to hidden)
        delta_weights_i_h += np.dot(x[:, None], hidden_error_term.T)
        # Weight step (hidden to output)
        delta_weights_h_o += hidden_outputs[:, None] * output_error_term

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """ Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        """
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        """ Run a forward pass through the network with input features
        
            Arguments
            ---------
            features: 1D array of feature values
        """
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs


# Hyperparameters:
iterations = 25000
learning_rate = 0.01
hidden_nodes = 53
output_nodes = 1
