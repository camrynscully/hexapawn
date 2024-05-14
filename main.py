# CISC 681 - Programming Assignment 3
# Camryn Scully

# Very early on - will eventually be used as the main script to run the program

from hexapawn import *

initial_state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]

# Part 1 
poss_actions = actions(initial_state)
print("Possible Actions of Initial State: ", poss_actions)

update_state = result(initial_state, poss_actions[1])
print("Updated State after taking 2nd Action: ", update_state)

terminal_state = [0, -1, -1, 0, 0, 0, 0, 1, 1, -1]
print("Check if Terminal State is Terminal or Not: ", is_terminal(terminal_state))
print("Max's Score:", utility(terminal_state))

# Part 2
print("Minimax Search: ", minimax_search(initial_state))

print("------------------Policy Table------------------\n")
print("Initial State: ", initial_state)
policy = policy_table(initial_state) 
for state in policy:
    vals = policy[state]
    print("State:", state, "\nInitial Value:", vals[0], ", New Value:", vals[1], ", Action:", vals[2])
    print("\n")
    
# Part 3
layer1 = Layer(neurons=2, num_inputs=2)
layer2 = Layer(neurons=2, num_inputs=2)
network = NeuralNetwork(num_layers=2, layers=[layer1, layer2])
print("---------------Neural Network-----------------\n")
print("Layer 1 - Weights:", network.layer1.weights, "Biases:", network.layer1.biases)
print("Layer 2 - Weights:", network.layer2.weights, "Biases:", network.layer2.biases)

# Part 4
x1 = -1
x2 = 0
x3 = 1
x4 = 100

inputs = np.array([0, 1])

print("Sigmoid - Scalar Input:", x1, "Output:", sigmoid(x1))
print("Sigmoid - Vector Input:", inputs, "Output:", sigmoid(inputs))

print("ReLU - Scalar Input:", x1, "Output:", ReLU(x1))
print("ReLU - Vector Input:", inputs, "Output:", ReLU(inputs))

print("ReLU NN Output: ", classify(network, inputs, ReLU))
print("Sigmoid NN Output: ", classify(network, inputs, sigmoid))