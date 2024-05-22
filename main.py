# CISC 681 - Programming Assignment 3
# Camryn Scully

from hexapawn import *
import random

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

inputs = [[0,0], [0,1], [1,0], [1,1]]       
outputs = [[0,0], [0,1], [0,1], [1,0]]

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

# Part 5
print("Derivative of Sigmoid:", derivative_sigmoid(inputs))
print("Derivative of ReLU:", derivative_ReLU(inputs))

print("\n------------------Initial Weights & Biases--------------------")
print("Layer 1 weights: ", network.layer1.weights, "\nLayer 2 weights:", network.layer2.weights)
print("\nLayer 1 Biases:", network.layer1.biases, "\nLayer 2 Biases:", network.layer2.biases)

print("\n----------------Two-Bit Adder Network------------------------")
layer1 = Layer(neurons=2, num_inputs=2)
layer2 = Layer(neurons=2, num_inputs=2)
AdderNetworkReLU = NeuralNetwork(num_layers=2, layers=[layer1, layer2])
print("Initial Network")
print("Layer 1 Weights: ", AdderNetworkReLU.layer1.weights, "Biases: ", AdderNetworkReLU.layer1.biases)
print("Layer 2 Weights: ", AdderNetworkReLU.layer2.weights, "Biases: ", AdderNetworkReLU.layer2.biases)

inputs = [[0,0], [0,1], [1,0], [1,1]]       
outputs = [[0,0], [0,1], [0,1], [1,0]]

# print("\nTraining Neural Network...")
# train_network(AdderNetworkReLU, inputs, outputs, ReLU, 10)
# print("...Weights and Biases Updated!")
# print("\nFinal Network")
# print("Layer 1 Weights: ", AdderNetworkReLU.layer1.weights, "Biases: ", AdderNetworkReLU.layer1.biases)
# print("Layer 2 Weights: ", AdderNetworkReLU.layer2.weights, "Biases: ", AdderNetworkReLU.layer2.biases)

# print("\n-------------Check Classification, ReLU Activation-----------------------")
# x, y, z1, z2 = classify(AdderNetwork, inputs[0], ReLU)
# print("Expected Output: ", outputs[0], "Actual: ", z1, z2)
# x, y, z1, z2 = classify(AdderNetwork, inputs[1], ReLU)
# print("Expected Output: ", outputs[1], "Actual: ", z1, z2)
# x, y, z1, z2 = classify(AdderNetwork, inputs[2], ReLU)
# print("Expected Output: ", outputs[2], "Actual: ", z1, z2)
# x, y, z1, z2 = classify(AdderNetwork, inputs[3], ReLU)
# print("Expected Output: ", outputs[3], "Actual: ", z1, z2)

layer1 = Layer(neurons=2, num_inputs=2)
layer2 = Layer(neurons=2, num_inputs=2)
AdderNetworkSig = NeuralNetwork(num_layers=2, layers=[layer1, layer2])

print("\nTraining Neural Network...")
train_network(AdderNetworkSig, inputs, outputs, 1000)
print("...Weights and Biases Updated!")
print("\nFinal Network")
print("Layer 1 Weights: ", AdderNetworkSig.layer1.weights, "Biases: ", AdderNetworkSig.layer1.biases)
print("Layer 2 Weights: ", AdderNetworkSig.layer2.weights, "Biases: ", AdderNetworkSig.layer2.biases)

print("\n--------------------Check Classification---------------------------")
x, y, z1, z2 = classify(AdderNetworkSig, inputs[0])
print("Expected Output: ", outputs[0], "Actual: ", z1, z2)
x, y, z1, z2 = classify(AdderNetworkSig, inputs[1])
print("Expected Output: ", outputs[1], "Actual: ", z1, z2)
x, y, z1, z2 = classify(AdderNetworkSig, inputs[2])
print("Expected Output: ", outputs[2], "Actual: ", z1, z2)
x, y, z1, z2 = classify(AdderNetworkSig, inputs[3])
print("Expected Output: ", outputs[3], "Actual: ", z1, z2)

# Part 6 
l1 = Layer(neurons=10, num_inputs=10)
l2 = Layer(neurons=9, num_inputs=10)
HexapawnNetwork = NeuralNetwork(num_layers=2, layers=[l1, l2])

# print("Layer 1")
# print("Weights:", HexapawnNetwork.layer1.weights)
# print("\nBiases:", HexapawnNetwork.layer1.biases)

# print("\nLayer 2")
# print("Weights:", HexapawnNetwork.layer2.weights)
# print("\nBiases:", HexapawnNetwork.layer2.biases)

input1 = [1,-1,-1,-1,0,0,0,1,1,1]
# h1, h2, y1, y2 = classify(HexapawnNetwork, input1)


