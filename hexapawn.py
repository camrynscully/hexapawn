# CISC 681 - Programming Assignment 3
# Camryn Scully

""" 
Hexapawn : 3x3 board with 6 pawns ( 3 per player) 
White pawns move up one square or capture a black pawn diagonally 
Black pawns move down one square or capture a white pawn diagonally
Goal : Get one of your own pawns to the other side of the board or get your 
opponents pawns stuck on their move
"""

import numpy as np

"""
The board is represented as a vector of real numbers to be a valid input to the neural network.
The first value, 0 or 1, represents whose turn it is. The following values present the state
of the board in row major order. 0 is an empty space, 1 is a white pawn, and 2 is a black pawn.
"""

"""
For reference, the output vector will be composed of 9 values, one for each cell on the board.
A 0 indicates that moving a pawn to that cell is not a valid move, while a 1 indicates the moving a pawn there is optimal.
"""

## -------- Part 1 : Formalization of Hexapawn ---------- ##
initial_state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]

def to_move(state):
    " Returns the player whose turn it is "
    
    return state[0]

def change_turns(state):
    " Changes the player whose turn it is "
    
    state[0] = (state[0] + 1) % 2
    
    return state

def actions(state):
    " Returns a list of legal actions for the given state "
    
    possible_actions = []
    board = [state[i:i+3] for i in range(1, len(state), 3)]
    
    if to_move(state) == 0: # black pawn's turn
        for row in range(0, len(board)):
            for col in range(0, len(board[0])):
                if board[row][col] == -1: # only consider black pawns
                    if row == 0 or row == 1: # check if the pawn can move down
                        if board[row+1][col] == 0:
                            possible_actions.append(["advance", row, col])
                        if col == 0:
                            if board[row+1][col+1] == 1:
                                possible_actions.append(["capture-left", row, col])
                        elif col == 1:
                            if board[row+1][col-1] == 1:
                                possible_actions.append(["capture-right", row, col])
                            if board[row+1][col+1] == 1:
                                possible_actions.append(["capture-left", row, col])
                        elif col == 2:
                            if board[row+1][col-1] == 1:
                                possible_actions.append(["capture-right", row, col])                     
    elif to_move(state) == 1: # white pawn's turn
        for row in range(0, len(board)):
            for col in range(0, len(board[0])):
                if board[row][col] == 1: # only consider white pawns
                    if row == 2 or row == 1: # check if the pawn can move up
                        if board[row-1][col] == 0:
                            possible_actions.append(["advance", row, col])
                        if col == 0:
                            if board[row-1][col+1] == -1:
                                possible_actions.append(["capture-right", row, col])
                        elif col == 1:
                            if board[row-1][col-1] == -1:
                                possible_actions.append(["capture-left", row, col])
                            if board[row-1][col+1] == -1:
                                possible_actions.append(["capture-right", row, col])
                        elif col == 2:
                            if board[row-1][col-1] == -1:
                                possible_actions.append(["capture-left", row, col])          
    
    return possible_actions
    

def result(state, action):
    " Returns the state that results from taking the action in the given state"
    
    new_state = []
    row, col = action[1], action[2]
    
    # convert to a 2d array to iterate through easily
    board = [state[i:i+3] for i in range(1, len(state), 3)]
    
    if board[row][col] == -1: # black pawn
        if action[0] == "advance": 
            board[row+1][col] = -1  # black pawn moves down (increase row)
            board[row][col] = 0     # replace previous position with empty space
        elif action[0] == "capture-left":
            board[row+1][col+1] = -1  # empty space moves down (increase row)
            board[row][col] = 0     # replace previous position with empty space
        elif action[0] == "capture-right":
            board[row+1][col-1] = -1  # empty space moves down (increase row)
            board[row][col] = 0     # replace previous position with empty space
    elif board[row][col] == 1: # white pawn
        if action[0] == "advance": 
            board[row-1][col] = 1   # white pawn moves up (decrease row)
            board[row][col] = 0     # replace previous position with empty space
        elif action[0] == "capture-right":
            board[row-1][col+1] = 1   # empty space moves up (decrease row)
            board[row][col] = 0     # replace previous position with empty space
        elif action[0] == "capture-left":
            board[row-1][col-1] = 1   # empty space moves up (decrease row)
            board[row][col] = 0     # replace previous position with empty space
        
    new_state.append(state[0])
    for row in board:
        new_state.extend(row)
    
    # change the player's turn
    change_turns(new_state)
    
    return new_state

def is_terminal(state):
    " Returns True if the game is over, False otherwise"
    
    # Terminal States: 
    # Black or White pawns are on the other side of the board (black in row 2, white in row 0)
    # The other player can not make a move (no legal actions)
    
    game_over = False
    
    # check if either the black or white pawns are on the other side of the board
    if any(element == 1 for element in state[1:4]):
        game_over = True
    elif any(element == -1 for element in state[8:10]):
        game_over = True
    
    # check if the other player can make a move
    if len(actions(state)) == 0:
        game_over = True
        
    return game_over

def utility(state):
    "Returns a numerical value for Max (the white team) given a terminal state"
    
    # check whose turn it currently is (meaning whoever just made the last move produced a terminal state) - aka they won 
    pawn_turn = to_move(state)
    
    if pawn_turn == 0: # white pawn's turn would be next, black made the last move
        score = 0  
    elif pawn_turn == 1: # black pawn's turn would be next, white made the last move
        score = 1

    ## Need to consider a tie?? The score would be 1/2 for each player 
    ## Would add it to the terminal state to check if neither player can make a move?
    
    return score

## ------------- Part 2 : Minimax search ---------------------- ## 
def minimax_search(state):
    "Return the optimal action to be taken by the current player given the current state"

    # pawn_turn = to_move(state) - in the pseudo code but seems unnecessary?
    value, action = max_value(state, state)     # find the best action to take
    
    return action
    
def max_value(game, state):
    "Return the maximum utility value and action for the current player and state"
    
    if is_terminal(state):              # if the state is terminal, return the value of the state and no action
        return utility(state), None

    max = -np.inf
    for a in actions(state):            # iterate through all possible actions
        temp_val, temp_action = min_value(state, result(state, a)) # find min value and action opponent to take
        if temp_val > max:              # if the value of the state is greater than the current max, store the value & action
            max, action = temp_val, a
            
    return max, action

def min_value(game, state):
    "Return the minimum utility value and action for the opponent"
    if is_terminal(state):              # if the state is terminal, return the value of the state and no action
        return utility(state), None

    min = np.inf
    for a in actions(state):            # iterate through all possible actions
        temp_val, temp_action = max_value(state, result(state, a))
        if temp_val < min:              # if the value of the state is less than the current min, store the value & action
            min, action = temp_val, a
            
    return min, action

# build the policy table
def generate_boards(state, boards):
    "Return all possible states given the current state"
    
    for poss_action in actions(state):                              # iterate through all possible actions
        boards.append(result(state, poss_action))                   # generate and store the resulting board given an action
        if not is_terminal(result(state, poss_action)):             # check if the resulting board is not a terminal state
            generate_boards(result(state, poss_action), boards)     # if not, recursively call generate boards to find the successive boards given the new board
    
    return boards

def policy_table(state):
    "Return a policy table containing each state along with the value and every action that achieves such value"
    
    table = {}                                  # initialize a dict for the policy table
    for board in generate_boards(state, []):    # iterate through every possible board given a state
        action = minimax_search(board)          # find best action to be taken for the current board (state)
        initial_value = utility(board)          # get the initial value of the board (state)
        
        new_value = initial_value if action is None else utility(result(board, action)) # if an action exists, find the value of the board following that action
        
        table[str(board)] = initial_value, new_value, action # add board to the policy table along with initial, new, and action
        
    return table

## ---------------- Part 3: Graph Structure ------------------ ##
class Layer:
    def __init__(self, neurons, num_inputs):
        "Initialize a layer in the neural network"
        np.random.seed(10)
        self.neurons = neurons
        self.weights = 2 * np.random.random((num_inputs, neurons)) - 1  # multiply by 2 to get range to be [0.0, 2.0] 
        self.biases = 2 * np.random.random(neurons) - 1                 # subtract 1 to get range to be [-1.0, 1.0] 
        
class NeuralNetwork:
    def __init__(self, num_layers, layers):
        "Initialize a neural network with the provided number of layers"
        for i in range(num_layers):
            print(i)
            setattr(self, f"layer{i+1}", layers[i])

## -------------- Part 4: Classify Function ----------------- ##
def classify(network, inputs, activation_function):
    "Returns the output of each layer of the provided neural network using the given activation function"
    
    # compute the output of each layer
    output1 = activation_function(np.dot(network.layer1.weights, inputs)) + network.layer1.biases
    output2 = activation_function(np.dot(network.layer2.weights, output1)) + network.layer2.biases
    
    return output1, output2

# activation functions
def sigmoid(x):
    "Sigmoid (or Logistic) - Returns a value between 0 and 1 provided x"
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    "Rectified Linear Unit - Returns 0 if x is negative, otherwise returns x"
    return np.maximum(0, x)
