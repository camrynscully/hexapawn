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

"""Part 1 : Formalization of Hexapawn"""
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