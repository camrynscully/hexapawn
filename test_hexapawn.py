# CISC 681 - Programming Assignment 3
# Camryn Scully

from hexapawn import *

# Part 1 Validation Tests
def test_to_move():
    initial_state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]
    assert to_move(initial_state) == 0
    
def test_change_turns():
    initial_state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]
    
    state = change_turns(initial_state)
    assert state[0] == 1
    state = change_turns(state)
    assert state[0] == 0
    state = change_turns(state)
    assert state[0] == 1
    

def test_actions():
    initial_state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]
    
    actions = actions(initial_state)
    assert len(actions) == 6