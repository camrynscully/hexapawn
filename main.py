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