# CISC 681 - Programming Assignment 3
# Camryn Scully

# Very early on - will eventually be used as the main script to run the program

from hexapawn import *

initial_state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]

# Part 1 
poss_actions = actions(initial_state)
print(poss_actions)

update_state = result(initial_state, poss_actions[1])
print(update_state)

terminal_state = [0, -1, -1, 0, 0, 0, 0, 1, 1, -1]

print(actions(terminal_state))
print(is_terminal(terminal_state))
print("Max's Score:", utility(terminal_state))


