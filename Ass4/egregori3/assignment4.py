'''
ML CS7641   Assignment 4    Eric Gregori

https://github.com/aimacode/aima-python/blob/master/mdp.py
https://github.com/aimacode/aima-python/blob/master/rl.py
https://github.com/aimacode/aima-python/blob/master/tests/test_mdp.py
https://github.com/aimacode/aima-python/blob/master/tests/test_rl.py
'''
from time import time
from mdp import *
from rl import *


sr1 = -0.04
srg = +1
srp = -1
sro = None
small_MDP = GridMDP([[sr1,  sr1,  sr1,  sr1],
                     [sr1,  sro,  sr1,  srg],
                     [sr1,  sro,  sr1,  srp],
                     [sr1,  sr1,  sr1,  sr1]],
                     terminals=[(3, 2), (3, 1)],
                     init=(0,0), gamma = 0.9)
small_MDP.display_grid()

lr1 = -0.04
lrg = +1
lrp = -1
lro = None
large_MDP = GridMDP([[lr1,  lro,  lro,  lro,  lro,  lro,  lro,  lro,  lro,  lro,  lr1],
                     [lr1,  lro,  lr1,  lr1,  lr1,  lr1,  lr1,  lr1,  lr1,  lro,  lr1],
                     [lr1,  lro,  lr1,  lro,  lro,  lr1,  lro,  lro,  lr1,  lro,  lr1],
                     [lr1,  lro,  lr1,  lro,  lr1,  lr1,  lr1,  lro,  lr1,  lro,  lr1],
                     [lr1,  lro,  lr1,  lro,  lr1,  lrp,  lr1,  lro,  lr1,  lro,  lr1],
                     [lr1,  lro,  lr1,  lro,  lr1,  lrg,  lr1,  lro,  lr1,  lro,  lr1],
                     [lr1,  lro,  lr1,  lro,  lr1,  lr1,  lr1,  lro,  lr1,  lro,  lr1],
                     [lr1,  lro,  lr1,  lro,  lro,  lro,  lro,  lro,  lr1,  lro,  lr1],
                     [lr1,  lro,  lr1,  lr1,  lr1,  lr1,  lr1,  lr1,  lr1,  lro,  lr1],
                     [lr1,  lro,  lro,  lro,  lro,  lr1,  lro,  lro,  lro,  lro,  lr1],
                     [lr1,  lr1,  lr1,  lr1,  lr1,  lr1,  lr1,  lr1,  lr1,  lr1,  lr1]],
                     terminals=[(5, 5), (5, 6)],
                     init=(0,0), gamma = 0.9)
large_MDP.display_grid()

if 0: # Small MDP
    t0 = time()
    small_MDP_value_iteration, small_MDP_value_iteration_data = value_iteration(small_MDP, .01)
    print("small_MDP value iteration time = "+str(time()-t0))
    print(small_MDP.to_arrows(best_policy(small_MDP, small_MDP_value_iteration)))
    print("Plot iterations")
    keys = [key for key in small_MDP_value_iteration_data]
    for key in keys:
        print(key, end="|")
    print()
    for i in range(len(small_MDP_value_iteration_data[keys[0]])):
        for key in keys:
            print(small_MDP_value_iteration_data[key][i], end="|")
        print()

if 1: # Large MDP
    t0 = time()
    large_MDP_value_iteration, large_MDP_value_iteration_data = value_iteration(large_MDP, .01)
    print("large_MDP value iteration time = "+str(time()-t0))
    print(large_MDP.to_arrows(best_policy(large_MDP, large_MDP_value_iteration)))
    print("Plot iterations")
    keys = [key for key in large_MDP_value_iteration_data]
    for key in keys:
        print(key, end="|")
    print()
    for i in range(len(large_MDP_value_iteration_data[keys[0]])):
        for key in keys:
            print(large_MDP_value_iteration_data[key][i], end="|")
        print()


