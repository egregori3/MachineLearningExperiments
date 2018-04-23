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


sr1 = 0
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
lrp = 0
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


#------------------------------------------------------------------------------
# Value Iterations
#------------------------------------------------------------------------------
if 1: # Small MDP
    t0 = time()
    small_MDP_value_iteration, small_MDP_value_iteration_data = value_iteration(small_MDP, .01)
    print("small_MDP value iteration time = "+str(time()-t0))
    print(small_MDP.to_arrows(best_policy(small_MDP, small_MDP_value_iteration)))
    if 1:
        print("Plot iterations")
        keys = [key for key in small_MDP_value_iteration_data]
        keys = [(0,1),(1,0)]
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
    if 1:
        print("Plot iterations")
#       keys = [key for key in large_MDP_value_iteration_data]
        keys = [(0,10),(10,10),(4,2),(6,2),(4,7),(6,7)]
        for key in keys:
            print(key, end="|")
        print()
        for i in range(len(large_MDP_value_iteration_data[keys[0]])):
            for key in keys:
                print(large_MDP_value_iteration_data[key][i], end="|")
            print()

#------------------------------------------------------------------------------
# Policy Iterations
#------------------------------------------------------------------------------
if 1: # Small MDP
    t0 = time()
    small_MDP_policy = policy_iteration(small_MDP, k=20)
    print("small_MDP policy iteration time = "+str(time()-t0))
    small_MDP.display_policy(small_MDP_policy)

if 1: # Large MDP
    t0 = time()
    large_MDP_policy = policy_iteration(large_MDP, k=20)
    print("large_MDP policy iteration time = "+str(time()-t0))
    large_MDP.display_policy(large_MDP_policy)

#------------------------------------------------------------------------------
# Reinforcement Learning
#------------------------------------------------------------------------------
if 1:
    t0 = time()
    q_agent = QLearningAgent(small_MDP, Ne=5, Rplus=2, alpha=lambda n: 60./(59+n))
    for i in range(200):
        run_single_trial(q_agent,small_MDP)
    print("small_MDP q-learning time = "+str(time()-t0))
    for q in q_agent.Q:
        print(q, q_agent.Q[q])
    q_agent.qtopolicy(small_MDP)
    print("Q-Matrix")
    q_agent.display(small_MDP)
    print("Policy")
    policy = q_agent.qtopolicy(small_MDP)
    small_MDP.display_policy(policy)

if 1:
    t0 = time()
    q_agent = QLearningAgent(large_MDP, Ne=5, Rplus=2, alpha=lambda n: 60./(59+n))
    for i in range(200):
        run_single_trial(q_agent,large_MDP)
    print("large_MDP q-learning time = "+str(time()-t0))
    for q in q_agent.Q:
        print(q, q_agent.Q[q])
    q_agent.qtopolicy(large_MDP)
    print("Q-Matrix")
    q_agent.display(large_MDP)
    print("Policy")
    policy = q_agent.qtopolicy(large_MDP)
    large_MDP.display_policy(policy)
