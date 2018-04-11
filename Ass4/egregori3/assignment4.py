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
                     terminals=[(3, 2), (3, 1)])
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
                     terminals=[(5, 5), (5, 6)])
large_MDP.display_grid()

# small MDP
t0 = time()
small_MDP_solve_using_value = value_iteration(small_MDP, .01)
print("small_MDP_solve_using_value = "+str(time()-t0))
t0 = time()
small_MDP_solve_using_policy = policy_iteration(small_MDP)
print("small_MDP_solve_using_policy = "+str(time()-t0))

# large MDP
t0 = time()
large_MDP_solve_using_value = value_iteration(large_MDP, .01)
print("large_MDP_solve_using_value = "+str(time()-t0))
t0 = time()
large_MDP_solve_using_policy = policy_iteration(large_MDP)
print("large_MDP_solve_using_policy = "+str(time()-t0))


# small_MDP_best_value_policy = best_policy(small_MDP, small_MDP_solve_using_value)



# for row in small_MDP.to_arrows(small_pi):
#     print(row)
