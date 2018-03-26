'''
CS7641 Assignment 3          Eric Gregori

To enable a simulation, change if 0: to if 1:

# PART 1 - Run EM and K-means on two datasets
# PART 2 - Run PCA, ICA, RP, ?? on two datasets
# PART 3 - Run EM and K-means on part 2 results
# PART 4 - Take part 2 results (one dataset) and run on ass1 NN
# PART 5 - 
'''

from LoadPreprocessDataset import LoadPreprocessDataset
from Visualize import Visualize
from part1_km import part1_km
from part1_em import part1_em

# -----------------------------------------------------------------------------
# Load datasets from ass 1 (unscaled)
# -----------------------------------------------------------------------------
datasets = LoadPreprocessDataset()

# -----------------------------------------------------------------------------
# Visualize DataSet - Scatter Matrix
# -----------------------------------------------------------------------------
if 0: # set to 1 to enable
    print(82 * '_')
    Visualize(datasets['wifi'])
    print(82 * '_')
    Visualize(datasets['letter'])

# -----------------------------------------------------------------------------
# PART 1 - Run EM and K-means on two datasets
# -----------------------------------------------------------------------------
if 0: # set to 1 to enable
    print(82 * '_')
    print("PART 1a - Run K-means on two datasets")
    print(82 * '_')
    print()
    part1km_wifi = part1_km(datasets['wifi'])
    print(82 * '_')
    print()
    part1km_letter = part1_km(datasets['letter'])

if 1: # set to 1 to enable
    print(82 * '_')
    print("PART 1b - Run EM on two datasets")
    print(82 * '_')
    print()
    part1em_wifi = part1_em(datasets['wifi'])
    print(82 * '_')
    print()
    part1em_letter = part1_em(datasets['letter'])


# -----------------------------------------------------------------------------
# PART 2 - Run PCA, ICA, RP, ?? on two datasets
# -----------------------------------------------------------------------------
if 0: # set to 1 to enable
    print(82 * '_')
    print("PART 2 - Run PCA, ICA, RP, ?? on two datasets")
    print(82 * '_')
    print()
    part2_wifi = part2(datasets['wifi'])
    print(82 * '_')
    print()
    part2_letter = part2(datasets['letter'])


print("Done")




