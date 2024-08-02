import os
import sys
sys.path.append('./')
import numpy as np
from simulation_loader import SimulationLoader

regions = [
    'beaverhead_500',
    'beaverhead_750',
    'beaverhead_1000',
    'cabinet_500',
    'cabinet_750',
    'cabinet_1000',
    'mission_500',
    'mission_750',
    'mission_1000',
    'pintlers_500',
    'pintlers_750',
    'pintlers_1000'
]

for region in regions:
    sim_loader = SimulationLoader(region)
    sim_loader.load_features_from_h5()
    sim_loader.save_feature_arrays()