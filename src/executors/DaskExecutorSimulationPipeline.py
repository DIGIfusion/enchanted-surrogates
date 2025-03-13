"""
# executors/DaskExecutor.py
Contains logic for executing surrogate workflow on Dask.
"""

import time
import os

class DaskExecutorSimulationPipeline():
    """
    Handles the consecutive running of DaskExecutorSimulations. 
    This allows different simulations to be ran in series,
    one output being the input to the next, 
    each can be ran on a different cluster with different resources per worker.

    Attributes:
        simulations: List of simulations to run
    """
    def __init__(self, simulations: list):
        self.simulations = simulations
     
    def start_pipeline(self):
        print(100 * "=")
        print('STARTING PIPELINE')
        
        for simulation in self.simulations:
            simulation.start_runs() 