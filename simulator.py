import numpy as np

from function import Function
from item import Item
from constants import *
from static.planner import OGPlanner

class Simulator(object):

    def __init__(self, num_items, num_functions, budget, num_epochs, num_threads=1, filter=0.2):
        # num_types and num_threads are 1 for now
        # needs to be change later

        self.num_items = num_items
        self.num_functions = num_functions
        self.budget = budget
        self.num_epochs = num_epochs
        self.num_threads = num_threads
        self.filter = filter

        self.items = []
        self.functions = []
        self.transition_prob = []
        self.epoch_weights = []
        self.func_sel = []
        self.func_cost = []

    def init_simulation(self):
        self.create_items()
        self.create_func_sel()
        self.create_func_costs()
        self.create_functions()
        self.create_prob_matrix_true()
        self.create_prob_matrix_resolve()
        self.generate_epoch_weights()

    def create_items(self):
        for i in range(self.num_items):
            self.items.append(Item(i))

    def create_func_sel(self):
        self.func_sel = [0]*(self.num_functions+1)
        self.func_sel[0] = 1
        self.func_sel[self.num_functions] = self.filter
        max_gap = (1-self.filter)/(self.num_functions-1)
        min_gap = max_gap/50

        for i in range(self.num_functions-1, 0, -1):
            self.func_sel[i] = np.random.uniform(self.func_sel[i+1]+min_gap, self.func_sel[i+1]+max_gap)
            max_gap = (1 - self.func_sel[i]) / i
            min_gap = max_gap / 50

    def create_func_costs(self):
        self.func_cost = [0]*(self.num_functions+1)
        self.func_cost[0] = 0
        self.func_cost[self.num_functions] = MAX_FUNC_COST
        max_gap = (MAX_FUNC_COST-MIN_FUNC_COST)/self.num_functions
        min_gap = max_gap/1.5

        for i in range(self.num_functions-1, 0, -1):
            self.func_cost[i] = np.random.uniform(self.func_cost[i+1]-max_gap, self.func_cost[i+1]-min_gap)
            max_gap = (self.func_cost[i] - MIN_FUNC_COST) / i
            min_gap = max_gap / 1.2

    def create_functions(self):
        self.functions.append(Function(0, 0, 1, dummy=True))
        for i in range(1, self.num_functions+1):
            if i==1:
                self.functions.append(
                    Function(i, self.func_cost[i], self.func_sel[i], is_first=True))
            else:
                self.functions.append(
                    Function(i, self.func_cost[i], self.func_sel[i], is_last=True, prev_func=self.functions[i - 1]))

            self.functions[i-1].next_func = self.functions[i]

    def create_prob_matrix_true(self):
        for i in range(self.num_items):
            pass

    def create_prob_matrix_resolve(self):
        for i in range(self.num_items):
            pass

    def generate_epoch_weights(self):
        self.epoch_weights = [0] * (self.num_epochs)
        self.epoch_weights[0] = MAX_EPOCH_WT
        max_gap = (MAX_EPOCH_WT - MIN_EPOCH_WT) / self.num_epochs

        for i in range(1, self.num_epochs):
            self.epoch_weights[i] = np.random.uniform(self.epoch_weights[i - 1] - max_gap, self.epoch_weights[i - 1])

    def start(self):
        for i in range(self.num_epochs):
            pass

    def run_epoch(self, epoch_id):
        pass

    def print_simulator_parameters(self):
        pass

if __name__ == "__main__":

    sim = Simulator(10000, 50, 50, 100)
    sim.init_simulation()
    ogp = OGPlanner(sim.num_functions)
    print (sim.func_sel)
    print(sim.func_cost)
    print (ogp.plan(sim.num_functions, sim.func_sel, sim.func_cost))

