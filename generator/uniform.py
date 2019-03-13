from constants import *
from item import Item
from function import Function
from generator.default import DefaultDataGenerator

import numpy as np


class UniformDataGenerator(DefaultDataGenerator):

    def __init__(self, num_items, num_functions, epochs, filter_):
        super().__init__(num_items, num_functions, epochs, filter_)

    def create_items(self):
        for i in range(self.num_items):
            self.items.append(Item(i))

    def create_func_sel(self):
        self.func_sel = [0]*(self.num_functions+1)
        self.func_sel[0] = 1
        self.func_sel[self.num_functions] = self.filter
        max_gap = (1-self.filter)/(self.num_functions-1)/100
        min_gap = 0

        for i in range(self.num_functions-1, 0, -1):
            self.func_sel[i] = np.random.uniform(self.func_sel[i+1]+min_gap, self.func_sel[i+1]+max_gap)
            max_gap = (1 - self.func_sel[i]) / i
            min_gap = max_gap / 50

    def create_func_costs(self):
        self.func_cost = [0]*(self.num_functions+1)
        self.func_cost[0] = 0
        self.func_cost[self.num_functions] = MAX_FUNC_COST
        max_gap = (MAX_FUNC_COST-MIN_FUNC_COST)/self.num_functions
        min_gap = max_gap/2

        for i in range(self.num_functions-1, 0, -1):
            self.func_cost[i] = np.random.uniform(self.func_cost[i+1]-max_gap, self.func_cost[i+1]-min_gap)
            max_gap = (self.func_cost[i] - MIN_FUNC_COST) / i
            min_gap = max_gap / 1.5

    def create_functions(self):
        self.functions.append(Function(0, 0, 1, dummy=True))
        for i in range(1, self.num_functions+1):
            if i==1:
                self.functions.append(
                    Function(i, self.func_cost[i], self.func_sel[i], is_first=True))
            else:
                self.functions.append(
                    Function(i, self.func_cost[i], self.func_sel[i], is_last=False, prev_func=self.functions[i - 1]))

            self.functions[i-1].next_func = self.functions[i]
        self.functions[self.num_functions].is_last = True

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
