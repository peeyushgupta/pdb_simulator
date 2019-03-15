import numpy as np


class DefaultDataGenerator(object):

    def __init__(self, num_items, num_functions, epochs, filter_):
        self.items = []
        self.functions = []
        self.transition_prob = []
        self.epoch_weights = []
        self.func_sel = []
        self.func_cost = []

        self.num_items = num_items
        self.num_functions = num_functions
        self.filter = filter_
        self.num_epochs = epochs

    def run(self):
        self.create_items()
        self.create_func_sel()
        self.create_func_costs()
        self.create_functions()
        self.create_prob_matrix_true()
        self.create_prob_matrix_resolve()
        self.generate_epoch_weights()

        return self.items, self.functions, self.transition_prob, self.epoch_weights, self.func_sel, self.func_cost

    def create_items(self):
        pass

    def create_func_sel(self):
        pass

    def create_func_costs(self):
        pass

    def create_functions(self):
        pass

    def create_prob_matrix_true(self):
        pass

    def create_prob_matrix_resolve(self):
        pass

    def create_prob_row_resolve(self, prev_prob_row,start=0, low=0.0, high=1.0):
        res_prob_row = [0.0] * (self.num_functions + 1)
        for i in range(start + 1, self.num_functions + 1):
            low = max(res_prob_row[i - 1], prev_prob_row[i])
            res_prob_row[i] = np.random.uniform(low, high)
        return res_prob_row

    def generate_epoch_weights(self):
        pass