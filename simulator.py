import numpy as np
import plot
import executor
import plan

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

    def start(self):
        for i in range(self.num_epochs):
            pass

    def run_epoch(self, epoch_id):
        pass

    def print_simulator_parameters(self):
        pass

    def simulate_progressive_ogp(self):
        ogp = OGPlanner(self.num_functions)
        cost, path = ogp.plan(self.num_functions, self.func_sel, self.func_cost)
        functions = list(filter(lambda x: x.id in path, self.functions))
        true = [0]
        false = [0]
        maybe = [0]
        cumu_cost = [0]

        items_batch = int(self.budget/cost)

        epoch = 1
        for i in range(0, self.num_items, items_batch):
            items = self.items[i:i+items_batch]
            function_map = {j:functions for j in range(i, i+items_batch)}

            epoch_plan = plan.Plan(items, function_map)
            exec = executor.DefaultExectuor(epoch_plan, self.budget)
            epoch_cost, epoch_true, epoch_false, epoch_maybe = exec.run()

            true.append(true[-1]+len(epoch_true))
            false.append(false[-1]+len(epoch_false))
            maybe.append(len(epoch_maybe))
            cumu_cost.append(cumu_cost[-1]+epoch_cost)
            epoch += 1


        print(true[-1], false[-1], maybe[-1], epoch, items_batch)
        return (true, false, maybe, cumu_cost)


if __name__ == "__main__":

    sim = Simulator(10000, 10, 300000, 100)
    sim.init_simulation()
    ogp = OGPlanner(sim.num_functions)

    # plot.plot_selectivity(sim.func_sel)
    # plot.plot_cost(sim.func_cost)
    # plot.plot_norm_cost_selectivity(sim.func_sel, sim.func_cost)
    #
    # print (sim.func_sel)
    # print(sim.func_cost)
    ans = ogp.plan(sim.num_functions, sim.func_sel, sim.func_cost)
    print (ans)
    plot.plot_selected(sim.func_sel, sim.func_cost, ans[1])

    results = sim.simulate_progressive_ogp()
    plot.plot_results_with_epoch(results[0], results[1])
    plot.plot_epoch_times(results[3])

