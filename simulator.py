import numpy as np
import plot
import executor
import plan
import generator.uniform

from function import Function
from item import Item
from constants import *
from static.planner import OGPlanner

class Simulator(object):

    def __init__(self, num_items, num_functions, budget, num_epochs, num_threads=DEFAULT_NUM_THREADS, filter=LAST_FUNC_SEL):
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

        self.data_generator = generator.uniform.UniformDataGenerator(
            self.num_items, self.num_functions, self.num_epochs, self.filter)

    def init_simulation(self):
        self.items, self.functions, self.transition_prob, self.epoch_weights, self.func_sel, self.func_cost = \
            self.data_generator.run()

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

    sim = Simulator(10000, 10, 30000, 100)
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

