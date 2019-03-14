import numpy as np
import plot
import executor
import plan
import generator.uniform

from function import Function
from item import Item
from constants import *
from static.planner import OGPlanner, IGPlannerAlternative
from dynamic.planner import OGDyanmicPlanner


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

    def simulate_progressive_ogp(self, replay=False, shuffle=False):
        ogp = OGPlanner(self.num_functions)
        cost, path = ogp.plan(self.num_functions, self.func_sel, self.func_cost)
        functions = list(filter(lambda x: x.id in path, self.functions))
        true = [0]
        false = [0]
        maybe = [0]
        cumu_cost = [0]
        items_batch = int(self.budget/cost)
        epoch = 1

        if shuffle:
            np.random.shuffle(self.items)

        for i in range(0, self.num_items, items_batch):
            items = self.items[i:i+items_batch]
            function_map = {self.items[j].id:functions for j in range(i, min(i+items_batch, self.num_items))}

            epoch_plan = plan.Plan(items, function_map)
            exec = executor.DefaultExectuor(epoch_plan, self.budget)

            if not replay:
                epoch_cost, epoch_true, epoch_false, epoch_maybe = exec.run()
            else:
                epoch_cost, epoch_true, epoch_false, epoch_maybe = exec.replay()

            true.append(true[-1]+len(epoch_true))
            false.append(false[-1]+len(epoch_false))
            maybe.append(len(epoch_maybe))
            cumu_cost.append(cumu_cost[-1]+epoch_cost)
            epoch += 1


        print(true[-1], false[-1], maybe[-1], epoch, items_batch)
        return (true, false, maybe, cumu_cost)

    def simulate_progressive_dynamic_ogp(self, replay=False):
        dynamic_ogp = OGDyanmicPlanner(self.items, self.num_functions, self.func_cost)
        tuple_plans = sorted(dynamic_ogp.plan(), key=lambda x: x.cost)

        true = [0]
        false = [0]
        maybe = [0]
        cumu_cost = [0]
        epoch = 1
        p_count = 0
        start = 0

        while p_count<len(self.items):

            cost = 0
            start = p_count
            function_map ={}
            while p_count<len(self.items) and cost<self.budget:
                cost += tuple_plans[p_count].cost
                path = tuple_plans[p_count].path
                functions = list(filter(lambda x: x.id in path, self.functions))
                function_map[self.items[p_count].id] = functions
                p_count += 1

            items = self.items[start:p_count]

            epoch_plan = plan.Plan(items, function_map)
            exec = executor.DefaultExectuor(epoch_plan, self.budget)

            if not replay:
                epoch_cost, epoch_true, epoch_false, epoch_maybe = exec.run()
            else:
                epoch_cost, epoch_true, epoch_false, epoch_maybe = exec.replay()

            true.append(true[-1]+len(epoch_true))
            false.append(false[-1]+len(epoch_false))
            maybe.append(len(epoch_maybe))
            cumu_cost.append(cumu_cost[-1]+epoch_cost)
            epoch += 1


        print(true[-1], false[-1], maybe[-1], epoch)
        return (true, false, maybe, cumu_cost)


if __name__ == "__main__":

    sim = Simulator(10000, 10, 300000, 100)
    sim.init_simulation()
    ogp = OGPlanner(sim.num_functions)

    ans = ogp.plan(sim.num_functions, sim.func_sel, sim.func_cost)
    print (ans)
    plot.plot_selected(sim.func_sel, sim.func_cost, ans[1])

    results_s = sim.simulate_progressive_ogp(shuffle=False)
    # results_s = sim.simulate_progressive_ogp(replay=True, shuffle=True)
    # plot.plot_results_with_epoch_multi(results[0], results[1], results_s[0], results_s[1])
    # plot.plot_epoch_times(results[3])

    plot.plot_selectivity_item(list(map(lambda x: x.func_sel, sim.items[1::100])))
    results = sim.simulate_progressive_dynamic_ogp()
    plot.plot_results_with_epoch_multi(results[0], results[1], results_s[0], results_s[1])
    plot.plot_epoch_times(results[3])

    # num_items = 100000
    # num_func = 10
    # sim = Simulator(num_items, num_func, 50, 100)
    # sim.init_simulation()
    # versions = []
    # for i in range(num_func+1):
    #     versions.append(IGPlannerAlternative(i, num_func, sim.func_cost, num_items))
    # for i in range(num_func+1):
    #     versions[i].set_versions(versions)
    # for i in range(num_items):
    #     versions[0].plan(sim.items[i])
