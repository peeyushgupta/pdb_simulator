import numpy as np
import plot
import executor
import plan
from generator.uniform import UniformDataGenerator

from function import Function
from generator.default import DefaultDataGenerator
from item import Item
from constants import *
from static.planner import OGPlanner, IGPlannerAlternative
from dynamic.planner import OGDyanmicPlanner, IGPDyanmicPlanner


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
        self.data_generator = UniformDataGenerator(
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


        print (true[-1], false[-1], maybe[-1], epoch)
        return (true, false, maybe, cumu_cost)

    def simulate_progressive_dynamic_agp(self, ddg):
        dynamic_agp = IGPDyanmicPlanner(self.num_functions, self.functions, self.func_cost, self.items, ddg)
        tuples = []
        for i in range(self.num_items):
            t, prev, tmp_path, tmp_cost = dynamic_agp.plan_item(self.items[i], 0, 0)
            tuples.append((tmp_cost, t, prev, tmp_path, []))
        tuples.sort(key=lambda x: x[0])
        true = [0]
        false = [0]
        maybe = [0]
        cumu_cost = [0]
        epoch = 1

        while 0 < len(tuples):
            epoch_cost = 0
            epoch_true = 0
            epoch_false = 0
            epoch_maybe = 0
            while 0 < len(tuples) and epoch_cost <= self.budget:
                tmp_path = tuples[0][3]
                tmp_tuple = tuples.pop(0)
                current_version = min(tmp_path)
                last_version = tmp_tuple[2]
                tmp_tuple[4].append(current_version)
                epoch_cost += self.func_cost[current_version]
                tmp_status = dynamic_agp.evaluate(t, current_version, last_version)
                if tmp_status == YES:
                    epoch_true += 1
                elif tmp_status == NO:
                    epoch_false += 1
                else:
                    epoch_maybe += 1
                    t, prev, tmp_path, tmp_cost = dynamic_agp.plan_item(tmp_tuple[1], current_version, last_version)
                    if len(tmp_path) != 0:
                        tuples.insert(0, (tmp_cost, t, prev, tmp_path, tmp_tuple[4]))
                tuples.sort(key=lambda x: x[0])
            true.append(true[-1] + epoch_true)
            false.append(false[-1] + epoch_false)
            maybe.append(epoch_maybe)
            cumu_cost.append(cumu_cost[-1] + epoch_cost)
            epoch += 1
        print(true[-1], false[-1], maybe[-1], epoch)
        return (true, false, maybe, cumu_cost)


if __name__ == "__main__":

    sim = Simulator(10000, 10, 300000, 100)
    sim.init_simulation()
    #ogp = OGPlanner(sim.num_functions)
    #ans = ogp.plan(sim.num_functions, sim.func_sel, sim.func_cost)
    #print (ans)
    #plot.plot_selected(sim.func_sel, sim.func_cost, ans[1])

    #results_s = sim.simulate_progressive_ogp(shuffle=False)
    # results_s = sim.simulate_progressive_ogp(replay=True, shuffle=True)
    # plot.plot_results_with_epoch_multi(results[0], results[1], results_s[0], results_s[1])
    # plot.plot_epoch_times(results[3])

    #plot.plot_selectivity_item(list(map(lambda x: x.func_sel, sim.items[1::100])))
    #results = sim.simulate_progressive_dynamic_ogp()
    #plot.plot_results_with_epoch_multi(results[0], results[1], results_s[0], results_s[1])
    #plot.plot_epoch_times(results[3])

    num_items = 10000
    num_func = 10
    sim = Simulator(num_items, num_func, 300000, 100)
    sim.init_simulation()
    ddg = DefaultDataGenerator(num_items, num_func, 100, 0.4)
    results = sim.simulate_progressive_dynamic_agp(ddg)
    plot.plot_dynamic_results_with_epoch_multi(results[0], results[1])
    print(results[0], results[1], results[2])
    plot.plot_epoch_times(results[3])
