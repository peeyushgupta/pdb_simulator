import numpy as np

from static.planner import OGPlanner, IGPlannerAlternative
from constants import *
from plan import TuplePlan


class OGDyanmicPlanner(object):

    def __init__(self, items, n, c):
        self.items = items
        self.n = n
        self.c = c

    def plan(self):
        static_ogp = OGPlanner(self.n)

        tuple_plans = []
        for i in range(len(self.items)):
            cost, path = static_ogp.plan(self.n, self.items[i].get_false_prob(), self.c)
            tuple_plans.append(TuplePlan(self.items[i].id, path, cost))

        return tuple_plans


class IGPDyanmicPlanner(object):

    def __init__(self, n, functions, cost, items, ddg):
        self.n = n
        self.functions = functions
        self.items = items
        self.versions = []
        for i in range(n + 1):
            self.versions.append(IGPlannerAlternative(i, n, cost, len(items), ddg))
        for i in range(n + 1):
            self.versions[i].set_versions(self.versions)
        self.v = self.versions
        self.c = cost

    def plan_item(self, t, current_version, last_version):
        return self.versions[current_version].plan_next_iteration(t, last_version)

    def evaluate(self, t, current_version, last_version):
        return self.versions[last_version].evaluate_tuple(t, current_version)
