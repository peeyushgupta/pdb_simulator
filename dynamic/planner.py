import numpy as np

from static.planner import OGPlanner
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

