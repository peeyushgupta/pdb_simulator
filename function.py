import time
import random

import constants


class Function(object):

    def __init__(self, id, cost, selectivity, is_first=False, is_last= False, prev_func=None, next_func=None,
                 name=None, func=None, dummy=False):
        self.id = id
        self.cost = cost
        self.is_first = is_first
        self.is_last = is_last
        self.prev_func = prev_func
        self.next_func = next_func
        self.name = name
        self.func = func
        self.selectivity = selectivity
        self.dummy = dummy

    def run(self, item, transition_prob, params=None):
        # For simulation purposes just sleep
        # should change to actual function call later
        # time.sleep(self.cost*1.0/constants.SIMULATION_TIME_MUL)

        res = random.random() < transition_prob[item.state][self.id]
        item.lineage[self.id] = res

        if res == True:
            item.state = self.id
        else:
            item.state = -1

        return res

    def run_with_selectivity(self, item):
        # For simulation purposes just sleep
        # should change to actual function call later
        # time.sleep(self.cost*1.0/constants.SIMULATION_TIME_MUL)

        res = random.random() > self.selectivity
        item.lineage[self.id] = res

        if res == True:
            item.state = self.id
        else:
            item.state = -1

        return res