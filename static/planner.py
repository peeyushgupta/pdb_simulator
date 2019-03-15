import numpy as np

from constants import *


class OGPlanner(object):

    def __init__(self, n):
        self.n = n
        self.m = [0] * (n + 1)
        self.c = [0] * (n + 1)

    def plan(self, n, m, c):
        cost = 0
        included = []

        if n == 1:
            cost = c[n]
            included = [n]
        else:
            C = [[0] * (n + 1) for _ in range(n)]
            skip = [[0] * (n + 1) for _ in range(n)]
            for k in range(n):
                C[k][n] = m[k] * c[n]
                skip[k][n] = False

            for l in range(n - 1, 0, -1):
                for k in range(l):
                    notSkipCost = m[k] * c[l] + C[l][l + 1]
                    skipCost = C[k][l + 1]
                    C[k][l] = min(notSkipCost, skipCost)
                    skip[k][l] = (notSkipCost > skipCost)

            cost = C[0][1]
            included = []
            k = 0
            for l in range(1, n):
                if not skip[k][l]:
                    included.append(l)
                    k = l

            included.append(n)
        return (cost, included)


class IGPlanner(object):

    def __init__(self, n, functions, prob_matrix_resolve):
        self.I = 0
        self.n = n
        self.v = functions
        self.c = [0] * (n + 1)
        self.OGP = OGPlanner(n)
        self.pmat_resolve = prob_matrix_resolve

    def evaluate(self, t):
        return np.random.randint(0, 2)

    def resProb(self, t, j, I, pmat_resolve):
        return pmat_resolve[t][I][j]

    def tupleHandler(self, t, I, c, n, v, path, OGP):
        cost = 0
        included = []
        path.append(I)
        status = self.evaluate(t)
        if status == YES or (status == MAYBE and I == n):
            return (t, YES, cost, path)

        elif status == MAYBE:
            if I == n - 1:
                v[n].tupleHandler(t)
            else:
                P = [0.1] * (n - (I + 1) + 1)
                for j in range(I + 1, n + 1):
                    P[j - (I + 1)] = self.pmat_resolve[t][I][j]
                (cost, included) = OGP.plan(map(lambda x: 1 - x, P), c[I + 1:])
                next = min(included) + (I + 1)
                v[next].tupleHandler(t)
        else:
            return (t, NO)

    def plan(self, t, n, c, v):
        path = []
        print(self.tupleHandler(t, 0, c, n, ))
        print(path)


class IGPlannerAlternative(object):

    def __init__(self, i, n, cost, num_items, ddg):
        self.i = i
        self.n = n
        self.v = None
        self.c = cost
        self.num_items = num_items
        self.OGP = OGPlanner(n)
        self.pmat_resolve = [[0] * (n+1) for _ in range(num_items)]
        self.ddg = ddg

    def set_versions(self, functions):
        self.v = functions

    def set_res_prob(self, t, start, last_executed_func=0):
        low = 0.0
        high = 0.8
        prev_res_prob_row = [0.0] * (self.n + 1)
        if last_executed_func != 0:
            prev_res_prob_row = self.v[last_executed_func].pmat_resolve[t.id]
        self.pmat_resolve[t.id] = self.ddg.create_prob_row_resolve(prev_res_prob_row,start, low, high)

    def evaluate(self, t, p):
        ans = np.random.uniform()
        if ans <= p:
            if ans <= p/2:
                return YES
            else:
                return NO
        else:
            return MAYBE

    def evaluate_tuple(self, t, current_version):
        p = self.pmat_resolve[t.id][current_version]
        return self.evaluate(t, p)

    def tupleHandler(self, t, path, cost, last_executed_func=0):
        included = []
        path.append(self.i)
        if self.i == 0:
            status = MAYBE
        else:
            status = self.evaluate(t, self.v[last_executed_func].pmat_resolve[t.id][self.i])
        cost = cost + self.c[self.i]
        if status == YES or (status == MAYBE and self.i == self.n):
            return t, YES, path, cost, self.i, 0

        elif status == MAYBE:
            if self.i == self.n-1:
                self.set_res_prob(t, self.i, last_executed_func)
                next_v = self.i+1
                return t, MAYBE, path, cost, self.i, next_v
            else:
                self.set_res_prob(t, self.i, last_executed_func)
                p_list = [1.0]
                for j in range(self.i+1, self.n+1):
                    p_list.append(1 - (self.pmat_resolve[t.id][j]))
                cost_list = [0.0]
                for j in range(self.i+1, self.n+1):
                    cost_list.append(self.c[j])
                (tmp_cost, included) = self.OGP.plan((len(p_list)-1), p_list, cost_list)
                next_v = min(included) + self.i
                return t, MAYBE, path, cost, self.i, next_v
        else:
            return t, NO, path, 0, 0, 0

    def plan_one_iteration(self, t, path, cost, prev_version, next_version):
        return self.v[next_version].tupleHandler(t, path, cost, prev_version)

    def plan_next_iteration(self, t, last_executed_func):
        path = []
        if self.i == self.n:
            return t, self.i, path, 0
        elif self.i == self.n - 1:
            self.set_res_prob(t, self.i, last_executed_func)
            next_v = self.i + 1
            path.append(next_v)
            return t, self.i, path, self.c[next_v]
        else:
            self.set_res_prob(t, self.i, last_executed_func)
            p_list = [1.0]
            for j in range(self.i + 1, self.n + 1):
                p_list.append(1 - (self.pmat_resolve[t.id][j]))
            cost_list = [0.0]
            for j in range(self.i + 1, self.n + 1):
                cost_list.append(self.c[j])
            (tmp_cost, path) = self.OGP.plan((len(p_list)-1), p_list, cost_list)
            return t, self.i, list(map(lambda x: x+self.i, path)), tmp_cost

    def plan(self, t):
        path = []
        cost = 0
        status = MAYBE
        prev_version = 0
        next_version = 0
        while status == MAYBE:
            (t, status, path, cost, prev_version, next_version) = self.plan_one_iteration(t, path, cost, prev_version, next_version)
        return t, status, path, cost
