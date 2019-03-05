class OGPlanner(object):

    def __init__(self, n):
        self.n = n
        self.m = [0]*(n+1)
        self.c = [0]*(n+1)

    def plan(self, n, m, c):
        cost = 0
        included = []

        if n==1:
            cost = c[n]
            included = [n]
        else:
            C = [[0]*(n+1) for _ in range(n)]
            skip = [[0]*(n+1) for _ in range(n)]
            for k in range(n):
                C[k][n] = m[k] * c[n]
                skip[k][n] = False

            for l in range(n-1, 0, -1):
                for k in range(l):
                    notSkipCost = m[k] * c[l]+C[l][l+1]
                    skipCost = C[k][l+1]
                    C[k][l] = min(notSkipCost, skipCost)
                    skip[k][l] = (notSkipCost > skipCost)

            cost = C[0][1]
            included = [n]
            k = 0
            for l in range(1, n):
                if not skip[k][l]:
                    included = [l] + included
                    k = l

        return (cost, included)


Yes = 0
No = 1
Maybe = 2

class IGPlanner(object):

    def __init__(self, n):
        self.I = 0
        self.n = n
        self.v = [0]*(n+1)
        self.c = [0]*(n+1)
        self.OGP = OGPlanner(n)

    def evaluate(self, t):
        return (Yes, No, Maybe)[0]

    def resProb(self, t, j):
        return 0.5

    def tupleHandler(self, t, I, c, n, P, v, OGP):
        out  = []
        cost = 0
        included = []
        status = self.evaluate(t)
        if status==Yes or (status==Maybe and i == n):
            out.append(t)
        elif status==Maybe:
            if I==n-1:
                v[n].tupleHandler(t)
            else:
                P[I+1..n]
                for j in range(I+1, n+1):
                    P[j] = self.resProb(t, j)
                (cost, included) ← OGP.plan(1−P, c)
                next = min(included)
                v[next].tupleHandler(t)

