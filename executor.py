from plan import Plan


class BaseExecutor(object):

    def __init__(self, plan, budget):
        self.plan = plan
        self.budget = budget


class DefaultExectuor(BaseExecutor):

    def __init__(self, plan, budget):
        super().__init__(plan, budget)

    def run(self):
        items = self.plan.items
        function_map = self.plan.function_map
        N = len(items)
        true = []
        false = []
        maybe = []
        cost = 0

        # print(items)
        # print(function_map)
        for i,item in enumerate(items):
            for j in range(len(function_map[item.id])):
                function = function_map[item.id][j]
                cost += function.cost
                res = function.run_with_selectivity(items[i])

                if not function.is_last:
                    if res and j == len(function_map[item.id]):
                        maybe.append(items[i])
                    elif res:
                        continue
                    else:
                        false.append(items[i])
                        break
                else:
                    if res:
                        true.append(items[i])
                        break
                    else:
                        false.append(items[i])
                        break
        # print (cost, len(true), len(false), maybe, len(items))
        return (cost, true, false, maybe)


class ExecutorWithReordering(BaseExecutor):

    def __init__(self, plan, budget):
        super().__init__(plan, budget)
