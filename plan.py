class Plan(object):

    def __init__(self, items, function_map):
        self.items = items
        self.function_map = function_map


class TuplePlan(object):

    def __init__(self, item_id, path, cost):
        self.path = path
        self.cost = cost
        self.item_id = item_id