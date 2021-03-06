class Item(object):

    def __init__(self, id, state=0, is_candidate=True, selected=False, func_sel=None):
        self.id = id
        self.state = state
        self.path = [0]
        self.is_candidate = is_candidate
        self.selected = selected
        self.func_sel = func_sel
        self.lineage = {}

    def result(self, prob_matrix_true, num_functions, honest=True):

        if honest:
            if prob_matrix_true[self.id][num_functions-1] == 1:
                return True
            else:
                return False
        else:
            # Cannot say, since result depends upon the path the item takes
            pass

    def get_false_prob(self):
        return self.func_sel