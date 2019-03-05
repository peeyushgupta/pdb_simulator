class Item(object):

    def __init__(self, id, state=0, is_candidate=True, selected=False):
        self.id = id
        self.state = state
        self.path = [0]
        self.is_candidate = is_candidate
        self.selected = selected

    def result(self, prob_matrix_true, num_functions, honest=True):

        if honest:
            if prob_matrix_true[self.id][num_functions-1] == 1:
                return True
            else:
                return False
        else:
            # Cannot say, since result depends upon the path the item takes
            pass