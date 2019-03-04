class Item(object):

    def __init__(self, id, state=0):
        self.id = id
        self.state = state
        self.path = [0]
		self.is_candidate = True