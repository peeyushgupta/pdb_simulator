import constants
import function
import item


class Simulator(object):

    def __init__(self, num_items, num_functions, budget, num_epochs, num_threads=1):
        # num_types and num_threads are 1 for now
        # needs to be change later

        self.num_items = num_items
        self.num_functions = num_functions
        self.budget = budget
        self.num_epochs = num_epochs
        self.num_types = num_types
        self.num_threads = num_threads

        self.items = []
        self.functions = []
        self.transition_prob = []
        self.weights = []

    def init_simulation(self):
        self.create_items()
        self.create_functions()
        self.create_prob_matrix()
        self.generate_epoch_weights()

    def create_items(self):
        pass

    def create_functions(self):
        pass

    def create_prob_matrix(self):
        pass

    def generate_epoch_weights(self):
        pass

    def start(self):
        for i in range(self.num_epochs):
            pass

    def run_epoch(self, epoch_id):
        pass



if __name__ == "__main__":

    sim = Simulator(10000, 10, 50, 100)
    sim.init_simulation()
