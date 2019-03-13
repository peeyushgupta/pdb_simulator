import matplotlib.pyplot as plt

def plot_selectivity(data):

    plt.plot(data, marker="o")

    plt.xlabel('Functions')
    plt.ylabel('Selectivity')
    plt.show()


def plot_cost(data):

    plt.plot(data, marker="o")

    plt.xlabel('Functions')
    plt.ylabel('Cost')
    plt.show()


def plot_norm_cost_selectivity(sel, cost):

    plt.plot(sel, marker="o")
    plt.plot(list(map(lambda x: x/max(cost), cost)), marker="^")

    plt.xlabel('Functions')
    plt.legend(['Selectivity', 'Cost'])
    plt.show()


def plot_selected(sel, cost, picked):
    plt.plot(sel, marker="o")
    plt.plot(list(map(lambda x: x / max(cost), cost)), marker="^")
    temp = [0]*len(sel)
    for i in range(len(picked)):
        print (int(picked[i]))
        temp[int(picked[i])] = sel[int(picked[i])]
    plt.plot(temp, 'rs', marker="s")

    plt.xlabel('Functions')
    plt.legend(['Selectivity', 'Cost', 'Picked'])
    plt.show()


def plot_results_with_epoch(true, false):
    plt.plot(true, )
    plt.plot(false,)

    plt.xlabel('Epochs')
    plt.legend(['True', 'False'])
    plt.show()


def plot_results_with_epoch_multi(true, false, true_s, false_s):
    plt.plot(true, )
    plt.plot(false,)

    plt.plot(true_s,)
    plt.plot(false_s)

    plt.xlabel('Epochs')
    plt.legend(['True(NS)', 'False(NS)', 'True(S)', 'False(S)'])
    plt.show()


def plot_epoch_times(times):
    plt.plot(times, marker="o")

    plt.xlabel('Epochs')
    plt.ylabel('Cumulative Cost')

    plt.show()