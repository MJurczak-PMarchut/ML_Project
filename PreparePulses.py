import csv

import numpy as np
from matplotlib import pyplot as plt


def read_csv_data(file, max_size=1000):
    with open(file, "r") as pulse_file:
        csv_read = list(csv.reader(pulse_file))
    csv_read = [list(map(int, i)) for i in csv_read]
    return csv_read[:max_size]

def filter_pulse(pulse_data):
    ret_data = []
    ret_diff = []
    lead_time = []
    for pulse in pulse_data:
        if len(pulse) < 20:
            continue
        if (max(pulse) - min(pulse)) < 5:
            continue
        if max(pulse) > 54:
            continue
        idx = pulse.index(min(pulse))
        pulse_ret = [int(p) for p in pulse[idx - 10: idx + 20]]
        if min(np.gradient(pulse_ret[15:])) < -5:
            continue
        ret_data.append(pulse_ret)
    for pulse in ret_data:
        ret_diff.append((sum(pulse[0:5])/6) - min(pulse))
    for pulse in ret_data:
        ret_val = (sum(pulse[-5:])/5)*0.8
        idx = pulse.index(min(pulse))
        ival = [y > ret_val for y in pulse[idx:]].index(True)
        lead_time.append(ival)
    return ret_data  #, ret_diff, lead_time

if __name__ == '__main__':
    fig, ax = plt.subplots(3, 3)
    data= read_csv_data("CalRead\\790mV.csv")
    pd, diff, ival= filter_pulse(data)
    pd = np.asarray(pd)
    plt.figure(1)
    ax[0, 0].set_ylim([0, 63])
    ax[1, 0].set_ylim([0, 63])
    ax[2, 0].set_ylim([0, 20])
    ax[0, 0].set_title("790mV")
    ax[0, 0].plot(list(range(0, 30)), pd.T)
    ax[1, 0].plot(list(range(len(diff))), diff)
    ax[2, 0].plot(list(range(len(ival))), ival)
    data= read_csv_data("CalRead\\800mV.csv")
    pd, diff, ival= filter_pulse(data)
    pd = np.asarray(pd)
    # plt.figure(2)
    ax[0, 1].set_ylim([0, 63])
    ax[1, 1].set_ylim([0, 63])
    ax[2, 1].set_ylim([0, 20])
    ax[0, 1].set_title("800mV")
    ax[0, 1].plot(list(range(0, 30)), pd.T)
    ax[1, 1].plot(list(range(len(diff))), diff)
    ax[2, 1].plot(list(range(len(ival))), ival)
    data= read_csv_data("CalRead\\810mV.csv")
    pd, diff, ival = filter_pulse(data)
    pd = np.asarray(pd)
    # plt.figure(2)
    ax[0, 2].set_ylim([0, 63])
    ax[1, 2].set_ylim([0, 63])
    ax[2, 2].set_ylim([0, 20])
    ax[0, 2].set_title("810mV")
    ax[0, 2].plot(list(range(0, 30)), pd.T)
    ax[1, 2].plot(list(range(len(diff))), diff)
    ax[1, 2].plot(list(range(len(diff))), diff)
    ax[2, 2].plot(list(range(len(ival))), ival)
    plt.show()
