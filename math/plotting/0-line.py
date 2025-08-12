#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    y = np.arange(0, 11) ** 3
    plt.figure()
    plt.plot(y, 'r-')
    plt.xlim([0, 10])
    plt.savefig('0-line.png');
