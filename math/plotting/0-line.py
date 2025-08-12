#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():
    y = np.arange(0, 11) ** 3
    
    # plot save
    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')
    plt.savefig('0-line.png');
