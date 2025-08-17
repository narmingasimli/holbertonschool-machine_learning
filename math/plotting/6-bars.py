#!/usr/bin/env python3
""" Module example for stacked bar graph """
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar graph:"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    t = np.arange(3)
    bar1 = plt.bar(t, fruit[0], 0.5, color='red')
    bar2 = plt.bar(t, fruit[1], 0.5,
                   color='yellow', bottom=fruit[0])
    bar3 = plt.bar(t, fruit[2], 0.5, color='#ff8000',
                   bottom=fruit[0]+fruit[1])
    bar4 = plt.bar(t, fruit[3], 0.5, color='#ffe5b4',
                   bottom=fruit[0]+fruit[1]+fruit[2])
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(t, ('Farrah', 'Fred', 'Felicia'))
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.legend((bar1, bar2, bar3, bar4),
               ('apples', 'bananas', 'oranges', 'peaches'))
    plt.tight_layout()
    plt.show()

bars()
