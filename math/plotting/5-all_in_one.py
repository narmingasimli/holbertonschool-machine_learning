#!/usr/bin/env python3
"""
Plots five different graphs in a single figure using a 3x2 grid layout.
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Generates and displays a figure containing five subplots:
    1. A line graph of y = x^3
    2. A scatter plot of Men's Height vs Weight
    3. A line graph of Exponential Decay of C-14 (logarithmic y-axis)
    4. A line graph of Exponential Decay of Radioactive Elements (C-14 and Ra-226)
    5. A histogram of student grades
    """
    x0 = np.arange(0, 11)
    y0 = x0 ** 3

    mean1 = [69, 0]
    cov1 = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    fig = plt.figure(figsize=(10, 12)) # Adjust figure size for better readability
    gs = fig.add_gridspec(3, 2) # 3 rows, 2 columns grid
    fig.suptitle('All in One', fontsize='large')
   
    ax0 = fig.add_subplot(gs[0, 0]) # Top-left subplot
    ax0.plot(x0, y0)
    ax0.set_xlabel('x', fontsize='x-small')
    ax0.set_ylabel('y', fontsize='x-small')
    ax0.set_title('x^3', fontsize='x-small') # Custom title for this generic plot
  
    ax1 = fig.add_subplot(gs[0, 1]) # Top-right subplot
    ax1.scatter(x1, y1, color='magenta')
    ax1.set_xlabel('Height (in)', fontsize='x-small')
    ax1.set_ylabel('Weight (lbs)', fontsize='x-small')
    ax1.set_title("Men's Height vs Weight", fontsize='x-small')
   
    ax2 = fig.add_subplot(gs[1, 0]) # Middle-left subplot
    ax2.plot(x2, y2)
    ax2.set_xlabel('Time (years)', fontsize='x-small')
    ax2.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax2.set_title('Exponential Decay of C-14', fontsize='x-small')
    ax2.set_yscale('log') # Logarithmic y-axis
    ax2.set_xlim(0, 28650) # Set x-axis range
   
    ax3 = fig.add_subplot(gs[1, 1]) # Middle-right subplot
    ax3.plot(x3, y31, 'r--', label='C-14') # Dashed red line for C-14
    ax3.plot(x3, y32, 'g-', label='Ra-226') # Solid green line for Ra-226
    ax3.set_xlabel('Time (years)', fontsize='x-small')
    ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax3.set_title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    ax3.set_xlim(0, 20000) # Set x-axis range
    ax3.set_ylim(0, 1) # Set y-axis range
    ax3.legend(loc='upper right', fontsize='x-small') # Legend in upper right

    ax4 = fig.add_subplot(gs[2, :]) # gs[row, :] means span all columns in that row
    bins = np.arange(0, 101, 10) # Bins every 10 units from 0 to 100
    ax4.hist(student_grades, bins=bins, edgecolor='black') # Bars outlined in black
    ax4.set_xlabel('Grades', fontsize='x-small')
    ax4.set_ylabel('Number of Students', fontsize='x-small')
    ax4.set_title('Project A', fontsize='x-small')
    ax4.set_xlim(0, 100) # Ensure x-axis range is 0 to 100

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle

    plt.show()


if __name__ == '__main__':
    all_in_one()

