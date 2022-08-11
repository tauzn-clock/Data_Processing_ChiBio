#!/bin/env python3
'''
This script attempts to look 
'''

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = str(Path(__file__).absolute().parents[0])
FILE_NAME = #FILE NAME GOES HERE

data = pd.read_csv(FILE_PATH + FILE_NAME, sep=',', header = None)
data = data.rename({0:"Time", 1:"OD", 39:"Rate"},axis = "columns")
print(data)

fig, axs = plt.subplots(2, 1)

#Top: Rate
axs[0].set_ylim([0.2,0.3])
axs[0].plot(data["Time"], data["OD"], "b-")

#Bottom: OD Readings
axs[1].set_ylim([-0.2,0.6])
axs[1].plot(data["Time"], data["Rate"], "g-")

plt.show()