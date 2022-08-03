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
#Have a sudden dip in growth rate
#FILE_NAME = "/2021-01-22 11_40_02_M0_data.csv"
#FILE_NAME = "/2021-01-22 11_40_04_M1_data.csv"
#FILE_NAME = "/2021-01-22 11_40_06_M2_data.csv"
#FILE_NAME = "/2021-01-22 11_40_07_M3_data.csv"

#Comparatively constant growth ratee
#FILE_NAME = "/2021-01-22 15_49_55_M0_data.csv"
#FILE_NAME = "/2021-01-22 15_50_37_M1_data.csv"
#FILE_NAME = "/2021-01-22 15_51_13_M2_data.csv"
FILE_NAME = "/2021-01-22 15_51_33_M3_data.csv"

data = pd.read_csv(FILE_PATH + FILE_NAME, sep=',', header = None)
data = data.rename({0:"Time", 1:"OD", 39:"Rate"},axis = "columns")
print(data)

fig, axs = plt.subplots(2, 1)

#Top: Rate
axs[0].set_ylim([0.2,0.3])
axs[0].plot(data["Time"], data["OD"], "b-")

axs[1].plot(data["Time"], data[38], "g-")
#24 switches something on???
"""
#Bottom: OD Readings
axs[1].set_ylim([-0.2,0.6])
axs[1].plot(data["Time"], data["Rate"], "g-")
"""
plt.show()