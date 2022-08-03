#!/bin/env python3

from pathlib import Path

FILE_PATH = str(Path(__file__).absolute().parents[0])
#Have a sudden dip in growth rate
FILE_NAME = "/2021-01-22 11_40_02_M0_data.csv"
#FILE_NAME = "/2021-01-22 11_40_04_M1_data.csv"
FILE_NAME = "/2021-01-22 11_40_06_M2_data.csv"
#FILE_NAME = "/2021-01-22 11_40_07_M3_data.csv"

#Comparatively constant growth ratee
#FILE_NAME = "/2021-01-22 15_49_55_M0_data.csv"
#FILE_NAME = "/2021-01-22 15_50_37_M1_data.csv"
#FILE_NAME = "/2021-01-22 15_51_13_M2_data.csv"
#FILE_NAME = "/2021-01-22 15_51_33_M3_data.csv"

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ekf

data = pd.read_csv(FILE_PATH + FILE_NAME, sep=',', header = None)
data = data.rename({0:"Time", 1:"OD", 39:"Rate"},axis = "columns")
print(data)

##################################
#       Innitialisation          #
##################################

x = np.asarray([data.iloc[0]["OD"],0]) #Previous State
x_cur = np.asarray([data.iloc[0]["OD"],0]) #Real Current State
row_prev = data.iloc[0]
dillution_cnt = 5 #A skip counter to ignore OD fluctuations post dillution
error_reading_cnt = 0 #A skip counter when an errornous_reading is detected

P = np.asarray([[10, 0],
                [0, 10]])


##################################
#              Main              #
##################################

results = []
results_header = ["Time", "OD_predict", "Rate_predict", "Var_OD", "Var_Rate", "OD_actual", "Rate_actual","Rate","Rate_Pred_Hour"]

for index, row in data.iterrows():

    #Prevents us from using the first variable twice
    if (row["Time"]==row_prev["Time"]): 
        continue

    # Increase trust_cnt when a dillution occurs
    if (row[10] != 0):
        dillution_cnt = 5

    # Increase error_reading_cnt when bubbling occurs
    if (row["OD"]<0): 
        error_reading_cnt = 3 

    #Process the current row
    dt = (row["Time"]-row_prev["Time"])
    x_cur = [row_prev["OD"], (math.log(max(1e-16,row["OD"])) - math.log(max(1e-16,row_prev["OD"])))/dt] #The max is there to catch any ODs that are negative

    if (dillution_cnt > 0):
        #Use new OD as the stating pt post dillution
        x[0] = x_cur[0] 
        dillution_cnt -= 1
    elif (error_reading_cnt > 0):
        #Extrapolate actual value
        x[0] = x[0] * math.exp(x[1]*dt) 
        error_reading_cnt -= 1
    else:
        #Run Kalman Filter
        x,P = ekf.ekf(x,x_cur,dt,P)

    #Update State
    rate_pred_hour = x[1]*3600
    results.append([row["Time"],x[0],x[1],P[0][0],P[1][1],x_cur[0], x_cur[1],row["Rate"],rate_pred_hour])
    row_prev = row

results = pd.DataFrame(results,columns = results_header)
print(results)


###########################################
#              Visualisation              #
###########################################


fig, axs = plt.subplots(3, 1)

#axs[0].set_ylim([0.8,1.5])
#axs[0].plot(results["Time"], results["Doubling"], "r-")

#Top: Rate
axs[0].set_ylim([-0.00000,0.00009])
axs[0].plot(results["Time"], results["Rate_actual"], "b-")
axs[0].plot(results["Time"], results["Rate_predict"], "r-")
#Sets bounds on OD_predict based on variance
axs[0].plot(results["Time"], results["Rate_predict"] + results["Var_Rate"]**0.5, "r--" )
axs[0].plot(results["Time"], results["Rate_predict"] - results["Var_Rate"]**0.5, "r--")

axs[1].set_ylim([-0.1,0.3])
axs[1].plot(results["Time"], results["Rate"], "k")
axs[1].plot(results["Time"], results["Rate_Pred_Hour"], "r-")

#Bottom: OD Readings
#axs[1].set_ylim([0,0.2])
axs[2].plot(results["Time"], results["OD_actual"], "g-")
axs[2].plot(results["Time"], results["OD_predict"], "r-")

plt.show()