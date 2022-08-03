#!/bin/env python3
'''
This is a more compact implementation of the Extended Kalmann Filter
Similar to model here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181923
'''
import math
import numpy as np

######################
#     Parameters     #
######################

I_0 = 10
Q = np.asarray([[0.002**2 , 0],
                [0, 5.0e-7]]) 

R = np.asarray([[40**2]])

def custom_mat_mul(*args):
    assert len(args) >= 2

    cur = args[0]
    for nxt in args[1:]:
        cur = np.matmul(cur,nxt)
    
    return cur

def f_func(x,dt):
    """
    Assumption:
    1) We have a geometric model of how the cells will grow.
       OD_n+1 = OD_n * (exp(rate_n*dt_n)) 
    2) Rate remains constant
       rate_n+1 = rate_n
    """
    
    x_pred = [x[0]*math.exp(x[1]*dt), x[1]]
    return np.asarray(x_pred)

def h_func(x):
    """
    Technically, the only variable we are measuring is the current of diode
    This is proportional to the negative log of OD
    I = I_0 * 10 ^ (-OD)
    TODO
    Figure out what is I_0
    """

    return np.asarray([I_0 * 10.0**(-x[0])])

def get_F(x,dt):
    """
    Returns the Jacobian of the Process Function
    """
    F = [[math.exp(x[1]*dt), x[0]*dt*math.exp(x[1]*dt)],
         [0, 1]]

    return np.asarray(F)
def get_H(x):
    """
    Returns the Jacaobian of the Measurement Function
    """

    H = [[- I_0 * math.log(10.0) * (10.0**(-x[0])),0]]

    return np.asarray(H)

def ekf(x,x_cur,dt,P, verbose = False):
    """
    x -> Previous State vector, [OD, rate]
    x_pred -> Current Predicted State Vector
    x_cur -> Real Current State Vector
    P -> Previous Variance of State Vector
    w -> Process Noise Vector
    z -> Observation Vector
    v -> Measurement Noise Vector
    F -> Process Function
    H -> Measurement Function
    Q -> Variance of System
    R -> Variance of Measurement
    """

    #Predict From existing Info
    x_pred = f_func(x,dt)
    F = get_F(x,dt)
    P_pred = custom_mat_mul(F,P,F.transpose()) + Q 

    #Update with new values

    H = get_H(x_pred)
    INVERSE = custom_mat_mul(H,P_pred,H.transpose()) + R
    INVERSE = np.linalg.inv(INVERSE)
    K = custom_mat_mul(P_pred,H.transpose(),INVERSE)

    z = h_func(x_cur)
    innovation = z - h_func(x_pred)
    x_tmp = x_pred  + np.matmul(K,innovation)


    KH = np.matmul(K,H)
    I = np.eye(KH.shape[0])
    DIFF = I - KH
    P_tmp = custom_mat_mul(DIFF,P_pred,DIFF.transpose()) + custom_mat_mul(K,R,K.transpose())
    if (verbose): 
        print("==========")
        print("x: ", x)
        print("P: ", P)
        print("x_pred: ", x_pred)
        print("P_pred: ", P_pred)
        print("F: ",F)
        print("H: ",H)
        print("K: ", K)
        print("Net Innovation: ", np.matmul(K,innovation))
        print("x_tmp: ",x_tmp)
        print("P_tmp: ",P_tmp)
        
    return x_tmp, P_tmp
