# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:43:06 2023

@author: Michael
"""
import os
import typing
import time
import numpy as np
import pandas as pd
import gc
import torch
from setup import *
pd.options.display.max_columns = 999

### Exact GP Experiments
for d in range(3):
    from setup import *
    a = experiments(d+1,'Yuan')
    mae_exact = torch.tensor([0.]).reshape([1,1])
    training_times_exact = np.array(0)
    if d == 0:
        a.n_training = 500
        a.n_test = 500
    else:
        a.n_training = 4000
        a.n_test = 2000
    for i in range(5):
        a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()
        start = time.time()
        exact_gp_model,l_exact = a.trainExactGP()
        end = time.time()
    
        means_exact,var_exact, mae_exact_tmp = a.testExactGPModel(exact_gp_model,l_exact)
        
        print(str(i+1) + '/' + str(5) +  ' Testing done')
    
        mae_exact = torch.cat([mae_exact, mae_exact_tmp.reshape([1,1])])
        training_times_exact = np.append(training_times_exact, end - start)
    
        if d == 0:
            mae_exact_1d = mae_exact[1:]
            training_times_1d_exact = training_times_exact[1:]
        elif d == 1:
            mae_exact_2d = mae_exact[1:]
            training_times_2d_exact = training_times_exact[1:]
        elif d == 2:
            mae_exact_3d = mae_exact[1:]
            training_times_3d_exact = training_times_exact[1:]
            
training_times_exact = [training_times_1d_exact.mean(),training_times_2d_exact.mean(),training_times_3d_exact.mean()]
mae_exact = pd.DataFrame({'Exact 1d MAE' : mae_exact_1d.detach().numpy().reshape([-1]),
                         'Exact 2d MAE' : mae_exact_2d.detach().numpy().reshape([-1]),
                         'Exact 3d MAE' : mae_exact_3d.detach().numpy().reshape([-1])})

training_times_exact = pd.DataFrame({'1d Exact Training Time' : training_times_1d_exact.mean().reshape([-1]),
                         '2d Exact Training Time' : training_times_2d_exact.mean().reshape([-1]),
                         '3d Exact Training Time' : training_times_3d_exact.mean().reshape([-1])})


mae_exact.to_clipboard(index=False)
training_times_exact.to_clipboard(index=False, header = None)




### NN Experiments

for d in range(3):
    from setup import *
    a = experiments(d+1,'Yuan')
    mae_nn = torch.tensor([0.]).reshape([1,1])
    training_times_nn = np.array(0)
    if d == 0:
        a.n_training = 500
        a.n_test = 500
    else:
        a.n_training = 500
        a.n_test = 2000
    for i in range(5):
        a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()
        start = time.time()
        nn_model = a.trainNeuralNetwork()
        end = time.time()
    
        nn_y_pred, nn_err = a.testNNModel(nn_model)
                
        print(str(i+1) + '/' + str(5) +  ' Testing done')
    
        mae_nn = torch.cat([mae_nn, nn_err.reshape([1,1])])
        training_times_nn = np.append(training_times_nn, end - start)
    
        if d == 0:
            mae_nn_1d = mae_nn[1:]
            training_times_1d_nn = training_times_nn[1:]
        elif d == 1:
            mae_nn_2d = mae_nn[1:]
            training_times_2d_nn = training_times_nn[1:]
        elif d == 2:
            mae_nn_3d = mae_nn[1:]
            training_times_3d_nn = training_times_nn[1:]
            
training_times_nn = [training_times_1d_nn.mean(),training_times_2d_nn.mean(),training_times_3d_nn.mean()]
mae_nn = pd.DataFrame({'NN 1d MAE' : mae_nn_1d.detach().numpy().reshape([-1]),
                         'NN 2d MAE' : mae_nn_2d.detach().numpy().reshape([-1]),
                         'NN 3d MAE' : mae_nn_3d.detach().numpy().reshape([-1])})

training_times_nn = pd.DataFrame({'1d NN Training Time' : training_times_1d_nn.mean().reshape([-1]),
                         '2d NN Training Time' : training_times_2d_nn.mean().reshape([-1]),
                         '3d NN Training Time' : training_times_3d_nn.mean().reshape([-1])})


mae_nn.to_clipboard(index=False)
training_times_nn.to_clipboard(index=False, header = None)

### Approximate GP Experiments

for d in range(3):
    from setup import *
    a = experiments(d+1,'Yuan')
    if d == 0:
        a.n_training = 500
        a.n_inducing = 250
        a.n_test = 500
    else:
        a.n_training = 1000
        a.n_inducing = 500
        a.n_test = 2000

    mae_approx = torch.tensor([0.]).reshape([1,1])
    
    training_times_approx = np.array(0)
    for i in range(5):
        a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()
       
        start_approx = time.time()
        approximate_gp_model, l_approx = a.trainApproximateGP()
        end_approx = time.time()

        print(str(i+1) + '/' + str(5) + ' Training done')
        
        
        means_approx,var_approx, mae_approx_tmp = a.testApproximateGPModel(approximate_gp_model,l_approx)    
   
        print(str(i+1) + '/' + str(5) +  ' Testing done')

        mae_approx = torch.cat([mae_approx, mae_approx_tmp.reshape([1,1])])
        training_times_approx = np.append(training_times_approx, end_approx - start_approx)
        
    if d == 0:
        mae_approx_1d = mae_approx[1:]
        training_times_1d_approx = training_times_approx[1:]
    elif d == 1:
        mae_approx_2d = mae_approx[1:]
        training_times_2d_approx = training_times_approx[1:]
    elif d == 2:
        mae_approx_3d = mae_approx[1:]
        training_times_3d_approx = training_times_approx[1:]
       

training_times_approx = [training_times_1d_approx.mean(),training_times_2d_approx.mean(),training_times_3d_approx.mean()]
mae_approx = pd.DataFrame({'Approx 1d MAE' : mae_approx_1d.detach().numpy().reshape([-1]),
                         'Approx 2d MAE' : mae_approx_2d.detach().numpy().reshape([-1]),
                         'Approx 3d MAE' : mae_approx_3d.detach().numpy().reshape([-1])})

training_times_approx = pd.DataFrame({'1d Approx Training Time' : training_times_1d_approx.mean().reshape([-1]),
                         '2d Approx Training Time' : training_times_2d_approx.mean().reshape([-1]),
                         '3d Approx Training Time' : training_times_3d_approx.mean().reshape([-1])})


mae_approx.to_clipboard(index=False)
training_times_approx.to_clipboard(index=False, header = None)
