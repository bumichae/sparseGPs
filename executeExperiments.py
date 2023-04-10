# -*- coding: utf-8 -*-
"""
Change the n_training and n_inducing variables in lines 18-24 to 
obtain the different results from the report
"""

import time
import numpy as np
import pandas as pd
import torch
from setup import *
pd.options.display.max_columns = 999


for d in range(3):
    from setup import *
    a = experiments(d+1,'Liu')
    if d == 0:
        a.n_training = 500
        a.n_inducing = 100
        a.n_test = 500
    else:
        a.n_training = 1000
        a.n_inducing = 100
        a.n_test = 2000

    mse_exact = torch.tensor([0.]).reshape([1,1])
    mse_approx = torch.tensor([0.]).reshape([1,1])
    mse_nn = torch.tensor([0.]).reshape([1,1])
    
    training_times_exact = np.array(0)
    training_times_approx = np.array(0)
    training_times_nn = np.array(0)
    for i in range(5):
        a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()
        start_exact = time.time()
        exact_gp_model,l_exact = a.trainExactGP()
        end_exact = time.time()
        
        start_approx = time.time()
        approximate_gp_model, l_approx = a.trainApproximateGP()
        end_approx = time.time()
        
        
        start_nn = time.time()
        nn_model = a.trainNeuralNetwork()
        end_nn = time.time()
        
        print(str(i+1) + '/' + str(5) + ' Training done')
        
        
        means_exact,var_exact, mse_exact_tmp = a.testExactGPModel(exact_gp_model,l_exact)
        means_approx,var_approx, mse_approx_tmp = a.testApproximateGPModel(approximate_gp_model,l_approx)    
        nn_y_pred, nn_err = a.testNNModel(nn_model)
   
        print(str(i+1) + '/' + str(5) +  ' Testing done')
    
        mse_nn = torch.cat([mse_nn, nn_err.reshape([1,1])])
        mse_exact = torch.cat([mse_exact, mse_exact_tmp.reshape([1,1])])
        mse_approx = torch.cat([mse_approx, mse_approx_tmp.reshape([1,1])])
        training_times_exact = np.append(training_times_exact, end_exact - start_exact)
        training_times_approx = np.append(training_times_approx, end_approx - start_approx)
        training_times_nn = np.append(training_times_nn, end_nn - start_nn)
        
    if d == 0:
        mse_exact_1d = mse_exact[1:]
        mse_approx_1d = mse_approx[1:]
        training_times_1d_exact = training_times_exact[1:]
        training_times_1d_approx = training_times_approx[1:]
        mse_nn_1d = mse_nn[1:]
        training_times_nn_1d = training_times_nn[1:]
    elif d == 1:
        mse_exact_2d = mse_exact[1:]
        mse_approx_2d = mse_approx[1:]
        training_times_2d_exact = training_times_exact[1:]
        training_times_2d_approx = training_times_approx[1:]
        mse_nn_2d = mse_nn[1:]
        training_times_nn_2d = training_times_nn[1:]
    elif d == 2:
        mse_exact_3d = mse_exact[1:]
        mse_approx_3d = mse_approx[1:]
        training_times_3d_exact = training_times_exact[1:]
        training_times_3d_approx = training_times_approx[1:]
        mse_nn_3d = mse_nn[1:]
        training_times_nn_3d = training_times_nn[1:]

mse = pd.DataFrame({'Exact 1d MAE' : mse_exact_1d.detach().numpy().reshape([-1]),
                         'Exact 2d MAE' : mse_exact_2d.detach().numpy().reshape([-1]),
                         'Exact 3d MAE' : mse_exact_3d.detach().numpy().reshape([-1]),
                         'Approximate 1d MAE' : mse_approx_1d.detach().numpy().reshape([-1]),
                         'Approximate 2d MAE' : mse_approx_2d.detach().numpy().reshape([-1]),
                         'Approximate 3d MAE' : mse_approx_3d.detach().numpy().reshape([-1]),
                         'MAE NN 1d' : mse_nn_1d.detach().numpy().reshape([-1]),
                         'MAE NN 2d' : mse_nn_2d.detach().numpy().reshape([-1]),
                         'MAE NN 3d' : mse_nn_3d.detach().numpy().reshape([-1])})

training_times_exact = [training_times_1d_exact.mean(),training_times_2d_exact.mean(),training_times_3d_exact.mean()]
training_times_approx = [training_times_1d_approx.mean(),training_times_2d_approx.mean(),training_times_3d_approx.mean()]

training_times = pd.DataFrame({'1d Exact Training Time' : training_times_1d_exact.mean().reshape([-1]),
                         '2d Exact Training Time' : training_times_2d_exact.mean().reshape([-1]),
                         '3d Exact Training Time' : training_times_3d_exact.mean().reshape([-1]),
                         '1d Approximate Training Time' : training_times_1d_approx.mean().reshape([-1]),
                         '2d Approximate Training Time' : training_times_2d_approx.mean().reshape([-1]),
                         '3d Approximate Training Time' : training_times_3d_approx.mean().reshape([-1]),
                         '1d NN Training Time' : training_times_nn_1d.mean().reshape([-1]),
                         '2d NN Training Time' : training_times_nn_2d.mean().reshape([-1]),
                         '3d NN Training Time' : training_times_nn_3d.mean().reshape([-1])})


mse.to_clipboard(index=False)
training_times.to_clipboard(index=False, header = None)
