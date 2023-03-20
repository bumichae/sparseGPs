# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 08:33:44 2022

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


a = experiments(2,'Liu')

a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()

a.n_training = int(a.gridPoints.shape[0]/10)
a.n_test = 2000
a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()
nn_model = a.trainNeuralNetwork()
a.testNNModel(nn_model)

nn_model = a.trainNeuralNetwork(True)

exact_gp_model,l_exact = a.trainExactGP(True)
approximate_gp_model, l_approx = a.trainApproximateGP()
approximate_gp_model_2, l_approx_2 = a.trainApproximateGP_noMB()

nn_y_pred, nn_err = a.testNNModel(nn_model)
means_exact,var_exact, err_exact = a.testExactGPModel(exact_gp_model,l_exact)
means_approx,var_approx, err_approx = a.testApproximateGPModel(approximate_gp_model,l_approx)
means_approx_2,var_approx_2, err_approx_2 = a.testApproximateGPModel(approximate_gp_model_2,l_approx_2)

 
a.plot1d(a.x_test, means_exact,var_exact)
a.plot1d(a.x_test, means_approx,var_approx)

a.plot2d(a.x_test, means_exact,var_exact)
a.plot2d(a.x_test, means_approx,var_approx)

a.plot3d(a.x_test, means_exact,var_exact)
a.plot3d(a.x_test, means_approx,var_approx)

### train and plot model output on whole grid

a = experiments(2,'Liu')
a.n_training = 2000
a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()

exact_gp_model,l_exact = a.trainExactGP()
gc.collect()
approximate_gp_model, l_approx = a.trainApproximateGP()
gc.collect()
nn_model = a.trainNeuralNetwork()
gc.collect()

exact_gp_model.eval()
gridDistExact = exact_gp_model(a.gridPoints)
gridDistApprox = approximate_gp_model(a.gridPoints)
nnGrid = nn_model(a.gridPoints.float())

a.plot1d(a.gridPoints, gridDistExact.mean.detach(),
         gridDistExact.variance.detach())
a.plot1d(a.gridPoints, gridDistApprox.mean.detach(),
         gridDistApprox.variance.detach())

a.plot2d(a.gridPoints, gridDistExact.mean.detach(),
         gridDistExact.variance.detach())
a.plot2d(a.gridPoints, gridDistApprox.mean.detach(),
         gridDistApprox.variance.detach())
a.plot2d(a.gridPoints, nnGrid.detach())


a.plot3d(a.gridPoints, gridDistExact.mean.detach(),
         gridDistExact.variance.detach())
a.plot3d(a.gridPoints, gridDistApprox.mean.detach(),
         gridDistApprox.variance.detach())



err = torch.tensor([0.])
for d in range(5):
    x, y = a.sampleDataPoint(100)
    m,l = a.naiveUpdateExactModel(x,y)
    means_exact,var_exact, err_exact = a.testExactGPModel(m,l)
    err = torch.cat([err, err_exact.reshape(1)])

err = torch.tensor([0.])
m = exact_gp_model
for d in range(5):
    x, y = a.sampleDataPoint(100)
    m = a.updateModel(m,x,y)
    means_exact,var_exact, err_exact = a.testExactGPModel(m,m.likelihood)
    err = torch.cat([err, err_exact.reshape(1)])



#x = a.gridPoints[torch.argmax(var_approx)]
#y = a.sampleToyField(x)

x, y = a.sampleDataPoint(5)
new_exact_model = a.updateModel(exact_gp_model, x,y)
new_approx_model = a.updateModel(approximate_gp_model, x, y)
### the new x and y points go into the inducing points
means_exact_update,var_exact_update, err_exact_update = a.testExactGPModel(new_exact_model,new_exact_model.likelihood)
means_approx_update,var_approx_update, err_approx_update = a.testApproximateGPModel(new_approx_model,new_approx_model.likelihood)


x, y = a.sampleDataPoint(50)
new_approx_model = a.updateModel(new_approx_model, x, y)
means_approx_update,var_approx_update, err_approx_update = a.testApproximateGPModel(new_approx_model,new_approx_model.likelihood)


new_approx_model = a.updateModel(approximate_gp_model, x.reshape(1,2), y.reshape(1))
means_approx_update,var_approx_update, err_approx_update = a.testApproximateGPModel(new_approx_model,new_approx_model.likelihood)
