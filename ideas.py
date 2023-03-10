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
import torch
from doExperiments import *
pd.options.display.max_columns = 999


a = experiments(2,'Liu')

a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()

a.n_training = int(a.gridPoints.shape[0]/10)
a.n_test = 2000
a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()


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

### plot model output on whole grid

exact_gp_model.eval()
gridDistExact = exact_gp_model(a.gridPoints)
gridDistApprox = approximate_gp_model(a.gridPoints)

a.plot1d(a.gridPoints, gridDistExact.mean.detach(),
         gridDistExact.variance.detach())
a.plot1d(a.gridPoints, gridDistApprox.mean.detach(),
         gridDistApprox.variance.detach())

a.plot2d(a.gridPoints, gridDistExact.mean.detach(),
         gridDistExact.variance.detach())
a.plot2d(a.gridPoints, gridDistApprox.mean.detach(),
         gridDistApprox.variance.detach())

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


for d in range(3):
    from doExperiments import *
    a = experiments(d+1,'Liu')
    if d == 0:
        a.n_training = 500
        a.n_inducing = 250
        a.n_test = 500
    else:
        a.n_training = int(a.gridPoints.shape[0]/2)
        a.n_inducing = 500
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
        exact_gp_model,l_exact = a.trainExactGP(earlyStopping=True)
        end_exact = time.time()
        
        start_approx = time.time()
        approximate_gp_model, l_approx = a.trainApproximateGP()
        end_approx = time.time()
        
        
        start_nn = time.time()
        nn_model = a.trainNeuralNetwork(earlyStopping=True)
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

training_times.to_clipboard(index=False, header = False)




a.plot1d(a.x_test, nn_y_pred)

a.plot2d(a.x_test, nn_y_pred)

a.plot3d(a.x_test, nn_y_pred)

pd.DataFrame(mse_nn_1d.detach().numpy()).to_clipboard(index=False)


from doExperiments import *
a = experiments(2,'Liu')
a.n_training = 1000
a.n_test = 20000
a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()

nn_model = a.trainNeuralNetwork()

nn_y_pred, err = a.testNNModel(nn_model)
# x_test = torch.linspace(-2.5,2.5,100).reshape([100,1])
# nn_y_pred = nn_model(x_test.float())

x_test = torch.linspace(0,1,100).reshape([100,1])
nn_y_pred = nn_model(a.x_test.float())
#nn_y_pred = nn_y_pred.detach().numpy()


a.plot1d(a.x_test, nn_y_pred)

a.plot2d(a.x_test, nn_y_pred)

a.plot3d(a.x_test, means_exact)


fig = plt.figure(figsize=(16,9))

ax0 = fig.add_subplot(1, 2, 1)        
ax0.set_title('NN Test Predictions', size=7,pad=3.0)
ax0.plot(x_test,nn_y_pred)
ax0.set_xlabel(r'x', size=7)
ax0.set_ylabel(r'y', size=7)
ax0.set_xlim((xmin,xmax))
ax0.set_ylim((nn_y_pred.min()-0.5,nn_y_pred.max()+0.5))
plt.show()


fig = plt.figure(figsize=(16,9))

ax0 = fig.add_subplot(1, 2, 1)        
ax0.set_title('NN Test Predictions', size=7,pad=3.0)
ax0.plot(x_test,nn_y_pred)
ax0.set_xlabel(r'x', size=7)
ax0.set_ylabel(r'y', size=7)
ax0.set_xlim((0,1))
ax0.set_ylim((-2,2))
plt.show()


    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:40:27 2023

@author: Michael
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from doExperiments import *

a = experiments(3,'Liu')

a.n_training = 4000
a.n_test = 125000
a.x_train, a.y_train, a.x_test, a.y_test = a.prepareTrainingData()

exact_gp_model,l_exact = a.trainExactGP()

approximate_gp_model, l_approx = a.trainApproximateGP()

means_exact,var_exact, err_exact = a.testExactGPModel(exact_gp_model,l_exact)
means_approx,var_approx, err_approx = a.testApproximateGPModel(approximate_gp_model,l_approx)

### 3d plotting general
fig = plt.figure(figsize=(16, 9))
ax0 = fig.add_subplot(121, projection='3d')
ax0.set_title('Posterior mean', size=7,pad=3.0)
x = a.x_train[:,0].numpy()
y = a.x_train[:,1].numpy()
z = a.x_train[:,2].numpy()
c = a.y_train.numpy()

im0 = ax0.scatter(x, y, z, c=c, cmap=cm.RdBu_r, s=0.1)
ax0.set_xlabel(r'x', size=7)
ax0.set_ylabel(r'y', size=7)
ax0.set_zlabel(r'z', size=7)

plt.colorbar(im0, ax=ax0, shrink=0.3)

ax1 = fig.add_subplot(122, projection='3d')
ax1.set_title('Posterior variance', size=7,pad=3.0)

x = a.x_test[:,0].numpy()
y = a.x_test[:,1].numpy()
z = a.x_test[:,2].numpy()
c = var_exact.numpy()

im1 = ax1.scatter(x, y, z, c=c, cmap=plt.hot(), s=0.1)
ax1.set_xlabel(r'x', size=7)
ax1.set_ylabel(r'y', size=7)
ax1.set_zlabel(r'z', size=7)
plt.colorbar(im1, ax=ax1, shrink=0.3)
plt.show()



### 2d plotting general

fig = plt.figure(figsize=(16, 9))

ax0 = fig.add_subplot(1, 2, 1)        
ax0.set_title('Posterior mean', size=7,pad=3.0)

x = a.x_test[:,0].numpy()
y = a.x_test[:,1].numpy()
c = means_exact.numpy()

im0 = ax0.scatter(x,y,c)
plt.colorbar(im0, ax=ax0, shrink=0.3)

ax1 = fig.add_subplot(1, 2, 2)        
ax1.set_title('Posterior variance', size=7,pad=3.0)

x = a.x_test[:,0].numpy()
y = a.x_test[:,1].numpy()
c = var_exact.numpy()

im1 = ax1.scatter(x,y,c)
plt.colorbar(im1, ax=ax1, shrink=0.3)
plt.show()

