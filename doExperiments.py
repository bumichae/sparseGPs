# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:59:04 2022

@author: Michael
"""
import os
import typing
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import math
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import tqdm
from models import * 
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class experiments:
    
    def __init__(self, dim = 1, field = 'Liu', gridSize = 2000):
        
        torch.manual_seed(0)
        self.dim = dim
        self.n_test = gridSize
        self.n_training = gridSize
        self.n_inducing = 1000
        self.x_train = None
        self.y_train = None
        self.field = field
        self.x_test = None
        self.y_test = None
        self.fMin = None
        self.fMax = None
        self.fVarMin = None
        self.fVarMax = None
        self.gridPoints = None
    
    def toyField(self, x):
        
        if self.field == 'Yuan':
            m = 2*(torch.exp(-30*(torch.prod(x.T,axis=0)-0.25)**2) + torch.sin(3.14159*(torch.prod(x.T,axis=0))**2)) -2 
            r = torch.exp(torch.sin(2*math.pi*torch.prod(x.T,axis=0)))
            return m, r
        
        if self.field == 'Liu':
            m = torch.sinc(torch.prod(x.T,axis=0))
            r = 0.05 + 0.2*(1 + torch.sin(2*torch.prod(x.T,axis=0)))/(1+torch.exp(-0.2*torch.prod(x.T,axis=0)))
            return m,r
        
    def sampleToyField(self, x):

        m,r = self.toyField(x)    
        y = torch.normal(m.to(torch.float64), torch.sqrt(r).to(torch.float64))

        return y
    
    def plotAnalyticalField(self):
        
        # Domain limits
        
        if self.field == 'Liu':

            # x-axis
            xmin = -2.5
            xmax = 2.5
            
            # y-axis
            ymin = -2.5
            ymax = 2.5
            
            # z-axis
            zmin = -2.5
            zmax = 2.5
        
        else:
                        
            xmin = 0
            xmax = 1
            
            # y-axis
            ymin = 0
            ymax = 1
            
            # z-axis
            zmin = 0
            zmax = 1
        
        
        # Define the grid
        if self.dim == 3:
        
            nx = int(50)
            ny = int(50)
            nz = int(50)
            
            x0 = torch.unsqueeze(torch.linspace(xmin,xmax,nx),-1)
            x1 = torch.unsqueeze(torch.linspace(ymin,ymax,ny),-1)
            x2 = torch.unsqueeze(torch.linspace(zmin,zmax,nz),-1)
        
            xmax = torch.max(x0)
            xmin = torch.min(x0)
        
            ymax = torch.max(x1)
            ymin = torch.min(x1)
            
            zmax = torch.max(x2)
            zmin = torch.min(x2)
        
            GridX, GridY, GridZ = np.meshgrid(x0, x1, x2)
            GridX = GridX.T
            GridY = GridY.T
            GridZ = GridZ.T
            gridPoints = torch.from_numpy(np.vstack([GridX.ravel(), GridY.ravel(), GridZ.ravel()])).t().to(torch.float64)
        
        elif self.dim == 2:
        
            nx = int(500)
            ny = int(500)
            
            x0 = torch.unsqueeze(torch.linspace(xmin,xmax,nx),-1)
            x1 = torch.unsqueeze(torch.linspace(ymin,ymax,ny),-1)
        
            xmax = torch.max(x0)
            xmin = torch.min(x0)
        
            ymax = torch.max(x1)
            ymin = torch.min(x1)
        
            GridX, GridY = np.meshgrid(x0, x1)
            GridX = GridX.T
            GridY = GridY.T
            gridPoints = torch.from_numpy(np.vstack([GridX.ravel(), GridY.ravel()])).t().to(torch.float64)
        
        else:
            
            nx = int(1000)
            
            x0 = torch.unsqueeze(torch.linspace(xmin,xmax,nx),-1).to(torch.float64)
            
            xmax = torch.max(x0)
            xmin = torch.min(x0)
            
            gridPoints = x0.detach()
        
        mGrid,rGrid = self.toyField(gridPoints)

        fMin = torch.min(mGrid)
        fMax = torch.max(mGrid)
        
        fVarMin = torch.min(rGrid)
        fVarMax = torch.max(rGrid)
        
        fig0 = plt.figure(figsize=(16,9))
        
        if self.dim == 3:
            extent = (xmin,xmax,ymin,ymax,zmin,zmax)
            
            fMean = mGrid.detach().numpy().reshape(nx,ny,nz)
            fVar = rGrid.detach().numpy().reshape(nx,ny,nz)
        
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(121, projection='3d')
            ax.set_title('Posterior mean', size=7,pad=3.0)
        
            # Display the mean field
            pcm = ax.scatter(gridPoints[:,0], gridPoints[:,1], gridPoints[:,2],
                             c=fMean.T,s=0.1,vmin=fMin,vmax=fMax,
                             cmap=cm.RdBu_r)
            plt.colorbar(pcm, ax=ax, shrink=0.3)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            ax.set_zlim(zmin,zmax)
            
            # display the variance field
            ax = fig.add_subplot(122, projection='3d')
            ax.set_title('Posterior variance', size=7,pad=3.0)
            pcm = ax.scatter(gridPoints[:,0], gridPoints[:,1], gridPoints[:,2],
                             c=fVar.T,s=0.1,vmin=fVarMin,vmax=fVarMax,
                             cmap=cm.RdBu_r)
            plt.colorbar(pcm, ax=ax, shrink=0.3)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            ax.set_zlim(zmin,zmax)
            
            plt.show()
        
        elif self.dim == 2:
            
            extent = (xmin,xmax,ymin,ymax)
            
            fMean = mGrid.detach().numpy().reshape(ny,nx)
            fVar = rGrid.detach().numpy().reshape(ny,nx)
        
            ax0 = fig0.add_subplot(1, 2, 1)        
            ax0.set_title('Posterior mean', size=7,pad=3.0)
            im0 = ax0.imshow(fMean.T,  extent=extent,cmap=cm.RdBu_r,vmin=fMin,vmax = fMax)
            plt.colorbar(im0, ax=ax0, shrink=0.3)
            ax0.set_xlabel(r'x', size=7)
            ax0.set_ylabel(r'y', size=7)
            ax0.set_xlim((xmin,xmax))
            ax0.set_ylim((ymin,ymax))
        
            ax1 = fig0.add_subplot(1, 2, 2)
            ax1.set_title('Posterior variance', size=7,pad=3.0)
            im1 = ax1.imshow(fVar.T, extent=extent, cmap=cm.RdBu_r,vmin=fVarMin,vmax = fVarMax)
            plt.colorbar(im1, ax=ax1, shrink=0.3)              
            ax1.set_xlabel(r'x', size=7)
            ax1.set_ylabel(r'y', size=7)
            ax1.set_xlim((xmin,xmax))
            ax1.set_ylim((ymin,ymax))
        
        else:
            
            fMean = mGrid.detach().numpy()
            fVar = rGrid.detach().numpy()
            
            ax0 = fig0.add_subplot(1, 2, 1)        
            ax0.set_title('Posterior mean', size=7,pad=3.0)
            ax0.plot(gridPoints,fMean)
            ax0.set_xlabel(r'x', size=7)
            ax0.set_ylabel(r'y', size=7)
            ax0.set_xlim((xmin,xmax))
            ax0.set_ylim((fMin-0.5,fMax+0.5))
        
            ax1 = fig0.add_subplot(1, 2, 2)
            ax1.set_title('Posterior variance', size=7,pad=3.0)
            ax1.plot(gridPoints,fVar)         
            ax1.set_xlabel(r'x', size=7)
            ax1.set_ylabel(r'y', size=7)
            ax1.set_xlim((xmin,xmax))
            ax1.set_ylim((fVarMin-0.5,fVarMax+0.5))
    
    def prepareTrainingData(self):
        # Define the grid
        
        if self.field == 'Liu':
            # x-axis
            xmin = -2.5
            xmax = 2.5
            
            # y-axis
            ymin = -2.5
            ymax = 2.5
            
            # z-axis
            zmin = -2.5
            zmax = 2.5
            
        else:
            
            xmin = 0
            xmax = 1
            
            # y-axis
            ymin = 0
            ymax = 1
            
            # z-axis
            zmin = 0
            zmax = 1
            
        if self.dim == 3:
        
            nx = int(50)
            ny = int(50)
            nz = int(50)
            
            x0 = torch.unsqueeze(torch.linspace(xmin,xmax,nx),-1)
            x1 = torch.unsqueeze(torch.linspace(ymin,ymax,ny),-1)
            x2 = torch.unsqueeze(torch.linspace(zmin,zmax,nz),-1)
        
            xmax = torch.max(x0)
            xmin = torch.min(x0)
        
            ymax = torch.max(x1)
            ymin = torch.min(x1)
            
            zmax = torch.max(x2)
            zmin = torch.min(x2)
        
            GridX, GridY, GridZ = np.meshgrid(x0, x1, x2)
            GridX = GridX.T
            GridY = GridY.T
            GridZ = GridZ.T
            self.gridPoints = torch.from_numpy(np.vstack([GridX.ravel(), GridY.ravel(), GridZ.ravel()])).t().to(torch.float64)
            # x_test = self.gridPoints[torch.randperm(len(self.gridPoints))][:self.n_test]
            # y_test = self.sampleToyField(x_test)
            
        elif self.dim == 2:
        
            nx = int(150)
            ny = int(150)
            
            x0 = torch.unsqueeze(torch.linspace(xmin,xmax,nx),-1)
            x1 = torch.unsqueeze(torch.linspace(ymin,ymax,ny),-1)
        
            xmax = torch.max(x0)
            xmin = torch.min(x0)
        
            ymax = torch.max(x1)
            ymin = torch.min(x1)
        
            GridX, GridY = np.meshgrid(x0, x1)
            GridX = GridX.T
            GridY = GridY.T
            self.gridPoints = torch.from_numpy(np.vstack([GridX.ravel(), GridY.ravel()])).t().to(torch.float64)

        else:
            
            nx = int(1000)
            
            x0 = torch.unsqueeze(torch.linspace(xmin,xmax,nx),-1).to(torch.float64)
            
            xmax = torch.max(x0)
            xmin = torch.min(x0)
            
            self.gridPoints = x0.detach()


        mGrid,rGrid = self.toyField(self.gridPoints)
        
        self.fMin = torch.min(mGrid)
        self.fMax = torch.max(mGrid)

        self.fVarMin = torch.min(rGrid)
        self.fVarMax = torch.max(rGrid)
        
        ind_train, ind_test, rest = random_split(range(len(self.gridPoints)), 
                                                 [self.n_training,self.n_test, 
                                                  len(self.gridPoints)-self.n_training-self.n_test]
                                                 )
        x_train = self.gridPoints[ind_train]
        y_train = self.sampleToyField(x_train)
        x_test = self.gridPoints[ind_test]
        y_test = self.sampleToyField(x_test) 
        return x_train, y_train, x_test, y_test
    
    def trainNeuralNetwork(self):
        
        if self.field == 'Liu':
        
            in1 = self.dim
            out1 = 500
            in2 = out1
            out2 = 1000
            in3 = out2
            out3 = 1
            
            model = nn.Sequential(nn.Linear(in1, out1),
                          nn.Tanh(),
                          nn.Linear(in2,out2),
                          nn.Tanh(),
                          nn.Linear(in3,out3),
                          nn.Tanh())
            
           
    
            loss_function = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            losses = []
            start = time.time()
            for epoch in range(1000):
                nn_y_pred = model(self.x_train.float())
                loss = loss_function(nn_y_pred, self.y_train.reshape(nn_y_pred.shape).float())
                losses.append(loss.item())
            
                model.zero_grad()
                loss.backward()
            
                optimizer.step()
                if(epoch % 100 == 0):
                    print("Finished Epoch " + str(epoch) + " out of " + str(1000))
            end = time.time()
            
            print('Training took %.3f seconds' %(end - start))
        
        else:
            
                
            model = DenseNet(self.dim, (1000,500,500,100), 1)
            loss_function = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            losses = []
            start = time.time()
            for epoch in range(1000):
                nn_y_pred = model(self.x_train.float())
                loss = loss_function(nn_y_pred, self.y_train.reshape(nn_y_pred.shape).float())
                losses.append(loss.item())
            
                model.zero_grad()
                loss.backward()
            
                optimizer.step()
                if(epoch % 100 == 0):
                    print("Finished Epoch " + str(epoch) + " out of " + str(1000))
            end = time.time()
            
            print('Training took %.3f seconds' %(end - start))            
        return model

        
    def trainExactGP(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(self.x_train, self.y_train, likelihood).double()
        
        
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        training_iter = 100
        start = time.time()
        loss = [1000]
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(self.x_train)
            
            loss.append(-mll(output, self.y_train))
            
            if (loss[i] - loss[i+1])/loss[i] < 0.0001:
                break
            loss[i+1].backward()
            optimizer.step()
            if not i % 10:
                print('Step ' + str(i) + ' out of ' + str(training_iter))
        end = time.time()
        print('Training took %.0f seconds' % (end-start))
        
        return model, likelihood
                
    def trainApproximateGP(self):
        
        train_dataset = TensorDataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        inducing_points = self.x_train[torch.randperm(len(self.x_train))[:self.n_inducing]] 
        
        model = ApproximateGPModel(inducing_points = inducing_points).double()
        
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam([{'params' : model.parameters(), 'lr' : 0.1},
                                     {'params' : likelihood.parameters(), 'lr' : 0.1}])
        
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, 
                                                    model, 
                                                    num_data=self.y_train.size(0))
        
        num_epochs = 5        
        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        
        start = time.time()
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
        end = time.time()
        
        print('Training took %.3f seconds' % (end-start))

        return model, likelihood
        
    def testExactGPModel(self, model, likelihood):
        
        test_dataset = TensorDataset(self.x_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False) 
        
        model.eval()
        likelihood.eval()
        means = torch.tensor([0.])
        var = torch.tensor([0.])
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds = likelihood(model(x_batch))
                means = torch.cat([means, preds.mean.cpu()])
                var = torch.cat([var, preds.variance.cpu()])
        means = means[1:]
        var = var[1:]
        
        print('Test MAE: {}'.format(torch.mean(torch.abs(means - self.y_test))))
        
        return means, var, torch.mean(torch.abs(means - self.y_test))
    
    def testApproximateGPModel(self, model, likelihood):
        test_dataset = TensorDataset(self.x_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False) 
        model.eval()
        likelihood.eval()
        means = torch.tensor([0.])
        var = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds = likelihood(model(x_batch))
                means = torch.cat([means, preds.mean.cpu()])
                var = torch.cat([var, preds.variance.cpu()])
        means = means[1:]
        var = var[1:]

        
        print('Test MAE: {}'.format(torch.mean(torch.abs(means - self.y_test))))
        
        return means, var, torch.mean(torch.abs(means - self.y_test))

    def testNNModel(self, model):
        if self.dim == 1:
            y_pred = model(self.x_test.float())
        else:
            
            test_dataset = TensorDataset(self.x_test, self.y_test)
            test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False) 
            y_pred = torch.tensor([0.]).reshape([1,1])
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    preds = model(x_batch.float())
                    y_pred = torch.cat([y_pred, preds])
            y_pred = y_pred[1:]
            y_pred = y_pred.reshape_as(self.y_test)
            
        print('Test MAE: {}'.format(torch.mean(torch.abs(y_pred - self.y_test))))
        
        return y_pred, torch.mean(torch.abs(y_pred - self.y_test))
    
    # def plot1d(self,x, y, var = None):
    #     # x-axis
    #     if self.field == 'Liu':
    #         xmin = -2.5
    #         xmax = 2.5
            
    #     else:
    #         xmin = 0
    #         xmax = 1
        
    #     if var != None:
            
    #         fig0 = plt.figure(figsize = (16,9))
            
    #         ax0 = fig0.add_subplot(1, 2, 1)        
    #         ax0.set_title('Posterior mean', size=7,pad=3.0)
    #         ax0.plot(x.detach().numpy(),y.detach().numpy(),'b.')
    #         ax0.set_xlabel(r'x', size=7)
    #         ax0.set_ylabel(r'y', size=7)
    #         ax0.set_xlim((xmin,xmax))
    #         ax0.set_ylim((y.detach().min()-0.5,y.detach().max()+0.5))
        
    #         ax1 = fig0.add_subplot(1, 2, 2)
    #         ax1.set_title('Posterior variance', size=7,pad=3.0)
    #         ax1.plot(x.detach().numpy(),var.detach().numpy(), 'b.')         
    #         ax1.set_xlabel(r'x', size=7)
    #         ax1.set_ylabel(r'y', size=7)
    #         ax1.set_xlim((xmin,xmax))
    #         ax1.set_ylim((var.detach().min()-0.5,var.detach().max()+0.5))
            
            
    #     else:
    #         fig0 = plt.figure(figsize=(16,9))

    #         ax0 = fig0.add_subplot(1, 2, 1)        
    #         ax0.set_title('Prediction', size=7,pad=3.0)
    #         ax0.plot(x.detach().numpy(),y.detach().numpy())
    #         ax0.set_xlabel(r'x', size=7)
    #         ax0.set_ylabel(r'y', size=7)
    #         ax0.set_xlim((xmin,xmax))
    #         ax0.set_ylim((y.detach().min()-0.5,y.detach().max()+0.5))
        
    #     return fig0
    
    # def plot2d(self,x,y, var = None):
        
    #     if self.field == 'Liu':
    #         # x-axis
    #         xmin = -2.5
    #         xmax = 2.5
    #         nx = 150
    #         # y-axis
    #         ymin = -2.5
    #         ymax = 2.5
    #         ny = 150
    #     else:
    #         xmin = 0
    #         xmax = 1
    #         ymin = 0
    #         ymax = 1
    #         nx = 150
    #         ny = 150

    #     if var != None:
            
    #         fig0 = plt.figure(figsize=(16,9))
    
    #         extent = (xmin,xmax,ymin,ymax)
            
    #         ax0 = fig0.add_subplot(1, 2, 1)        
    #         ax0.set_title('Posterior mean', size=7,pad=3.0)
    #         im0 = ax0.imshow(y.detach().numpy().reshape(ny,nx).T,  extent=extent,
    #                           cmap=cm.RdBu_r,vmin=self.fMin,vmax = self.fMax)
    #         plt.colorbar(im0, ax=ax0, shrink=0.3)
    #         ax0.set_xlabel(r'x', size=7)
    #         ax0.set_ylabel(r'y', size=7)
    #         ax0.set_xlim((xmin,xmax))
    #         ax0.set_ylim((ymin,ymax))
            
    #         ax1 = fig0.add_subplot(1, 2, 2)
    #         ax1.set_title('Posterior variance', size=7,pad=3.0)
    #         im1 = ax1.imshow(var.detach().numpy().reshape(ny,nx).T, extent=extent, 
    #                           cmap=cm.RdBu_r,vmin=self.fVarMin,vmax = self.fVarMax)
    #         plt.colorbar(im1, ax=ax1, shrink=0.3)              
    #         ax1.set_xlabel(r'x', size=7)
    #         ax1.set_ylabel(r'y', size=7)
    #         ax1.set_xlim((xmin,xmax))
    #         ax1.set_ylim((ymin,ymax))
            
    #     else:
    #         fig0 = plt.figure(figsize=(16,9))
    
    #         extent = (xmin,xmax,ymin,ymax)
            
    #         ax0 = fig0.add_subplot(1, 2, 1)        
    #         ax0.set_title('Posterior mean', size=7,pad=3.0)
    #         im0 = ax0.imshow(y.detach().numpy().reshape(ny,nx).T,  extent=extent,
    #                           cmap=cm.RdBu_r,vmin=self.fMin,vmax = self.fMax)
    #         plt.colorbar(im0, ax=ax0, shrink=0.3)
    #         ax0.set_xlabel(r'x', size=7)
    #         ax0.set_ylabel(r'y', size=7)
    #         ax0.set_xlim((xmin,xmax))
    #         ax0.set_ylim((ymin,ymax))
            
    #     return fig0

    # def plot3d(self,x,y,var = None):     
        
    #     if self.field == 'Liu':
    #         # x-axis
    #         nx = 50
    #         xmin = -2.5
    #         xmax = 2.5
    #         nx = 50
    #         # y-axis
    #         ny = 50
    #         ymin = -2.5
    #         ymax = 2.5
    #         ny = 50
    #         # z-axis
    #         nz = 50
    #         zmin = -2.5
    #         zmax = 2.5        
    #         nz = 50
    #     else:
    #         nx = 50
    #         ny = 50
    #         nz = 50
    #         xmin = 0
    #         xmax = 1
    #         ymin = 0
    #         ymax = 1
    #         zmin = 0
    #         zmax = 1
    #     fMean = y.detach().numpy().reshape(nx,ny,nz)       
        
    #     if var != None:
    #         fVar = var.detach().numpy().reshape(nx,ny,nz)
    #         fig0 = plt.figure(figsize=(16, 9))
    #         ax = fig0.add_subplot(121, projection='3d')
            
    #         # Display the mean field
    #         pcm = ax.scatter(x[:,0], x[:,1], x[:,2],c=fMean.T,
    #                           s=0.1,vmin=self.fMin,vmax=self.fMax)
    #         plt.colorbar(pcm, ax=ax, shrink=0.3)
    #         ax.set_xlim(xmin,xmax)
    #         ax.set_ylim(ymin,ymax)
    #         ax.set_zlim(zmin,zmax)
            
    #         # display the variance field
    #         ax = fig0.add_subplot(122, projection='3d')
    #         pcm = ax.scatter(x[:,0], x[:,1], x[:,2],c=fVar.T,
    #                           s=0.1,vmin=self.fVarMin,vmax=self.fVarMax)
    #         plt.colorbar(pcm, ax=ax, shrink=0.3)
    #         ax.set_xlim(xmin,xmax)
    #         ax.set_ylim(ymin,ymax)
    #         ax.set_zlim(zmin,zmax)
    #     else:
    #         fig0 = plt.figure(figsize=(16, 9))
    #         ax = fig0.add_subplot(121, projection='3d')
            
    #         # Display the mean field
    #         pcm = ax.scatter(x[:,0], x[:,1], x[:,2],c=fMean.T,
    #                           s=0.1,vmin=self.fMin,vmax=self.fMax)
    #         plt.colorbar(pcm, ax=ax, shrink=0.3)
    #         ax.set_xlim(xmin,xmax)
    #         ax.set_ylim(ymin,ymax)
    #         ax.set_zlim(zmin,zmax)
            
    #     return fig0
    

    def plot1d(self,x, y, var = None):
        # x-axis
        if self.field == 'Liu':
            xmin = -2.5
            xmax = 2.5
            
        else:
            xmin = 0
            xmax = 1
        
        if var != None:
            
            fig0 = plt.figure(figsize = (16,9))
            
            ax0 = fig0.add_subplot(1, 2, 1)        
            ax0.set_title('Posterior mean', size=7,pad=3.0)
            ax0.plot(x.detach().numpy(),y.detach().numpy(),'b.')
            ax0.set_xlabel(r'x', size=7)
            ax0.set_ylabel(r'y', size=7)
            ax0.set_xlim((xmin,xmax))
            ax0.set_ylim((y.detach().min()-0.5,y.detach().max()+0.5))
        
            ax1 = fig0.add_subplot(1, 2, 2)
            ax1.set_title('Posterior variance', size=7,pad=3.0)
            ax1.plot(x.detach().numpy(),var.detach().numpy(), 'b.')         
            ax1.set_xlabel(r'x', size=7)
            ax1.set_ylabel(r'y', size=7)
            ax1.set_xlim((xmin,xmax))
            ax1.set_ylim((var.detach().min()-0.5,var.detach().max()+0.5))
            
            
        else:
            fig0 = plt.figure(figsize=(16,9))

            ax0 = fig0.add_subplot(1, 2, 1)        
            ax0.set_title('Prediction', size=7,pad=3.0)
            ax0.plot(x.detach().numpy(),y.detach().numpy())
            ax0.set_xlabel(r'x', size=7)
            ax0.set_ylabel(r'y', size=7)
            ax0.set_xlim((xmin,xmax))
            ax0.set_ylim((y.detach().min()-0.5,y.detach().max()+0.5))
        
        return fig0
    
    def plot2d(self,x,y, var = None):
        
        if self.field == 'Liu':
            # x-axis
            xmin = -2.5
            xmax = 2.5
            nx = 150
            # y-axis
            ymin = -2.5
            ymax = 2.5
            ny = 150
        else:
            xmin = 0
            xmax = 1
            ymin = 0
            ymax = 1
            nx = 150
            ny = 150

        if var != None:
            
            fig = plt.figure(figsize=(16, 9))

            ax0 = fig.add_subplot(1, 2, 1)        
            ax0.set_title('Posterior mean', size=7,pad=3.0)
            
            x = x[:,0].numpy()
            y = x[:,1].numpy()
            c = y.numpy()
            
            im0 = ax0.scatter(x,y,c)
            plt.colorbar(im0, ax=ax0, shrink=0.3)
            
            ax1 = fig.add_subplot(1, 2, 2)        
            ax1.set_title('Posterior variance', size=7,pad=3.0)
            
            x_plot = x[:,0].numpy()
            y_plot = x[:,1].numpy()
            c = var.numpy()
            
            im1 = ax1.scatter(x_plot,y_plot,c)
            plt.colorbar(im1, ax=ax1, shrink=0.3)
            
        else:
            fig = plt.figure(figsize=(16,9))
    
            ax0 = fig.add_subplot(1, 2, 1)        
            ax0.set_title('Posterior mean', size=7,pad=3.0)
            
            x_plot = x[:,0].numpy()
            y_plot = x[:,1].numpy()
            c = y.numpy()
            
            im0 = ax0.scatter(x_plot,y_plot,c)
            plt.colorbar(im0, ax=ax0, shrink=0.3)
            
        return fig

    def plot3d(self,x,y,var = None):     
        
        if var != None:
            fig = plt.figure(figsize=(16, 9))
            ax0 = fig.add_subplot(121, projection='3d')
            ax0.set_title('Posterior mean', size=7,pad=3.0)
            x_plot = x[:,0].numpy()
            y_plot = x[:,1].numpy()
            z_plot = x[:,2].numpy()
            c = y.numpy()
            
            im0 = ax0.scatter(x_plot, y_plot, z_plot, c=c, cmap=cm.RdBu_r, s=0.1)
            ax0.set_xlabel(r'x', size=7)
            ax0.set_ylabel(r'y', size=7)
            ax0.set_zlabel(r'z', size=7)
            
            plt.colorbar(im0, ax=ax0, shrink=0.3)
            
            ax1 = fig.add_subplot(122, projection='3d')
            ax1.set_title('Posterior variance', size=7,pad=3.0)
            
            c = var.numpy()
            
            im1 = ax1.scatter(x_plot, y_plot, z_plot, c=c, cmap=plt.hot(), s=0.1)
            ax1.set_xlabel(r'x', size=7)
            ax1.set_ylabel(r'y', size=7)
            ax1.set_zlabel(r'z', size=7)
            plt.colorbar(im1, ax=ax1, shrink=0.3)
        else:
            fig = plt.figure(figsize=(16, 9))
            ax0 = fig.add_subplot(121, projection='3d')
            ax0.set_title('Posterior mean', size=7,pad=3.0)
            x_plot = x[:,0].numpy()
            y_plot = x[:,1].numpy()
            z_plot = x[:,2].numpy()
            c = y.numpy()
            
            im0 = ax0.scatter(x_plot, y_plot, z_plot, c=c, cmap=cm.RdBu_r, s=0.1)
            ax0.set_xlabel(r'x', size=7)
            ax0.set_ylabel(r'y', size=7)
            ax0.set_zlabel(r'z', size=7)
            
            plt.colorbar(im0, ax=ax0, shrink=0.3)
            
        return fig
        

    def sampleDataPoint(self,num_points = 1):
        unif = torch.ones(self.gridPoints.shape[0])
        idx = unif.multinomial(num_points, replacement = False)
        new_x = self.gridPoints[idx]
        new_y = self.sampleToyField(new_x)
        return new_x, new_y
    
    def sampleHighVariancePoint(self,model,var):
        
        new_x = self.x_test[torch.argmax(var)]
        new_y = self.y_test[torch.argmax(var)]
        return new_x, new_y
        
    
    def updateModel(self, model, x, y):
        fantasy_model = model.get_fantasy_model(x,y)
        return fantasy_model
    
    def naiveUpdateExactModel(self, x,y):
        
        self.x_train = torch.cat([self.x_train, x])
        self.y_train = torch.cat([self.y_train ,y])
        
        m, l = self.trainExactGP()
        
        return m,l
    
    def naiveUpdateApproximateModel(self, x,y):
        
        self.x_train = torch.cat([self.x_train, x])
        self.y_train = torch.cat([self.y_train ,y])
        
        m, l = self.trainApproximateGP()
        
        return m,l
        
        