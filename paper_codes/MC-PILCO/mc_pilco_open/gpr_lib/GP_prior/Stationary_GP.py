""" 
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""

import torch
from . import GP_prior
import numpy as np


class Stationary_GP(GP_prior.GP_prior):
    """
    Superclass of the stationary GP:
    Define common initializations and provide a function that
    computes the squared distances weighted by the lengthscales
    """


    def __init__(self, active_dims,
                 lengthscales_init=None, flg_train_lengthscales=True, 
                 sigma_n_init=None, flg_train_sigma_n=True,
                 name='', dtype=torch.float64, sigma_n_num=None, device=None):
        """
        Initialize the module and set the lengthscales parameters
        In order to constrain the parameters to be positive we considered
        the log of the lengthscales
        """
        super(Stationary_GP,self).__init__(active_dims,
                                           sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
                                           name=name, dtype=dtype, sigma_n_num=sigma_n_num, device=device)
        #get the number of features
        if active_dims is None:
            raise RuntimeError('Stationary_GP obj require active_dims')
        else:
            self.num_features = active_dims.size
        #check the ARD flag and set the length scalesinitial value
        flg_ARD = True
        if lengthscales_init.size == 1:
                flg_ARD = False
        if lengthscales_init is None:
            lengthscales_init = np.ones(self.num_features)
        self.flg_ARD = flg_ARD
        #get the lengthscale
        self.log_lengthscales_par = torch.nn.Parameter(torch.tensor(np.log( lengthscales_init ), dtype=self.dtype, device=self.device), requires_grad=flg_train_lengthscales)


    def get_weigted_distances(self,X1,X2):
        """
        Computes (X1-X2)^T*sigma^-2*(X1-X2),
        where Sigma = diag(lengthscales)
        """
        if self.flg_ARD:
            lengthscales = torch.exp(self.log_lengthscales_par)
        else:
            lengthscales = torch.exp(self.log_lengthscales_par*torch.ones(self.num_features, dtype=self.dtype, device=self.device))
        #get dimensions and if X1=X2
        N1,D1 = X1.size()
        if X2 is None:
            flg_single_input = True
            N2 = N1
            D2 = D1
        else:
            flg_single_input = False
            N2,D2 = X2.size()
        #slice the inputs and get the weighted distances
        X1_sliced = X1[:,self.active_dims]/lengthscales
        X1_squared = torch.sum(X1_sliced.mul(X1_sliced), dim=1, keepdim=True)
        if flg_single_input:
            dist = X1_squared + X1_squared.transpose(dim0=0, dim1=1) -2*torch.matmul(X1_sliced,X1_sliced.transpose(dim0=0, dim1=1))
        else:
            X2_sliced = X2[:,self.active_dims]/lengthscales
            X2_squared = torch.sum(X2_sliced.mul(X2_sliced), dim=1, keepdim=True)
            dist = X1_squared + X2_squared.transpose(dim0=0, dim1=1) -2*torch.matmul(X1_sliced,X2_sliced.transpose(dim0=0, dim1=1))

        #print(X1_squared)
        #print(X1_squared.transpose(dim0=0, dim1=1))
        #print(X1_squared + X1_squared.transpose(dim0=0, dim1=1))
        #print(2*torch.matmul(X1_sliced,X1_sliced.transpose(dim0=0, dim1=1)))
        #print(dist)

        return dist 



class RBF(Stationary_GP):
    """Implementation of the standard RBF GP with constant mean"""
    

    def __init__(self, active_dims,
                 lengthscales_init=None, flg_train_lengthscales=True,
                 sigma_n_init=None, flg_train_sigma_n=True,
                 lambda_init=None, flg_train_lambda=True,
                 mean_init=None, flg_train_mean = False,
                 name='', dtype=torch.float64, sigma_n_num=None, device=None):
        super(RBF, self).__init__(active_dims,
                                  lengthscales_init=lengthscales_init, flg_train_lengthscales=flg_train_lengthscales,
                                  sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
                                  name=name, dtype=dtype, sigma_n_num=sigma_n_num, device=device)
        #set the scale parameter
        if lambda_init is None:
            lambda_init = np.ones(1)
        if lambda_init.size != 1:
            raise RuntimeError("Lambda must be a np array qith dimension 1")
        self.log_lambda_par = torch.nn.Parameter(torch.tensor(np.log(lambda_init), dtype=self.dtype, device=self.device), requires_grad=flg_train_lambda)
        #set the mean parameters
        if mean_init is None:
            mean_init = np.zeros(1)
        self.mean_par = torch.nn.Parameter(torch.tensor(mean_init, dtype=self.dtype, device=self.device), requires_grad=flg_train_mean)


    def get_mean(self, X):
        """Return constant mean"""
        N = X.size()[0]
        return self.mean_par.repeat(N,1)


    def get_covariance(self, X1, X2=None, flg_noise=False):
        """Compute the exponential of the negative squared weighted distance"""
        if flg_noise & self.GP_with_noise:
            N = X1.size()[0]
            return torch.exp(self.log_lambda_par)*torch.exp(-self.get_weigted_distances(X1, X2)) + self.get_sigma_n_2()*torch.eye(N, dtype=self.dtype, device=self.device)
        else:
            return torch.exp(self.log_lambda_par)*torch.exp(-self.get_weigted_distances(X1, X2))


    def get_diag_covariance(self, X, flg_noise=False):
        """Returns the vector containing the element along the diagonal of the covariance matrix"""
        N = X.size()[0]
        if flg_noise:
            return torch.exp(self.log_lambda_par)*torch.ones(N, dtype=self.dtype, device=self.device) + self.get_sigma_n_2()
        else:
            return torch.exp(self.log_lambda_par)*torch.ones(N, dtype=self.dtype, device=self.device)                
