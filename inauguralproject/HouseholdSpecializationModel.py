
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    
    def calc_home_production(self,HM,HF):
        """ calculate home production (piecewise function) """

        par = self.par

        if par.sigma == 0:
            return np.fmin(HM,HF)
        elif par.sigma == 1:
            return HM**(1-par.alpha)*HF**(par.alpha)
        else:
            return ((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        # H = HM**(1-par.alpha)*HF**par.alpha (original variable H commented out)
        H = self.calc_home_production(HM,HF)

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_continuous(self,do_print=False):
        """ solve model continuously """

        par = self.par
        opt = SimpleNamespace()

        # a. objective function (to minimize)
        def obj(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF)
        
        # b. constraints (violated if negative) and bounds
        constraints = (
            {'type': 'ineq', 'fun': lambda x : 24 - (x[0] + x[1])},
            {'type': 'ineq', 'fun': lambda x : 24 - (x[2] + x[3])},    
        )
        bounds = (
            (0, None), # LM >= 0
            (0, None), # HM >= 0
            (0, None), # LF >= 0
            (0, None) # HF >= 0
        )

        # c. call solver
        x0 = np.array([12,12,12,12]) # initial guess
        result = optimize.minimize(
            obj, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        # d. unpack solution
        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
        

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        
        par = self.par
        sol = self.sol

        for i, wF in enumerate(par.wF_vec):
            # Set the value of wF in the parameter namespace
            par.wF = wF 
            
            # Call the continuous solver
            opt = self.solve_continuous()
            
            # Unpack the solution into the appropriate vectors
            sol.LM_vec[i] = opt.LM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF
            sol.HM_vec[i] = opt.HM

        return sol

    def run_regression(self,alpha=None,sigma=None):
        """ run regression """

        par = self.par
        sol = self.sol

        sol.x = np.log(par.wF_vec)
        sol.y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(sol.x.size),sol.x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,sol.y,rcond=None)[0]

        return sol
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        
        par = self.par
        sol = self.sol

        beta0_target = 0.4
        beta1_target = -0.1

        # a. loop over values of alpha and sigma
        N = 20
        alphas = np.linspace(0.1, 0.9, N)
        sigmas = np.linspace(0.1, 2.0, N)
        min_sum_squares = np.inf

        for alpha in alphas:
            for sigma in sigmas:
                par.alpha = alpha
                par.sigma = sigma

                # i. solve model
                self.solve_wF_vec()
                self.run_regression()

                # ii. calculate sum of squares
                sum_squares = (beta0_target-sol.beta0)**2 + (beta1_target-sol.beta1)**2

                # iii. update minimum sum of squares and save parameters
                if sum_squares < min_sum_squares:
                    min_sum_squares = sum_squares
                    best_alpha = alpha
                    best_sigma = sigma

        # b. set parameters to best values and solve model
        par.alpha = best_alpha
        par.sigma = best_sigma
        self.solve_wF_vec()
        
        return (par.alpha, par.sigma, sol.beta0, sol.beta1)

    def estimate_sigma(self,sigma=None):
        """ estimate sigma"""
        
        par = self.par
        sol = self.sol

        beta0_target = 0.4
        beta1_target = -0.1

        par.alpha = 0.5 # set alpha back to 0.5

        # a. loop over values of sigma
        N = 100
        sigmas = np.linspace(0.1, 2.0, N)
        min_sum_squares = np.inf

        for sigma in sigmas:
            par.sigma = sigma

            # i. solve model
            self.solve_wF_vec()
            self.run_regression()

            # ii. calculate sum of squares
            sum_squares = (beta0_target-sol.beta0)**2 + (beta1_target-sol.beta1)**2

            # iii. update minimum sum of squares and save parameters
            if sum_squares < min_sum_squares:
                min_sum_squares = sum_squares
                best_sigma = sigma

        # b. set parameters to best values and solve model
        par.sigma = best_sigma
        self.solve_wF_vec()
        
        return (par.alpha, par.sigma, sol.beta0, sol.beta1)
