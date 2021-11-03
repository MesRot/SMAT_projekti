import numpy as np
from scipy.stats import exponnorm
from scipy.optimize import curve_fit, minimize
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd


def omaEMG(x, myy, sigma, lamda):
    #myy = huippu
    #sigma = leveys(?)
    #lamda = hännän pituus(?)
    #palauttaa funktion arvon pisteessä x, VAIN YKSI ARVO!!!
    result=(lamda/2) * np.exp((lamda/2) * (2*myy + lamda*sigma*sigma - 2*x)) * scipy.special.erfc((myy + lamda*sigma*sigma-x) / (np.sqrt(2)*sigma))
    return result

def kokoEMG(x, myy, sigma, lamda):
    #palauttaa jokaista x arvoa kohden yhden y arvon
    y_values = np.array([])
    for x_ in x:
        #print(x_)
        y = omaEMG(x_, myy, sigma, lamda)
        y_values = np.append([y_values], [y])
    return y_values

def EMG(x, mu, sigma, lambda_, h):
    #TODO : muuta rajoitus pois

    # Change the factor of x to modify the position of the graph.
    # Also changes the power of parameters ie. x = 0.5x means the same
    # change in for example lambda is stronger effect when x is not modified.
    #x = x
    
    if sigma <= 1e-3:
        sigma = 1e-3

    def special_error_function(x):
        temp = scipy.special.erfc(x)
        if temp < 1e-5:
            return 0
        else:
            return temp

    def formula(x, mu, sigma):
        result=(lambda_/2) * np.exp((lambda_/2) * (2*mu + lambda_*sigma*sigma - 2*x)) * scipy.special.erfc((mu + lambda_*sigma*sigma-x) / (np.sqrt(2)*sigma))
        return result

    def formula1(x, mu, sigma):
        tau = 1 / lambda_

        if tau <= 1e-3:
            tau = 1e-3

        z = (1 / np.sqrt(2)) * ((sigma / tau) - ((x - mu) / sigma))

        if z < 0:
            val = ((sigma) / tau) * np.sqrt(np.pi / 2) * np.exp(0.5 * (sigma / tau)**2 - ((x - mu) / tau)) * scipy.special.erfc(z)
        elif z < 6.71*10**7:
            val = np.exp(-0.5 * ((x - mu) / sigma)**2) * (sigma/tau) *  np.sqrt(np.pi / 2) * scipy.special.erfcx(z)
        else:
            val = np.exp(-0.5 * ((x - mu) / sigma)**2) / (1 + (((x  -mu) * tau) / sigma**2))

        return val
    #y = np.asarray([formula(x_, mu, sigma, tau) for x_ in x])
    y = np.asarray([formula1(x_, mu, sigma) for x_ in x])
    return h * y
