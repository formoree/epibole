import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def check_normality(pd_series):
    """Creates a panel of three plots for checking Normality.
    
    The input is a pandas series of real numbers.
    
    The output is a plot figure with three subplots - a boxplot, a histogram and 
    a qq-plot.
    """
    plt.figure(1, figsize=(15, 4))
    plt.subplot(131)
    pd_series.plot(kind='box')

    plt.subplot(132)
    pd_series.hist(grid=False);

    plt.subplot(133)
    stats.probplot(pd_series, plot=plt);
    
def generate_one_sample(delta_m, sd1, n, alpha=0.05):
    """Simulates data and output for two-sample t-test.
    
    delta_m: true difference in means.
    sd1: standard deviation within each group.
    n: sample size in each group.
    alpha: significance level.
    
    A return value of 1 indicates the null hypothesis was rejected.
    A value of 0 is returned otherwise.
    """
    X = np.random.randn(n)*sd1
    Y = np.random.randn(n)*sd1 + delta_m
    
    ts2 = stats.ttest_ind(X, Y)
    if ts2.pvalue < alpha:
        return 1 # 1 means reject H0
    else:
        return 0

def estimate_power(delta_m, sd1, n, alpha=0.05, nsim=2000):
    """Estimates power for a two-sample t-test.
    
    delta_m: true difference in means.
    sd1: standard deviation within each group.
    n: sample size in each group.
    alpha: significance level.
    nsim: number of simulations to run to estimate the power.
    
    A real number between 0 and 1 is returned. It is an estimate of 
    power for the supplied configuration.
    """ 
    x = [generate_one_sample(delta_m, sd1, n,alpha) for ii in np.arange(0, nsim)]
    return np.mean(x)