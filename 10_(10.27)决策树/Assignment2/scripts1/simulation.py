import numpy as np
import pandas as pd
import simpy

from numpy.random import default_rng

rng = default_rng(5003)

def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        
        choices = np.array(list(cfdist[word].keys()))
        pp = np.array(list(cfdist[word].values()))
        pp = pp / np.sum(pp)
        word = rng.choice(choices, size=1, p = pp)[0]
        
# Configuration #1
def s1(env, interval, counter, service, cust_dict, verbose=False):
    """Source generates customers randomly #1.
    
    Only depicts arrivals, waiting and departures.
    All customers wait to be served.
    Arrival, wait and service times are monitored.
    Inter-arrival times are Exponential distributions.
    Service times are shifted Exponential distributions, because there is a minimum time of 1.5 mins.
    
    interval:  mean inter-arrival time
    counter:   the resource, the sandwich counter
    service:   mean service time
    cust_dict: an empty dictionary, where the monitored values will be stored.
    verbose:   True or False, indicating if output should be printed to console.
    """
    #for i in range(200):
    i = 0
    while True:
        i += 1
        inter_arr = rng.exponential(interval)
        yield env.timeout(inter_arr)
        c = c1(env, i, counter, service, cust_dict, verbose)
        env.process(c)

def c1(env, cust_num, counter, service, cust_dict, verbose):
    """Customer arrives, is served and leaves.
    
    cust_num: customer id number (integer)
    counter:  the resource
    service: 
    cust_dict: 
    verbose: 
    """
    arrive = env.now
    if verbose:
        print(f'ARR: Customer {cust_num:2d} arrives at time {arrive:.3f} minutes.')
    #print('ARR: Customer {:2d} arrives at time {:.3f} minutes.'.format(cust_num, arrive))
    cust_dict[cust_num] = np.array([np.nan, np.nan, np.nan])
    cust_dict[cust_num][0] = env.now

    with counter.request() as req:
        yield req

        wait = env.now - arrive
        cust_dict[cust_num][1] = wait
        if verbose:
            print(f'SERV: Now the time is {env.now:.3f}, {cust_num:2d} waited {wait:.3f} minutes')
        #print('SERV: Now the time is {0:.3f}, {1} waited {2:.3f} minutes'.format(env.now, str(cust_num), wait))
        service_time = rng.exponential(service) + 1.5
        cust_dict[cust_num][2] = service_time
        yield env.timeout(service_time)
        if verbose:
            print(f'DEP: {cust_num:2d} leaving, at {env.now:.3f} minutes past 11')
        #print('DEP: {} leaving, at {:.3f} minutes past 11'.format(str(cust_num), env.now))
            
# Configuration #2
def s2(env, interval, counter, service, cust_dict, verbose=False):
    """Source generates customers randomly #2.
    
    Depicts arrivals, waiting, departures, queue length and amount spent.
    Some customers will renege.
    Arrival, wait and service times, queue length, amount spent are monitored.
    Inter-arrival times are Exponential distributions,
    Service times are shifted Exponential distributions, because there is a minimum time of 1.5 mins.
    Renege time is uniform.
    Amount spent is gamma, with mean 3.

    In the output dict, 0 means did not renege, and 1 means reneged. The
    6 coordinates in the tuple correspond to arrival time, wait time, 
    service time, renege status, amount spent and queue length.
    
    interval:  mean inter-arrival time
    counter:   the resource, the sandwich counter
    service:   mean service time
    cust_dict: an empty dictionary, where the monitored values will be stored.
    verbose:   True or False, indicating if output should be printed to console.
    """
    #for i in range(200):
    i = 0
    while True:
        i += 1
        inter_arr = rng.exponential(interval)
        yield env.timeout(inter_arr)
        c= c2(env, i, counter, cust_dict, verbose)
        env.process(c)

def c2(env, cust_num, counter, cust_dict, verbose):
    """Customer arrives, is served and leaves; or loses patience and reneges."""
    arrive = env.now
    if verbose:
        print(f'ARR: Customer {cust_num:2d} arrives at time {arrive:.3f} minutes.')
        # arrival time, wait time, service time, renege status, amount spent, queue length
    cust_dict[cust_num] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    cust_dict[cust_num][0] = env.now
    cust_dict[cust_num][5] = len(counter.queue)

    with counter.request() as req:
        patience = rng.uniform(0.5, 4.0)
        amt_spent = rng.gamma(3, 2)
        cust_dict[cust_num][4] = amt_spent
        results = (yield req | env.timeout(patience))

        wait = env.now - arrive

        if req in results:
            cust_dict[cust_num][1] = wait
            cust_dict[cust_num][3] = 0 # means did not renege
            if verbose:
                print(f'SERV: Now the time is {env.now:.3f}, {cust_num:2d} waited {wait:.3f} minutes')
            service_time = rng.exponential(3)
            cust_dict[cust_num][2] = service_time
            yield env.timeout(service_time)
            if verbose:
                print(f'DEP: {cust_num:2d} leaving, at {env.now:.3f} minutes past 11')
        else:
            cust_dict[cust_num][1] = wait
            cust_dict[cust_num][3] = 1 # means did renege
            if verbose:
                print(f'REN: {cust_num:2d} leaving, at {env.now:.3f} minutes past 11')
