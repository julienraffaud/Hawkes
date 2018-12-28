import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def cif(t, P, mu, alpha, beta):
    
    " Conditional intensity function of a Hawkes process with exponential kernel.                  "
    "                                                                                              "
    " The conditional intensity function is:                                                       "
    " lambda*(t) = mu + sum(alpha*exp(-beta*(t - t[i])))                                           "
    "                                                                                              "
    " Parameters:                                                                                  "
    " - mu corresponds to the baseline intensity of the HP.                                        "
    " - alpha corresponds to the jump intensity, representing the jump in intensity upon arrival.  "
    " - beta is the decay parameter, governing the exponential decay of intensity.                 "
    
    return mu + sum(alpha*np.exp(-beta*np.subtract(t, P[np.where(P<t)])))



def simulate(T, mu, alpha, beta):
    
    "Ogata modified thinning algorithm to simulate Hawkes processes "

    e = 10**(-10)
    P = np.array([]); t = 0

    while (t < T):

        # find new upper bound M
        M = cif(t+e, P, mu, alpha, beta)

        # generate next candidate point 
        E = -(1/M)*np.log(np.random.uniform(0, 1))
        t += E

        # accept it with some probability: U[0, M]
        U = np.random.uniform(0, M)

        if (t < T) and (U <= cif(t, P, mu, alpha, beta)):
            
            P = np.append(P, t)

    return P


def ll(params, t, verbose=False):
    
    " HP log-likelihood objective function.                                               "
    "                                                                                     "
    " The function is:                                                                    "
    " ll = -t[k]*mu + alpha/beta*sum(exp(-beta*(t[k] - t[i])-1) + ...                     "
    "      sum(log(mu+alpha*A[i]))                                                        "
    "                                                                                     "
    " with:                                                                               "
    " A[0] = 0                                                                            "
    " A[i] = exp(-beta*(t[i] - t[i-1])) * (1 + A[i-1])                                    "
    "                                                                                     "
    " For computational efficiency here i compute the LL sums separately, defining:       "
    " s1 = alpha/beta*sum(exp(-beta*(t[k] - t[i])) - 1)                                   "
    " s2 = sum(log(mu + alpha*A[i]))                                                      "
    
    mu, alpha, beta = params[0], params[1], params[2]
    
    # compute s1 
    s1 = alpha/beta*sum(np.exp(-beta*(t[-1] - t))-1)
    
    # compute s2 
    A = np.zeros((len(t),1))
    for i in range(1,len(t)):
        A[i] = np.exp(-beta*(t[i] - t[i-1]))*(1+A[i-1])
    s2 = sum(np.log(mu + alpha*A))
    
    return (mu*t[-1] - s1 - s2)


def mle(t, verbose=False):
    
    " Maximum-Likelihood Estimation for HP parameters "  
    " given a sequence of observations.               "
    
    
    # generate random parameter estimates
    params = np.random.uniform(0,1,size=3)
    
    # minimize the negative log-likelihood function
    res = minimize(ll, params, args=(t, verbose), method="L-BFGS-B",
                options={"ftol": 1e-10, "maxls": 50, "maxcor":50, "maxiter":100000, "maxfun": 1000})
    
    # return estimated mu, alpha, beta
    return res.x[0], res.x[1], res.x[2]


def pp_plot(t, mu, alpha, beta):
    
    "Plot the point process and conditional intensity function lambda*(t) "
    
    x = np.linspace(0, t[-1], 200)
    ci = [cif(i, t, mu, alpha, beta) for i in x]
    t = [round(i, 1) for i in t]
    xx = [round(i, 1) for i in x]
    xxx = [1 if (i in t) else np.nan for i in xx]
    
    plt.figure(1, figsize=(9, 4))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex = ax1)
    
    ax1.plot(x, ci)
    ax1.set_ylabel(r'$\lambda$*(t)')
    ax1.set_xlabel('T')
    plt.tight_layout()
        
    ax2.scatter(x, xxx, edgecolors='white')
    ax2.set_ylabel('Hawkes process')
    ax2.set_xlabel('T')

    plt.show()
