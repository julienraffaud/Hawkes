import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def cif(t, P, mu, alpha, beta):
    
    " Conditional intensity function of a Hawkes process with parameters mu, alpha, beta. "
    " The CIF has function:                                                               "
    " lambda*(t) = mu + sum(alpha*exp(-beta*(t - t[i])))                                  "
    
    return mu + sum(alpha*np.exp(-beta*np.subtract(t, P[np.where(P<t)])))



def simulate(T, mu, alpha, beta):
    
    "Ogata modified thinning algorithm to simulate Hawkes processes "

    e = 10**(-10)
    P = np.array([]); t = 0

    while (t < T):

        "find new upper bound M "
        
        M = cif(t+e, P, mu, alpha, beta)

        "generate next candidate point "
        
        E = np.exp(M)
        t += E

        "accept it with some probability: U[0, M] "
        
        U = np.random.uniform(0,M)

        if (t < T) and (U <= cif(t, P, mu, alpha, beta)):
            
            P = np.append(P, t)

    return P



def ll(params, beta, t, verbose=False):
    
    " HP log-likelihood objective function.                                               "
    " The function is:                                                                    "
    " ll = -t[k]*mu + alpha/beta*sum(exp(-beta*(t[k] - t[i])-1) + ...                     "
    "      sum(log(mu+alpha*A[i]))                                                        "
    " with:                                                                               "
    " A[0] = 0                                                                            "
    " A[i] = exp(-beta*(t[i] - t[i-1])) * (1 + A[i-1])                                    "
    " For computational efficiency here i compute the LL sums separately, defining:       "
    " s1 = alpha/beta*sum(exp(-beta*(t[k] - t[i])) - 1)                                   "
    " s2 = sum(log(mu + alpha*A[i]))                                                      "
    
    mu, alpha = params[0], params[1]
    
    " compute s1 "
    
    s1 = alpha/beta*sum(np.exp(-beta*(t[-1] - t))-1)
    
    " compute s2 "
    
    A = np.zeros((len(t),1))
    for i in range(1,len(t)):
        A[i] = np.exp(-beta*(t[i] - t[i-1]))*(1+A[i-1])
    s2 = sum(np.log(mu + alpha*A))
    
    return (mu*t[-1] - s1 - s2)



def mle(t, beta, verbose=False):
    
    " Maximum-Likelihood Estimation for parameters mu & alpha "  
    " given a sequence of observations and a beta parameter.  "
    
    
    "generate random parameter estimates"
    
    params = np.random.uniform(0,1,size=2)
    
    "minimize the negative log-likelihood function"
    
    res = minimize(ll, params, args=(beta, t, verbose), method="L-BFGS-B",
                options={"ftol": 1e-10, "maxls": 50, "maxcor":50, "maxiter":100000, "maxfun": 1000})
    
    "return estimated mu & alpha"
    
    return res.x



def pp_plot(t, mu, alpha, beta):
    
    "Plot the point process and conditional intensity function lambda*(t) "
    
    x = np.linspace(0, t[-1], 300)
    ci = [cif(i, t, mu, alpha, beta) for i in x]
    t = [round(i) for i in t]
    xx = [round(i) for i in x]
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
    ax2.set_xlim([0, t[-1]])

    plt.show()
