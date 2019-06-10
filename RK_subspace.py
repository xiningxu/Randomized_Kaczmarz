# -*- coding: utf-8 -*-
"""
    Two-Subspace Randomized Kaczmarz
    
@author: Xining Xu 17110180016
"""
import numpy as np
import matplotlib.pyplot as plt


def build_uniform_matrix(m,n,c):
    np.random.seed(100)
    A = np.random.uniform(c,1,(m,n))
    A_norm  = np.linalg.norm(A, 
                             axis = 1, ord = 2).reshape((m,1))
    A = A/A_norm
    coh = []
    for i in range(m):
        for j in range(i+1,n):
            a1 = A[i].reshape((1,n))
            a2 = A[j].reshape((n,1))
            a = np.dot(a1,a2)
            coh.append(np.abs(a))
    
    coh = np.array(coh)
    delta_max = np.max(coh)
    delta_min = np.min(coh)
    return A, delta_max, delta_min



def RK(A, b, x_acc, x0, iter_max, A_norm):
    x = x0
    res_norm_record = [np.linalg.norm(x - x_acc, ord = 2)]
    
    A_norm = A_norm**2/np.sum(A_norm**2)
    A_csum = np.cumsum(A_norm)
    
    np.random.seed()
    for k in range(iter_max):
        rand_num = np.random.uniform(0,1)
        r = np.min(np.where(A_csum > rand_num))
        a = A[r].reshape((1,n))
        v = b[r] - np.dot(a,x)
        v_norm = np.linalg.norm(a,2)**2
        v = v/v_norm
        x = x + v*a.T
        r_norm2 = np.linalg.norm(x - x_acc,2)
        res_norm_record.append(float(r_norm2))
        
    return res_norm_record

def choice_row(A,b,m):
    while 1:
        r1, r2 = np.random.randint(0,m, 2)
        if (r1-r2):
            break
    return A[r1].reshape((n,1)), A[r2].reshape((n,1)),b[r1],b[r2]
        
  
def Subspace_RK(A, b, x_acc, x0, iter_max, A_norm):
    x = np.array(x0)
    iter_seq = [0]
    r_norm_record = [np.linalg.norm(x-x_acc, ord = 2)]
    
    np.random.seed()
    for k in range(iter_max//2):
        a1, a2, b1, b2 = choice_row(A,b,m)
        mu = np.dot(a1.T, a2)
        y = x + (b2 - np.dot(a2.T, x))*a2
        v = a1 - mu * a2
        v = v / np.sqrt(1 - mu**2)
        beta = b1 - mu * b2
        beta = beta / np.sqrt(1 - mu**2)
        x = y + (beta - np.dot(v.T,y))*v
        
        iter_seq.append(2*int(k+1))
        r_norm_record.append(np.linalg.norm(x-x_acc, ord = 2))
        
    return np.array(iter_seq), np.array(r_norm_record)
      
    
    
if __name__=='__main__':
    
    iter_max = 15000
    run_max = 50
    figcont = 0
    
    m = 300
    n = 100
    
    c = 0
    A, d1, d2 = build_uniform_matrix(m,n,c)
    x_acc = np.ones((n,1))
    b = np.dot(A, x_acc)
    A_norm = np.ones((m,1))
    
    x = np.random.uniform(-100,100,(n,1))
    
    
    r_RK = []
    r_2SRK = []
    for i in range(run_max):
        r_RK.append(RK(A, b, x_acc, x, iter_max, A_norm))
        iter_seq, r_norm_2RK = Subspace_RK(A, b, x_acc, x, iter_max, A_norm)
        r_2SRK.append(np.array(r_norm_2RK))
    r_RK = np.mean(np.array(r_RK), axis=0)    
    r_2SRK = np.mean(np.array(r_2SRK), axis=0)
    
    figcont += 1
    plt.figure(0,  figsize = (9,6))
    plt.semilogy(np.arange(0,iter_max+1),r_RK/r_RK[0],
                 linewidth='4',color = 'mediumblue', label = "RK")
    plt.semilogy(iter_seq, r_norm_2RK/r_norm_2RK[0],
                 linewidth='4',linestyle = '--',
                 color = 'firebrick', label = "Two-subspace RK")
    plt.xlabel("Number of projections " + r"$k$", fontsize =22)
    plt.ylabel(r"$||x_k -x||_2 \, / \,||x_0-x||_2}$", fontsize = 20)
    
    plt.legend(fontsize = 22, loc = 'lower left')
    plt.tick_params(labelsize=18)
    plt.xlim(0,iter_max)
    plt.ylim(1e-12,1e+0)
    plt.xticks(np.arange(0,iter_max+1,step = 5000))
    plt.yticks(np.logspace(-12,0,num = 5))
    plt.title(r"$c = " + str(c) + "$   ", fontsize = 25)
    
    ax = plt.gca()
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    
    plt.savefig("fig4_2SRK" + str(figcont)+".eps")
    plt.savefig("fig4_2SRK" + str(figcont)+".jpg")
    

