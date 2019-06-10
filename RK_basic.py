# -*- coding: utf-8 -*-
"""
        Randomized Kaczmarz

@author: Xining Xu 17110180016
"""

import numpy as np
import matplotlib.pyplot as plt


def build_normal_matrix(m,n):
    np.random.seed(200)
    A = np.random.normal(0,1,(m,n))
    A_norm = np.linalg.norm(A, axis = 1, ord = 2)
    
    return A, np.array(A_norm).reshape(m,1)



def build_complex_matrix(m,n):
    np.random.seed(1000)
    t = np.random.uniform(0,1,m+2)
    t = sorted(t)
    A = np.zeros((m,n))
    for i in range(m):
        for k in range(n):
            # use only real part
            A[i][k] = np.sqrt((t[i+2]-t[i])/2)*np.exp((0+1j)*2*np.pi*k*t[i+1])
            
    A_norm = np.linalg.norm(A, axis = 1, ord = 2)
        
    return A,np.array(A_norm).reshape(m,1) 
   
    

def classical_K(A, b, x_acc, x0, iter_max):
    x = x0 
    res = x - x_acc
    res_norm_record = [np.linalg.norm(res,2)]
    
    for k in range(iter_max):
        r = k%m 
        a = A[r].reshape((1,n))
        v = b[r] - np.dot(a,x)
        v_norm = np.linalg.norm(a,2)**2
        v = v/v_norm
        x = x + v*a.T
        r_norm2 = np.linalg.norm(x - x_acc,2)
        res_norm_record.append(float(r_norm2))
        
    return res_norm_record
    

def simple_RK(A, b, x_acc, x0, iter_max):
    x = x0 
    res = x - x_acc
    res_norm_record = [np.linalg.norm(res,2)]
    
    np.random.seed()
    for k in range(iter_max):
        r = np.random.randint(0,m)
        a = A[r].reshape((1,n))
        v = b[r] - np.dot(a,x)
        v_norm = np.linalg.norm(a,2)**2
        v = v/v_norm
        x = x + v*a.T
        r_norm2 = np.linalg.norm(x - x_acc,2)
        res_norm_record.append(float(r_norm2))
        
    return res_norm_record

def RK(A, b, x_acc, x0, iter_max, A_norm):
    x = x0
    res = x - x_acc
    res_norm_record = [np.linalg.norm(res,2)]
    
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

def upper_bound(A, x, x_acc, iter_max, A_norm):
    _, sigma,_ = np.linalg.svd(A)
    r0_norm = np.linalg.norm(x-x_acc)
    alpha = 1-sigma[-1]**2/np.sum(A_norm**2)
    k = np.arange(0,iter_max+1)
    r_norm_bd = r0_norm*alpha**(k/2)
    
    return r_norm_bd
    

  
    
if __name__=='__main__':

    iter_max = 15000
    run_max = 20
    figcont = 0
    cont = np.arange(0,iter_max+1)
    
    
    """
    =====================================
            Complex Example
    =====================================
    """
    m = 700
    n = 101
    
    A_cp, A_cp_norm = build_complex_matrix(m,n)
    x_acc = np.ones((n,1))
    x = np.array(np.random.uniform(-100,100,(n,1))) 
    b_cp = np.dot(A_cp,x_acc)
    
    r_norm_CK = classical_K(A_cp, b_cp, x_acc, x, iter_max)
    r_norm_bd = upper_bound(A_cp, x, x_acc, iter_max,A_cp_norm)
    
    r_norm_SRK = []
    r_norm_RK = []
    for i in range(run_max):
        r_norm_SRK.append(simple_RK(A_cp, b_cp, x_acc, x, iter_max))
        r_norm_RK.append(RK(A_cp, b_cp, x_acc, x, iter_max, A_cp_norm))
        if i%10 == 0:
            print(str(i))
    
    r_norm_SRK = np.mean(np.array(r_norm_SRK),axis=0)
    r_norm_RK = np.mean(np.array(r_norm_RK),axis=0)
    
    # Plot Figure
    plt.figure(1, figsize = (9,7))
    plt.semilogy(cont,r_norm_CK,
                 linewidth = '4',linestyle = '-.', 
                 color = 'mediumblue', label = "Kaczmarz")
    plt.semilogy(cont,r_norm_SRK,
                 linewidth = '4',linestyle = '--', 
                 color = 'black', label = "Simple Randomized Kaczmarz")
    plt.semilogy(cont,r_norm_RK,
                 linewidth = '4',linestyle = '-', 
                 color = 'mediumorchid', label = "Randomized Kaczmarz")
    plt.semilogy(cont,r_norm_bd,
                 linewidth = '5',linestyle = ':', 
                 color = 'red', label ="Upper Bound for RK")
    
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    plt.xticks(np.arange(0,4)*5000)
    plt.yticks(np.logspace(-16,4, num = 5))
    plt.xlim(0,iter_max)
    plt.ylim(1e-16,10**(8.5))
    legend = plt.legend(fontsize = 17)
    plt.ylabel(r"$E||x_{k}-x||_2 $",fontsize = "22")
    plt.xlabel("Number of projections "+ r"$k$", fontsize = "22")
    plt.tick_params(labelsize=15)
    plt.title("Nonuniform Sampling  ",fontsize=24)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    #plt.savefig("fig1_basic_fourier.jpg")
    #plt.savefig("fig1_basic_fourier.eps")
    plt.show()
    
    
    """
    =====================================
            Normal Random Example
    =====================================
    """
    m = 2000
    n = 100
    
    A_nm, A_nm_norm = build_normal_matrix(m,n)
    x_acc = np.ones((n,1))
    x = np.random.uniform(-100,100,(n,1))
    b_nm = np.dot(A_nm,x_acc)
    
    r_norm_CK = classical_K(A_nm, b_nm, x_acc, x, iter_max)
    r_norm_bd = upper_bound(A_nm, x, x_acc, iter_max, A_nm_norm)
    
    r_norm_SRK = []
    r_norm_RK = []
    
    for i in range(run_max):
        r_norm_SRK.append(simple_RK(A_nm, b_nm, x_acc, x, iter_max))
        r_norm_RK.append(RK(A_nm, b_nm, x_acc, x, iter_max, A_nm_norm))
        if i%5 ==0:
            print(str(i))
        
    r_norm_SRK = np.mean(np.array(r_norm_SRK), axis =0)
    r_norm_RK = np.mean(np.array(r_norm_RK), axis =0)

    
    # Plot Figure
    figcont += 1
    plt.figure(figcont, figsize = (9,7))
    plt.semilogy(cont,r_norm_CK,
                 linewidth = '4', linestyle = '-.', 
                 color = 'mediumblue', label = "Kaczmarz") 
    plt.semilogy(cont,r_norm_SRK,
                 linewidth = '4',linestyle = '--', 
                 color = 'black', label = "Simple Randomized Kaczmarz")  
    plt.semilogy(cont,r_norm_RK,
                 linewidth = '4',linestyle = '-', 
                 color = 'mediumorchid', label = "Randomized Kaczmarz")
    plt.semilogy(cont,r_norm_bd,
                 linewidth = '4',linestyle = ':', 
                 color = 'red', label = "Upper Bound for RK")
    
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    plt.xticks(np.arange(0,4)*5000)
    plt.yticks(np.logspace(-16,4, num = 5))
    plt.xlim(0,iter_max)
    plt.ylim(1e-16,1e+8)
    legend = plt.legend(fontsize = 17)
    plt.ylabel(r"$E||x_{k}-x||_2 $",fontsize = "22")
    plt.xlabel("Number of projections "+ r"$k$", fontsize = "22")
    plt.tick_params(labelsize=15)
    plt.title("Gaussian " + str(m) + " by " + str(n),fontsize=24)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    #plt.savefig("fig2_basic_Gaussian.jpg")
    plt.show()
    
    
    
    
    
    
        
        
        
        
