# -*- coding: utf-8 -*-
"""
    Randomized Kaczmarz for Noisy System

@author: Xining Xu 17110180016
"""
import numpy as np
import matplotlib.pyplot as plt


def build_normal_matrix(m,n):
    np.random.seed(200)
    A = np.random.normal(0,1,(m,n))
    A_norm = np.linalg.norm(A, axis = 1, ord = 2)
    
    return A, np.array(A_norm).reshape(m,1)


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

if __name__=='__main__':
    
    m = 2000
    n = 100
    
    A_nm, A_nm_norm = build_normal_matrix(m,n)
    x_acc = np.ones((n,1))
    
    x = np.random.uniform(-100,100,(n,1))
    res = np.linalg.norm(x-x_acc,ord=2)
    b_nm = np.dot(A_nm,x_acc)
    
    
    _, sigma,_ = np.linalg.svd(A_nm)
    A_norm_F = np.sum(np.linalg.norm(A_nm, ord =2, axis = 0)**2)
    R = A_norm_F/sigma[-1]**2
    
    
    """
        convergence
    """
    np.random.seed()
    e = np.random.normal(0,1,(m,1))
    e = 0.2*e/np.linalg.norm(e,ord=2)
    gamma = np.abs(e/(A_nm_norm))
    gamma = np.max(gamma)*np.sqrt(R)
    
    
    iter_max = 5000
    run_max = 100
    
    trail_1 = []
    for i in range(run_max):
        trail_1.append(RK(A_nm, (b_nm + e)/res, x_acc/res, x/res, iter_max, A_nm_norm))
          
    trail_1 = np.array(trail_1)
    trail_1_max = np.max(trail_1, axis = 0)
    trail_1_min = np.min(trail_1, axis = 0)
    trail_1_mean = np.mean(trail_1, axis =0)
    
    
    plt.figure(0,figsize=(8,6))
    plt.semilogy(np.arange(0,iter_max+1),trail_1_max, 
                 linewidth = 1, color = 'lightpink')
    plt.semilogy(np.arange(0,iter_max+1),trail_1_min,
                 linewidth = 1, color = 'lightpink')
    plt.fill_between(np.arange(0,iter_max+1),trail_1_max,trail_1_min,
                     color = 'lightpink', 
                     label = "difference between the maximium \n and the minimium")
    plt.semilogy(np.arange(0,iter_max+1),trail_1_mean, 
                 linewidth = 4, color = 'firebrick', label = "average error")
    plt.semilogy(np.arange(0,iter_max+1), gamma*np.ones((iter_max+1,1)),
                 linewidth = 4, linestyle = ":", color = "blue",
                 label = "threshold "+ r"$\sqrt{R}\gamma$")
    plt.legend(fontsize = 16,loc="upper right")
    plt.xlim(0,iter_max)
    plt.ylim(1e-6,1e+1)
    
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    plt.xticks(np.arange(0,iter_max+1,step=800))
    plt.yticks(np.logspace(-6,1,8))

    plt.ylabel(r"$||x_{k}-x||_2/||x_0-x||_2$",fontsize = "17")
    plt.xlabel(r"Number of iterations $k$", fontsize = "17")
    plt.tick_params(labelsize = 15)
    plt.title("Error in estimation \n (Gaussian " 
              + str(m) + " by " + str(n) +")", fontsize = 18)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    # plt.savefig("fig6_noise_Gaussian.eps")
    # plt.savefig("fig6_noise_Gaussian.jpg")
    
    
    
    """
        check threshold
    """
    iter_max = 800
    run_max = 100
    
    trail = []
    threshold = []
    np.random.seed()
    for i in range(run_max):
        e = np.random.normal(0,1,(m,1))
        e = 0.2*e/np.linalg.norm(e,ord=2)
        gamma = np.abs(e/(A_nm_norm))
        threshold.append(np.max(gamma)*np.sqrt(R)) 
        trail.append(RK(A_nm, (b_nm + e)/res, x_acc/res, x/res, iter_max, A_nm_norm))
        print(str(i))
        
    trail = np.array(trail)
    trail = trail[:,-1]
        
    plt.figure(1,figsize = (8,6))

    plt.plot(np.arange(0,run_max)+1, trail,
             linewidth = "2", color = "navy",
             label = "Error")
    plt.plot(np.arange(0,run_max)+1,np.array(threshold),
             linewidth = "5", color = "firebrick",
             label = "Threshold")
    plt.xlim(0,run_max)
    plt.ylim(0.01,0.052)
    
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    plt.xticks(np.arange(0,101,step=20))
    plt.yticks(np.arange(0.01,0.06,step=0.01))
    legend = plt.legend(fontsize = 20,loc="upper right")
    plt.ylabel(r"$||x_{k}-x||_2/||x_0-x||_2$",fontsize = "17")
    plt.xlabel("Trials", fontsize = "17")
    plt.tick_params(labelsize = 15)
    plt.title("Error in estimation \n (Gaussian " 
              + str(m) + " by " + str(n) + " after " + str(iter_max) 
              + " iterations)", fontsize = 18)
    #plt.savefig("fig3_noisy_Gaussian.eps")
    #plt.savefig("fig3_noisy_Gaussian.jpg")
    