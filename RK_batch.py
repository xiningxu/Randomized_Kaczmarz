# -*- coding: utf-8 -*-
"""
Randomized Kaczmarz with Batch version

@author: Xining Xu 17110180016
"""
import numpy as np
import matplotlib.pyplot as plt


def build_normal_matrix(m,n):
    np.random.seed(0)
    A = np.random.normal(0,1,(m,n))

    r = np.random.randint(0,m-1,50)
    A[r] = np.random.normal(0,10,(50,n))

    A_norm = np.linalg.norm(A,ord=2,axis = 1)
    
    return A, np.array(A_norm).reshape(m,1)

"""
def build_complex_matrix(m,n):
    np.random.seed(1000)
    t = np.random.uniform(0,1,m+2)
    t = sorted(t)
    A = np.zeros((m,n))
    A_norm = []
    for j in range(m):
        for k in range(n):
            A[j][k] = np.sqrt((t[j+2]-t[j])/2)*np.exp((0+1j)*2*np.pi*k*t[j+1])
        A_norm.append(np.linalg.norm(A[j],2))
        
    return A,np.array(A_norm) 
"""

def choice_no_repeat(A_norm,index,J):
    sample = []
    data = np.array(A_norm**2)
    for k in range(J):
        w_cusum = np.cumsum( data/np.sum(data) )
        rand_num = np.random.uniform(0,1,1)
        r = np.min(np.where(w_cusum > rand_num))
        sample.append(int(index[r]))
        data = np.delete(data,r)
        index = np.delete(index,r)
    return np.array(sample)



def RK_batch(A, b, x_acc, x0, iter_max, A_norm, J, gamma):
    x = x0
    res = x - x_acc
    r_norm_record = [np.linalg.norm(res,2)]
    
    np.random.seed()
    for k in range(iter_max):
        r = choice_no_repeat(A_norm,index,J)
        v = b[r] - np.dot(A[r],x)
        v_norm = A_norm[r].reshape(J,1)
        v = v/v_norm**2
        vec = np.mean(v * A[r], axis=0)
        x = x + gamma*vec.reshape((n,1))
        r_norm_record.append(float(np.linalg.norm(x - x_acc,2)))
        if k%1000 == 0:
            print(str(k))
        
    return np.array(r_norm_record)




if __name__=='__main__':

    iter_max = 15000
    run_max = 100
    figcont = 0
    cont = np.arange(0,iter_max+1)
    

    m = 2000
    n = 100
    A, A_norm = build_normal_matrix(m,n)
    x_acc = np.ones((n,1))
    x = np.array(np.random.uniform(-100,100,(n,1))) 
    b = np.dot(A,x_acc)
    
    index = np.arange(0,m)
    
    """
    ===============================
        alpha  = 1
    ===============================
    """
    
    r_norm_1 = RK_batch(A, b, x_acc, x, iter_max, A_norm, 1, 1)
    
    r_norm_10 = RK_batch(A, b, x_acc, x, iter_max, A_norm, 10, 1)

    r_norm_50 = RK_batch(A, b, x_acc, x, iter_max, A_norm, 50, 1)

    r_norm_100 = RK_batch(A, b, x_acc, x, iter_max, A_norm, 100, 1)
    
    plt.figure(3, figsize = (9,7))
    plt.semilogy(np.arange(0,iter_max+1),r_norm_1,
                 linewidth = "4",linestyle = '-', 
                 color = 'mediumorchid',label = r"$|\tau_k| = 1$")
    
    plt.semilogy(np.arange(0,iter_max+1),r_norm_10,
                 linewidth = "4", linestyle = '--', 
                 color = 'black', label = r"$|\tau_k| = 10$")

    plt.semilogy(np.arange(0,iter_max+1),r_norm_50,
                 linewidth = "5", linestyle = ':', 
                 color = 'red', label = r"$|\tau_k| = 50$")

    plt.semilogy(np.arange(0,iter_max+1),r_norm_100,
                 linewidth = "4", linestyle = '-.', 
                 color = 'mediumblue',label = r"$|\tau_k| = 100$")
    
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    plt.xlim(1e-16,iter_max)
    plt.xticks(np.arange(0,iter_max//5000 + 1)*5000)
    plt.ylim(1e-16,1e+4)
    plt.yticks(np.logspace(-16,4,num =6))
    legend = plt.legend(fontsize = 20,ncol = 2)
    plt.ylabel(r"$||x_{k}-x||_2 $",fontsize = "22")
    plt.xlabel("Number of iterations "+ r"$k$", fontsize = "22")
    plt.tick_params(labelsize=17)
    plt.title("Gaussian " + str(m) + " by " + str(n),fontsize=25)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    #plt.savefig("fig3_batch_Gaussian.eps")
    #plt.savefig("fig3_batch_Gaussian.jpg")
    plt.show()
    
    """
    ======================
        different alpha
    ======================
    """
    alpha = [0.5, 1.0, 5, 18]
    linsty = ['-','--',':','-.']
    colorlist = ['mediumorchid','black','mediumblue','red',]
    
    plt.figure(4, figsize = (9,7))
    for i in range(4):
        r_norm = RK_batch(A, b, x_acc, x, iter_max, A_norm, 10, alpha[i])
        plt.semilogy(np.arange(0,iter_max+1),r_norm,
                 linewidth = "4",linestyle = linsty[i], 
                 color = colorlist[i], label = r"$\alpha ="+str(alpha[i]) +"$")

    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    plt.xlim(1e-16,iter_max)
    plt.xticks(np.arange(0,iter_max//5000 + 1 )*5000)
    plt.ylim(1e-16,1e+4)
    plt.yticks(np.logspace(-16,4,num =6))
    legend = plt.legend(fontsize = 20,ncol = 2)
    plt.ylabel(r"$||x_{k}-x||_2 $",fontsize = "22")
    plt.xlabel("Number of iterations "+ r"$k$", fontsize = "22")
    plt.tick_params(labelsize = 17)
    plt.title("Gaussian " + str(m) + " by " + str(n),fontsize=25)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    #plt.savefig("fig4_batch_Gaussian_alpha.eps")
    #plt.savefig("fig4_batch_Gaussian_alpha.jpg")
    plt.show()
    