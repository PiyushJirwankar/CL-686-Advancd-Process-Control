import numpy as np
from scipy.linalg import expm
import random
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp as min_func
from scipy.optimize import Bounds as Bounds
from RK4_project import RK4 as RK4

np.random.seed(0)

A_mat = np.array([[-2.16087226e+00,  1.11251174e-02,  2.50592906e+00],
       [-5.80436132e+00, -9.44374413e-01,  1.25296453e+01],
       [ 0.00000000e+00,  0.00000000e+00, -5.00000000e-01]])

B_mat = np.array([[-0.40094865,  0.        ],
       [-2.00474325,  1.        ],
       [ 0.16      ,  0.        ]])

C_mat = np.array([[0, 1., 0],
       [0, 0, 1.]])

D_mat = np.array([[0., 0.],
       [0., 0.]])

# H_mat = np.array([[0.86639882],
#        [4.33199408],
#        [0.        ]])
H_mat = np.array([0.86639882, 4.33199408, 0.0])

start = 0
end = 25
N = 10*(end-start)+1
time = np.linspace(start, end, N)
h = (time[1]-time[0])

phi = expm(A_mat*h)
gamma_u = np.dot((phi - np.eye(3)), np.dot(np.linalg.inv(A_mat),B_mat))
gamma_d = np.dot((phi - np.eye(3)), np.dot(np.linalg.inv(A_mat),H_mat))

# Defining parameters for continuous time system from Assignment 1 code
alph1 = 300000.0
alph2 = 60000000.0
alph3 = 0.16
alph4 = 5.0
alph5 = 0.4

Us = np.array([1.0, 390.0])
Ds = 1.0
Xs = np.array([0.400948649, 392.004743, 0.16])
Ys = np.array([392.004743, 0.16])

C = np.array([[0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])

dk_sigma = 0.015
meas_sigma = np.array([0.2, 0.0035])
R = np.array([[meas_sigma[0]**2, 0.0],[0.0, meas_sigma[1]**2]])

X = np.zeros((3,N))
X[:,0] = Xs

U1_arr = Us[0]*np.ones(N)
U2_arr = Us[1]*np.ones(N)

D = np.zeros(N)
D[0] = Ds

Y = np.zeros((2,N))
Y[:,0] = Ys

# loop for defining control U
U1_amp = 0.15
U2_amp = 10.0


################################################
######### State augmentation matrices ##########
################################################

phi_a = np.zeros((5,5))
phi_a[:3,:3] = phi
phi_a[:3, 3:] = gamma_u
phi_a[3:, 3:] = np.eye(2)

gamma_u_a = np.zeros((5,2))
gamma_u_a[:3, :] = gamma_u

gamma_d_a = np.zeros((5,3))
gamma_d_a[:3, 0] = gamma_d
gamma_d_a[3:, 1:] = np.eye(2)

C_a = np.zeros((2,5))
C_a[:,:3] = C

gamma = 2.0/7.0
Q_beta = gamma*np.array([[0.001**2, 0.0],
                         [0.0, 0.05**2]])
Q_d = dk_sigma**2

Q_a = np.zeros((3,3))
Q_a[0,0] = Q_d
Q_a[1:,1:] = Q_beta

R_a = R

Wx = np.array([[10000.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 100.0]])

eta = 2.0/7.0
Wu = eta*np.array([[1.0, 0.0],
                   [0.0, 10.0]])

W_delta_u = 5.0*eta*np.eye(2)

### The following matrices are obtained using MATLAB functions and are then copied here
L_a = np.array([[0.0006 ,   0.0110],
    [0.0943 ,   0.0201],
    [0.0000 ,   0.0359],
    [0.0000 ,   0.1500],
    [0.1274 ,  -0.0396]])

Pa_inf = np.array([[0.0000 ,   0.0000  ,  0.0000 ,   0.0000 ,   0.0000],
                   [0.0000  ,  0.0040 ,   0.0000  ,  0.0000 ,   0.0056],
                   [0.0000 ,   0.0000  ,  0.0000  ,  0.0000 ,  -0.0000],
                   [0.0000 ,   0.0000  ,  0.0000  ,  0.0000 ,  -0.0000],
                   [0.0000 ,   0.0056 ,  -0.0000,   -0.0000 ,   0.0109]])

G_inf = np.array([[-11.3451 ,   0.0391  , 17.5651],
                  [0.2217  ,  0.0163 ,   0.8950]])

S_inf = 1.0e+4*np.array([[2.1524  ,  0.0081  ,  3.1617],
                  [0.0081  ,  0.0006  ,  0.0317],
                  [3.1617  ,  0.0317 ,   9.8595]])
            
Ev = np.array([0.9505, 0.9036, 0.1539])

W_inf = 1.0e+05*np.array([[0.2831 ,  0.0007 ,   0.21000],
                          [0.0007 ,  0.0001 ,   0.0034],
                          [0.2100  ,  0.0034  , 1.1583]]) ### Using dlyap function in MATLAB
#########################################################################################

xa_hat = np.zeros((5,N))
# xa_hat[:,0] = np.ones(5)
xa_true = np.zeros((5,N))

xs = np.zeros((3,N))

######## Using input-bias approach, find const matrices for control ###########
Ku = np.dot(C, np.dot(np.linalg.inv(np.eye(3)-phi), gamma_u))

######## Defining reference trajectory r(k) and disturbance mean value Ds(k) #########
r = np.zeros((2,N))
R = np.zeros((2,N))
Ys = np.dot(C, Xs)

for i in range(60):
    R[:,i] = np.dot(C, Xs)

for i in range(60, 120):
    r[:,i] = np.array([10.0, -0.025])
    R[:,i] = r[:,i] + Ys
for i in range(120,N):
    R[:,i] = np.dot(C, Xs)


Ds = np.ones(N)
D = np.zeros(N)
for i in range(180, N):
    Ds[i] = 1.5

uk_arr = np.zeros((2,N-1))
ea = np.zeros((2,N))

#################################
p = 40
q = 5
#################################
ukm1 = 0.0

Uf0 = np.zeros(2*p)

b1 = np.zeros(80)
b2 = np.zeros(80)
for i in range(40):
    b1[2*i] = -1.0
    b1[2*i+1] = -40.0
    b2[2*i] = 1.0
    b2[2*i+1] = 40.0
arr1 = [-1.0, 1.0]
arr2 = [-40.0, 40.0]
bounds = [arr1, arr2]
bound = bounds*40

Y= np.zeros((2,N))
Yhat = np.zeros((2,N))

U = np.zeros((2,N-1))

sse1_mpc = 0.0
sse2_mpc = 0.0
ssmv1_mpc = 0.0
ssmv2_mpc = 0.0


for i in range(N-1):  ## Add disturbance steady state values and control bounds
    yk = np.dot(C, xa_true[:3,i]) + np.random.multivariate_normal(np.zeros(2), R_a)
    Y[:,i] = yk + np.dot(C, Xs)
    sse1_mpc = sse1_mpc + (Y[0,i]-R[0,i])**2
    sse2_mpc = sse2_mpc + (Y[1,i]-R[1,i])**2
    Yhat[:,i] = np.dot(C_a, xa_hat[:,i]) + np.dot(C, Xs)
    ea_k = yk - np.dot(C_a, xa_hat[:,i])
    ea[:,i] = ea_k
    us = np.dot(np.linalg.inv(Ku), r[:,i]) - xa_hat[3:,i]
    xs[:,i] = np.dot(np.dot(np.linalg.inv(np.eye(3)-phi), gamma_u), np.dot(np.linalg.inv(Ku), r[:,i]))
    xhat = xa_hat[:3, i]
    beta = xa_hat[3:,i]
    def func(Uf):
        Uf_mat = np.zeros((2,p))
        for k in range(p):
            Uf_mat[:,k] = np.array([Uf[2*k], Uf[2*k+1]])
        zhat = np.zeros((3,p))
        for k in range(q,p):
            Uf_mat[:,k] = Uf_mat[:,q-1]
        zhat[:,0] = xhat
        for k in range(p-1):
            zhat[:,k+1] = np.dot(phi, zhat[:,k]) + np.dot(gamma_u, (Uf_mat[:,k]+beta))
        delu_kk = Uf_mat[:,0] - ukm1
        cost = np.dot(delu_kk.transpose(), np.dot(W_delta_u, delu_kk))
        for k in range(1,p):
            eps_kpj = xs[:,i] - zhat[:,k]
            du_kpj = Uf_mat[:,k] - us
            delu_kpj = Uf_mat[:,k] - Uf_mat[:,k-1]
            if k<=q:
                cost = cost + np.dot(du_kpj.transpose(), np.dot(Wu, du_kpj)) + np.dot(delu_kpj.transpose(), np.dot(W_delta_u, delu_kpj))
            cost = cost + np.dot(eps_kpj.transpose(), np.dot(Wx, eps_kpj))
        eps_kpp = xs[:,i] - zhat[:,p-1]
        cost = cost + np.dot(eps_kpp.transpose(), np.dot(W_inf, eps_kpp))
        return cost
    min_eval = min_func(func, Uf0, bounds=bound)
    uk = np.array([min_eval[0], min_eval[1]])
    U[:,i] = uk + Us
    ssmv1_mpc = ssmv1_mpc + (U[0,i]-Us[0])**2
    ssmv2_mpc = ssmv2_mpc + (U[1,i]-Us[1])**2
    uk_arr[:,i] = uk
    da_k = np.random.multivariate_normal(np.zeros(3), Q_a)
    dk = da_k[0]
    D[i] = dk + Ds[i]
    xa_true[:,i+1] = np.dot(phi_a, xa_true[:,i]) + np.dot(gamma_u_a, uk) + np.dot(gamma_d_a, da_k)
    xa_hat[:,i+1] = np.dot(phi_a, xa_hat[:,i]) + np.dot(gamma_u_a, uk) + np.dot(L_a, ea_k)
    ukm1 = uk
    print(i*100.0/250.0,'%', 'complete')

Y[:,N-1] = np.dot(C, xa_true[:3,N-1]) + np.random.multivariate_normal(np.zeros(2), R_a) + np.dot(C, Xs)
sse1_mpc = sse1_mpc + (Y[0,N-1]-R[0,N-1])**2
sse2_mpc = sse2_mpc + (Y[1,N-1]-R[1,N-1])**2
Yhat[:,N-1] = np.dot(C_a, xa_hat[:,N-1]) + np.dot(C, Xs)
D[N-1] = np.random.normal(1.5, 0.015)

np.savetxt('Y1_MPC.csv', Y[0,:], delimiter=',')
np.savetxt('Y2_MPC.csv', Y[1,:], delimiter=',')
np.savetxt('U1_MPC.csv', U[0,:], delimiter=',')
np.savetxt('U2_MPC.csv', U[1,:], delimiter=',')
np.savetxt('sse1_mpc.csv', sse1_mpc)
np.savetxt('sse2_mpc.csv', sse2_mpc)
np.savetxt('ssmv1_mpc.csv', ssmv1_mpc)
np.savetxt('ssmv1_mpc.csv', ssmv2_mpc)

# plt.plot(time, xa_true[0,:], 'r')
# plt.plot(time, xa_hat[0,:], 'c')
# plt.title('MPC')
# # plt.plot(time, ea[0,:], 'r')
# # plt.plot(time, ea[1,:], 'c')
# # plt.plot(time[:-1], uk_arr[0,:], 'r')
# # plt.plot(time[:-1], uk_arr[1,:], 'c')
# plt.show()