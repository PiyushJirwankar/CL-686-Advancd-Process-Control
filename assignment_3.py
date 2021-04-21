import numpy as np
from RK4_project import RK4 as RK4
import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy.signal

np.random.seed(0)

start = 0
end = 25
N = 10*(end-start)+1
time = np.linspace(start, end, N)
h = time[1]-time[0]

Us = np.array([1.0, 390.0])
Ds = 1.0
Xs = np.array([0.400948649, 392.004743, 0.16])
Ys = np.array([392.004743, 0.16])

C = np.array([[0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])

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

H_mat = np.array([0.86639882, 4.33199408, 0.0])

phi = expm(A_mat*h)
gamma_b = np.dot((phi - np.eye(3)), np.dot(np.linalg.inv(A_mat),B_mat))

X = np.zeros((3,N))
X[:,0] = Xs

x_hat = np.zeros((3,N))
x_hat[:,0] = np.zeros(3)

U = np.zeros((2,N))
# U[:,0] = Us

Y = np.zeros((2,N))
Y[:,0] = np.dot(C_mat, X[:,0])


alph1 = 300000.0
alph2 = 60000000.0
alph3 = 0.16
alph4 = 5.0
alph5 = 0.4

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

# r = np.zeros((2,N))
# D = Ds*np.ones(N)
# for i in range(60,120):
#     r[:,i] = np.array([10.0, -0.025])

# for i in range(180,N):
#     D[i] = Ds + 0.5

Ul = np.array([0.0, 350.0])
Uh = np.array([2.0, 430.0])
diffU1 = 1.0
diffU2 = 40.0

cont_poles = np.array([0.85, 0.9, 0.92])
G = scipy.signal.place_poles(phi, gamma_b, cont_poles).gain_matrix
# print(np.linalg.eig(phi - np.dot(gamma_b, G)))

obs_poles = np.array([0.3, 0.4, 0.5])
L = scipy.signal.place_poles(phi.transpose(), C_mat.transpose(), obs_poles).gain_matrix.transpose()

Ku    = np.dot(C, np.dot(np.linalg.inv((np.eye(3)-phi)), gamma_b))
gamma_h = L
Kbeta = np.dot(C, np.dot(np.linalg.inv((np.eye(3)-phi)), gamma_h))

alpha = 0.8
phi_e = alpha

e = np.zeros((2,N))
e[:,0] = (Y[:,0]-Ys) - np.dot(C_mat, x_hat[:,0])

ef = np.zeros((2,N))
ef[:,0] = (1-alpha)*e[:,0]

us = np.zeros((2,N))
xs = np.zeros((3,N))

dk_sigma = 0.015
meas_sigma = np.array([0.2, 0.0035])
R_cov = np.array([[meas_sigma[0]**2, 0.0],[0.0, meas_sigma[1]**2]])

sse1_asgn3 = (Y[0,0]-R[0,0])**2
sse2_asgn3 = (Y[1,0]-R[1,0])**2
ssmv1_asgn3 = 0.0
ssmv2_asgn3 = 0.0

for i in range(N-1):
    D[i] = Ds[i] + np.random.normal(0.0, dk_sigma)
    yk = Y[:,i] - Ys
    xk = X[:,i] - Xs
    e[:,i] = yk - np.dot(C_mat, x_hat[:,i])
    ef[:,i] = e[:,i] + (ef[:,i-1] - e[:,i])*alpha
    us[:,i] = np.dot(np.linalg.inv(Ku), (r[:,i] - np.dot(Kbeta + np.eye(2), ef[:,i])))
    xs[:,i] = np.dot(np.linalg.inv((np.eye(3)-phi)), (np.dot(gamma_b, us[:,i]) + np.dot(L,ef[:,i])))
    uk = us[:,i] - np.dot(G, (x_hat[:,i] - xs[:,i]))
    uk[0] = np.sign(uk[0])*(np.minimum(diffU1, np.absolute(uk[0]))) # Projecting U1 onto feasible range
    uk[1] = np.sign(uk[1])*(np.minimum(diffU2, np.absolute(uk[1]))) # Projecting U2 onto feasible range
    U[:,i] = uk + Us
    ssmv1_asgn3 = ssmv1_asgn3 + (U[0,i]-Us[0])**2
    ssmv2_asgn3 = ssmv2_asgn3 + (U[1,i]-Us[1])**2
    U1 = U[0,i]
    U2 = U[1,i]
    def sys_dyn(t, X):
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        k1 = alph1*np.exp(-5000.0/x2)
        k2 = alph2*np.exp(-7500.0/x2)
        x1_dot = -alph3*(U1*x1)/x3 + k1*(D[i]-x1) - k2*x1
        x2_dot = alph3*U1*(U2-x2)/x3 + alph4*(k1*(D[i]-x1) - k2*x1)
        x3_dot = alph3*U1 - alph5*np.sqrt(x3)
        x_dot = np.array([x1_dot, x2_dot, x3_dot])
        return x_dot
    X[:,i+1] = RK4(sys_dyn, X[:,i], time[i], time[i+1], h/10.0)
    x_hat[:,i+1] = np.dot(phi, x_hat[:,i]) + np.dot(gamma_b, uk) + np.dot(L, e[:,i])
    Y[:,i+1] = np.dot(C_mat, X[:,i+1]) + np.random.multivariate_normal(np.zeros(2), R_cov)
    sse1_asgn3 = (Y[0,i+1]-R[0,i+1])**2
    sse2_asgn3 = (Y[1,i+1]-R[1,i+1])**2
    e[:,i+1] = (Y[:,i+1]-Ys) - np.dot(C_mat, x_hat[:,i+1])
    # ef[:,i+1] =  phi_e*ef[:,i] + (1.0 - phi_e)*e[:,i+1]

e[:,-1] = (Y[:,-1]-Ys) - np.dot(C_mat, x_hat[:,-1])
ef[:,-1] = e[:,-1] + (ef[:,i-2] - e[:,-1])*alpha
us[:,-1] = np.dot(np.linalg.inv(Ku), (r[:,-1] - np.dot(Kbeta + np.eye(2), ef[:,-1])))
uk = us[:,-1] - np.dot(G, (x_hat[:,-1] - xs[:,-1]))
uk[0] = np.sign(uk[0])*(np.minimum(diffU1, np.absolute(uk[0]))) # Projecting U1 onto feasible range
uk[1] = np.sign(uk[1])*(np.minimum(diffU2, np.absolute(uk[1]))) # Projecting U2 onto feasible range
U[:,-1] = uk + Us
xs[:,-1] = np.dot(np.linalg.inv((np.eye(3)-phi)), (np.dot(gamma_b, us[:,-1]) + np.dot(L,ef[:,-1])))

D[N-1] = Ds[N-1] + np.random.normal(0.0, dk_sigma)

U = U[:,:N-1]

y = np.zeros((2,N))
R = np.zeros((2,N))
X_hat = np.zeros((3,N))
for i in range(N):
    y[:,i] = Y[:,i]-Ys
    R[:,i] = Ys + r[:,i]
    X_hat[:,i] = x_hat[:,i] + Xs

np.savetxt('Y1_asgn3.csv', Y[0,:], delimiter=',')
np.savetxt('Y2_asgn3.csv', Y[1,:], delimiter=',')
np.savetxt('U1_asgn3.csv', U[0,:], delimiter=',')
np.savetxt('U2_asgn3.csv', U[1,:], delimiter=',')
np.savetxt('sse1_asgn3.csv', sse1_asgn3)
np.savetxt('sse2_asgn3.csv', sse2_asgn3)
np.savetxt('ssmv1_asgn3.csv', ssmv1_asgn3)
np.savetxt('ssmv1_asgn3.csv', ssmv2_asgn3)


# fig, ax = plt.subplots(5,3)
# ax[0,0].plot(time*10.0, Y[0,:],'b')
# ax[0,0].plot(time*10.0, R[0,:],'g')
# ax[0,0].set_xlabel('k')
# ax[0,0].set_ylabel('Y1(k) & R1(k)')
# ax[0,0].legend(['Y1(k)', 'R1(k)'], fontsize='xx-small')

# ax[0,1].plot(time*10.0, Y[1,:],'b')
# ax[0,1].plot(time*10.0, R[1,:],'g')
# ax[0,1].set_xlabel('k')
# ax[0,1].set_ylabel('Y2(k) & R2(k)')
# ax[0,1].legend(['Y2(k)', 'R2(k)'], fontsize='xx-small')

# ax[0,2].plot(time*10.0, D,'b',  drawstyle='steps-pre')
# ax[0,2].set_xlabel('k')
# ax[0,2].set_ylabel('D(k)')
# ax[0,2].legend(['D(k)'], fontsize='xx-small')

# ax[1,0].plot(time*10.0, U[0,:],'b', drawstyle='steps-pre')
# ax[1,0].set_xlabel('k')
# ax[1,0].set_ylabel('U1(k)')
# ax[1,0].legend(['U1(k)'], fontsize='xx-small')

# ax[1,1].plot(time*10.0, U[1,:],'b', drawstyle='steps-pre')
# ax[1,1].set_xlabel('k')
# ax[1,1].set_ylabel('U2(k)')
# ax[1,1].legend(['U2(k)'], fontsize='xx-small')

# ax[1,2].plot(time*10.0, e[0,:], 'b')
# ax[1,2].set_xlabel('k')
# ax[1,2].set_ylabel('Innovation 1')
# ax[1,2].legend(['e1(k)'], fontsize='xx-small')

# ax[2,0].plot(time*10.0, e[1,:], 'b')
# ax[2,0].set_xlabel('k')
# ax[2,0].set_ylabel('Innovation 2')
# ax[2,0].legend(['e2(k)'], fontsize='xx-small')

# ax[2,1].plot(time*10.0, us[0,:], 'b')
# ax[2,1].set_xlabel('k')
# ax[2,1].set_ylabel('us_1(k)')
# ax[2,1].legend(['us_1(k)'], fontsize='xx-small')

# ax[2,2].plot(time*10.0, us[1,:], 'b')
# ax[2,2].set_xlabel('k')
# ax[2,2].set_ylabel('us_2(k)')
# ax[2,2].legend(['us_2(k)'], fontsize='xx-small')

# ax[3,0].plot(time*10.0, X[0,:]-X_hat[0,:], 'g')
# ax[3,0].set_xlabel('k')
# ax[3,0].set_ylabel('Estimation error')
# ax[3,0].legend(['X(k)-X_hat(k)'], fontsize='xx-small')

# ax[3,1].plot(time*10.0, X[1,:]-X_hat[1,:], 'g')
# ax[3,1].set_xlabel('k')
# ax[3,1].set_ylabel('Estimation error in X2')
# ax[3,1].legend(['X2(k)-X2_hat(k)'], fontsize='xx-small')

# ax[3,2].plot(time*10.0, X[2,:]-X_hat[2,:], 'g')
# ax[3,2].set_xlabel('k')
# ax[3,2].set_ylabel('Estimation error in X3')
# ax[3,2].legend(['X3(k)-X3_hat(k)'], fontsize='xx-small')

# ax[4,0].plot(time*10.0, xs[0,:], 'b')
# ax[4,0].set_xlabel('k')
# ax[4,0].set_ylabel('xs_1(k)')
# ax[4,0].legend(['xs_1(k)'], fontsize='xx-small')

# ax[4,1].plot(time*10.0, xs[2,:], 'b')
# ax[4,1].set_xlabel('k')
# ax[4,1].set_ylabel('xs_2(k)')
# ax[4,1].legend(['xs_2(k)'], fontsize='xx-small')

# ax[4,2].plot(time*10.0, xs[2,:], 'b')
# ax[4,2].set_xlabel('k')
# ax[4,2].set_ylabel('xs_3(k)')
# ax[4,2].legend(['xs_3(k)'], fontsize='xx-small')
# plt.show()

# ## Kindly enter full screen and increase workspace area to best view the plots