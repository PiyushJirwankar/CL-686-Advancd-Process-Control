import numpy as np
import random
from RK4_project import RK4 as RK4
import matplotlib.pyplot as plt

np.random.seed(0)

start = 0
end = 25
N = 10*(end-start)+1
time  = np.linspace(start, end, N)
h  = time[1]-time[0]

Us = np.array([1.0, 390.0])
# Ds = 1.0
Xs = np.array([0.400948649, 392.004743, 0.16])
Ys = np.array([392.004743, 0.16])

C = np.array([[0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])


X = np.zeros((3,N))
X[:,0] = Xs

U = np.zeros((2,N))
# U[:,0] = Us

Y = np.zeros((2,N))
Y[:,0] = Ys

alph1 = 300000.0
alph2 = 60000000.0
alph3 = 0.16
alph4 = 5.0
alph5 = 0.4

Cpi = np.array([[12.5/2.0, 0.0],
                [0.0, 2/0.475]])
# Cpi = np.array([[2/0.475, 0.0],
#                 [0.0, 12.5/2.0]])

Dpi = np.array([[12.5, 0.0],
                [0.0, 2.0]])
# Dpi = np.array([[2.0, 0.0],
#                 [0.0, 12.5]])


### Servo control begins ===>  Uncomment this and comment regulatory control to run servo control
# r = np.zeros((2,N))
# for i in range(4,N):
#     r[:,i] = np.array([10.0, -0.025])
# D = Ds*np.ones(N)
### Servo control ends


### Regulatory control begins ===>  ncomment this and comment servo control to run regulatory control
r = np.zeros((2,N))
R_track = np.zeros((2,N))
Ys = np.dot(C, Xs)

for i in range(60):
    R_track[:,i] = np.dot(C, Xs)

for i in range(60, 120):
    r[:,i] = np.array([10.0, -0.025])
    R_track[:,i] = r[:,i] + Ys
for i in range(120,N):
    R_track[:,i] = np.dot(C, Xs)



# r = np.zeros((2,N))
# D = Ds*np.ones(N)
Ds = np.ones(N)
D = np.zeros(N)
for i in range(180, N):
    Ds[i] = 1.5
### Regulatory control ends

Ul = np.array([0.0, 350.0])
Uh = np.array([2.0, 430.0])
diffU1 = 1.0
diffU2 = 40.0

nk = np.zeros(2)
e = np.zeros((2,N))

dk_sigma = 0.015
meas_sigma = np.array([0.2, 0.0035])
R = np.array([[meas_sigma[0]**2, 0.0],[0.0, meas_sigma[1]**2]])

sse1_asgn2 = 0.0
sse2_asgn2 = 0.0
ssmv1_asgn2 = 0.0
ssmv2_asgn2 = 0.0

sse1_asgn2 = (Y[0,0] - R[0,0])**2
sse1_asgn2 = (Y[1,0] - R[1,0])**2

for i in range(N-1):
    yk = Y[:,i] - Ys
    rk = r[:,i]
    D[i] = Ds[i] + np.random.normal(0.0, dk_sigma)
    var = rk - yk
    intvar1 = var[1]
    intvar2 = var[0]
    ek = np.array([intvar1, intvar2])
    e[:,i] = ek
    uk = np.dot(Cpi,nk) + np.dot(Dpi,ek)
    uk[0] = np.sign(uk[0])*(np.minimum(diffU1, np.absolute(uk[0]))) # Projecting U1 onto feasible range
    uk[1] = np.sign(uk[1])*(np.minimum(diffU2, np.absolute(uk[1]))) # Projecting U2 onto feasible range
    nk = nk + 0.1*ek
    U[:,i] = uk + Us
    ssmv1_asgn2 = ssmv1_asgn2 + (U[0,i]-Us[0])**2
    ssmv2_asgn2 = ssmv2_asgn2 + (U[1,i]-Us[1])**2
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
    Y[:,i+1] = np.dot(C, X[:,i+1]) + np.random.multivariate_normal(np.zeros(2), R)
    sse1_asgn2 = (Y[0,i+1] - R_track[0,i+1])**2
    sse1_asgn2 = (Y[1,i+1] - R_track[1,i+1])**2

D[N-1] = Ds[N-1] + np.random.normal(0.0, dk_sigma)
U = U[:,:N-1]
y = np.zeros((2,N))
# R_track = np.zeros((2,N))
for i in range(N):
    y[:,i] = Y[:,i]-Ys
    # R_track[:,i] = Ys + r[:,i]

np.savetxt('Y1_asgn2.csv', Y[0,:], delimiter=',')
np.savetxt('Y2_asgn2.csv', Y[1,:], delimiter=',')
np.savetxt('U1_asgn2.csv', U[0,:], delimiter=',')
np.savetxt('U2_asgn2.csv', U[1,:], delimiter=',')
np.savetxt('sse1_asgn2.csv', sse1_asgn2)
np.savetxt('sse2_asgn2.csv', sse2_asgn2)
np.savetxt('ssmv1_asgn2.csv', ssmv1_asgn2)
np.savetxt('ssmv1_asgn2.csv', ssmv2_asgn2)



# fig, ax = plt.subplots(4,2)
# ax[0,0].plot(time, X[0,:],'b')
# ax[0,0].set_xlabel('Time (min)')
# ax[0,0].set_ylabel('X1(k)')
# ax[0,0].legend(['X1(k)'], fontsize='xx-small')

# ax[0,1].plot(time, X[1,:],'b')
# ax[0,1].set_xlabel('Time (min)')
# ax[0,1].set_ylabel('X2(k)')
# ax[0,1].legend(['X2(k)'], fontsize='xx-small')

# ax[1,0].plot(time, X[2,:],'b')
# ax[1,0].set_xlabel('Time')
# ax[1,0].set_ylabel('X3 (min)(k)')
# ax[1,0].legend(['X3(k)'], fontsize='xx-small')

# ax[1,1].plot(time, Y[0,:], 'g')
# ax[1,1].plot(time, R_track[0,:], 'r')
# ax[1,1].set_xlabel('Time (min)')
# ax[1,1].set_ylabel('Y1(k) & R1(k)')
# ax[1,1].legend(['Y1(k)', 'R1(k)'], fontsize='xx-small')

# ax[2,0].plot(time, Y[1,:], 'g')
# ax[2,0].plot(time, R_track[1,:], 'r')
# ax[2,0].set_xlabel('Time (min)')
# ax[2,0].set_ylabel('Y2(k) & R2(k)')
# ax[2,0].legend(['Y2(k)', 'R2(k)'], fontsize='xx-small')

# ax[2,1].plot(time[:N-1], U[0,:],'r', drawstyle = 'steps-pre')
# ax[2,1].set_xlabel('Time (min)')
# ax[2,1].set_ylabel('U1(k)')
# ax[2,1].legend(['U1(k)'], fontsize='xx-small')

# ax[3,0].plot(time[:N-1], U[1,:],'r', drawstyle = 'steps-pre')
# ax[3,0].set_xlabel('Time (min)')
# ax[3,0].set_ylabel('U2(k)')
# ax[3,0].legend(['U2(k)'], fontsize='xx-small')

# ax[3,1].plot(time, D,'c')
# ax[3,1].set_xlabel('Time (min)')
# ax[3,1].set_ylabel('D(k)')
# ax[3,1].legend(['D(k)'], fontsize='xx-small')

# # fig.suptitle('Servo control - required plots')
# fig.suptitle('Regulatory control - required plots')

# plt.show()

# ## Kindly enter full screen to best view the plots


