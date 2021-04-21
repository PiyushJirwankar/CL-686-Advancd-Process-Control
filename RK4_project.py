# Name: Piyush Jirwankar
# Roll. no.: 170010003
# Assignment number 1

### RK4 to integrate y_dot = f{t,y} ; y(t0) = y0
### y is of dimension nx1
import numpy as np
import matplotlib.pyplot as plt

def RK4(f, y0, t0, tf, h):
    N = int((tf-t0)/h)
    time = np.linspace(t0, tf, N)
    for i in range(N):
        k1 = h*f(time[i], y0)
        k2 = h*f(time[i]+h/2.0, y0+h*k1/2.0)
        k3 = h*f(time[i]+h/2.0, y0+h*k2/2.0)
        k4 = h*f(time[i]+h, y0+h*k3)
        y0 = y0 + (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
    return y0
