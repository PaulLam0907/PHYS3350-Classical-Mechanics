"""
PHYS3350 Classical Mechanics HW2 Q5
Name : Lam Siu Ping
UID : 3036 219 487

Van der Pol Oscillator
x'' + μ(x^2 - a^2)x' + ω_0^2 x = 0

let v = x'  ----(1)
v' + μ(x^2 - a^2)v + ω_0^2 x = 0  ----(2)

linear damping strength μa^2
non-linear damping strength μ
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

########## Constants ##########
OMEGA_0 = 1  # eigenfrequency
a = 2
MU = [0.05, 1, 2, 5]
###############################

# initial condition
x0 = [1, 0]  # [x(0), x'(0)]

# range where the output is evaluated
t = np.arange(0, 60, 0.1)


def dxdt(x, t, _omega_0, _a, _mu):
    """
    Function for odeint to solve the ODE
    For custom parameters, specify args = (_omega_0, _a, _mu) in the odeint()
    
    :param x:
    :param t:
    :param _omega_0:
    :param _a:
    :param _mu:
    :return:
    """
    x, v = x
    
    dot_x = v  # ----(1)
    dot_v = -_mu * (x**2 - _a**2) * v - x * _omega_0**2  # ----(2)
    
    return [dot_x, dot_v]


# solve the equation
def solve_ode(_mu):
    """
    Function for getting the solution x(t), x'(t) with custom value of μ
    
    :param _mu:
    :return: tuple of ( x(t) , x'(t) )
    """
    x, v = odeint(dxdt, x0, t, args = (OMEGA_0, a, _mu)).T  # .T = transpose
    
    return x, v


# plot the result
fig = plt.figure("Van der Pol Oscillator")  # the figure window
fig.suptitle(f"Solution to Van der Pol Oscillator at Different Value of $\mu$\n(a = {a}, $\omega_0$ = {OMEGA_0})")  # title for whole figure
# fig.tight_layout()  # auto adjust axes margin

for i in range(len(MU)):
    # solving for x(t) with custom value of μ
    x, v = solve_ode(MU[i])
    
    # individual graphs
    axes = fig.add_subplot(2, 2, i + 1)  # row, col, n = 1 at top left
    axes.plot(t, x)
    axes.set_xlabel("$t$")
    axes.set_ylabel("$x(t)$")
    axes.set_title(f"$\mu$ = {MU[i]}")  # axes title
    axes.grid()  # enable major grid line for both axis

# x, v = solve(MU[0])
# plt.plot(t, x, label = "$x(t)$")
# plt.xlabel("$t$")
# plt.ylabel("$x(t)$")
# plt.title("Title")  # graph title
# plt.legend()  # enable legend
# plt.grid(color = "lightgray", linestyle = "--")  # enable grid
plt.show()
