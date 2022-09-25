import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos, radians

#-----------------------------------#
R = 1 # radius of circle
#-----------------------------------#
# plt.rcParams["figure.figsize"] = 4,3

fig, ax = plt.subplots(figsize=(6,3))
# fig = plt.figure(figsize=(6,3))
# ax = fig.add_subplot(111)
ax.set_ylim(0, 3)
ax.set_xlim(0, 14)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.grid()

raza = plt.Line2D((0, 0), (0, 0), linewidth=1, color="y")
circleR = plt.Circle((0, 0), R, color='g', fill=False)
punct = plt.Circle((0, 0), float(float(R)/10), color="b")
trace, = ax.plot([], [], color="b")
ax.add_artist(circleR)
ax.add_artist(raza)
ax.add_artist(punct)
ax.add_artist(trace)
time_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)

xx, yy = [], []

def animate(data):
    x, y, Rt = data
    time_text.set_text(r'$\theta$ = %.2f $\pi$' % (Rt/np.pi))
    xx.append(x)
    yy.append(y)
    trace.set_data(xx, yy)
    raza.set_data((x, Rt), (y , R))
    circleR.center = (Rt, R)
    punct.center = (x, y)
    #point.set_data(x, y)

def generate():
    for theta in np.linspace(0, 4*np.pi, 200):
        yield R*(theta-sin(theta)), R*(1-cos(theta)), R*theta

ani = FuncAnimation(fig, animate, generate, blit=False, interval=30)

plt.show()