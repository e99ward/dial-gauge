import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos, radians

#-----------------------------------#
R = 1 # radius of circle
#-----------------------------------#

instep = 200
x_points = []
y_points = []

for s in np.linspace(0, 4*np.pi, instep):
    x = R * (s - sin(s))
    y = R - R * cos(s)
    x_points.append(x)
    y_points.append(y)

raza = plt.Line2D((0, 0), (0, 0), linewidth=1, color="k")
circleR = plt.Circle((0, 0), R, color='r', fill=False)
punct = plt.Circle((0, 0), float(float(R)/10), color="b")
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_ylim(0, 3)
ax.set_xlim(0, 15)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()
trace, = ax.plot([], [], color="b")
ax.add_artist(circleR)
ax.add_artist(raza)
ax.add_artist(punct)
ax.add_artist(trace)
time_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)

def init():
    circleR.center = (0, 0)
    punct.center = (0, 0)
    ax.add_patch(circleR)
    ax.add_patch(punct)
    return circleR, punct,

def animate(i):
    theta = i / instep * (4 * np.pi) 
    x = R * theta
    y = R
    x2 = x_points[i]
    y2 = y_points[i]
    time_text.set_text(r'$\theta$ = %.2f $\pi$' % (theta / np.pi))
    raza.set_data((x, x2), (y , y2))
    circleR.center = (x, y)
    punct.center = (x2, y2)
    trace.set_data((x_points[:i], y_points[:i]))
    return circleR, punct, raza, trace, time_text,

ani = FuncAnimation(fig, animate, interval=10, blit=True,
                    init_func=init,
                    frames=instep)

plt.show()