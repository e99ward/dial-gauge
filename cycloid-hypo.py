import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos, radians

#-----------------------------------#
R = 7 #float(input(" R = "))
r = 2
reps = 3
#-----------------------------------#

circumference = reps * 360
x_points = []
y_points = []

for s in range(0, circumference):
    x = (R - r) * cos(radians(s)) + r * cos(radians(((R - r)/r)*s))
    y = (R - r) * sin(radians(s)) - r * sin(radians(((R - r)/r)*s))
    x_points.append(x)
    y_points.append(y)

raza = plt.Line2D((0, 0), (0, 0), linewidth=1, color="k")
circler = plt.Circle((0, 0), r, color='r', fill=False)
circleR = plt.Circle((0, 0), R, color='k', fill=False)
punct = plt.Circle((0, 0), float(float(r)/10), color="b")
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-1.2*(R + r), 1.2*(R + r))
ax.set_ylim(-1.2*(R + r), 1.2*(R + r))
trace, = ax.plot([], [], color="b")
ax.add_artist(circleR)
ax.add_artist(raza)
ax.add_artist(punct)
ax.add_artist(circler)
ax.add_artist(trace)

def init():
    circler.center = (0, 0)
    punct.center = (0, 0)
    ax.add_patch(circler)
    ax.add_patch(punct)
    return circler, punct,

def animate(i):
    x = (R - r) * np.cos(np.radians(i))
    y = (R - r) * np.sin(np.radians(i))
    x2 = x_points[i]
    y2 = y_points[i]
    raza.set_data((x, x2), (y , y2))
    circler.center = (x, y)
    punct.center = (x2, y2)
    trace.set_data((x_points[:i], y_points[:i]))
    return circler, punct, raza, trace,

anim = FuncAnimation(fig, animate, interval=10, blit=True,
                    init_func=init,
                    frames=circumference)

#plt.grid()
plt.show()