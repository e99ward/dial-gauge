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

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_ylim(0, 3)
ax.set_xlim(0, 15)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()

trace, = ax.plot([], [], 'r-', lw=2)
circle_line, = ax.plot([], [], 'g', lw=1)
line, = ax.plot([], [], '.-y', lw=1)
#point, = ax.plot([], [], 'go', ms=4)
time_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)

def animate(i):
    theta = i / instep * (4 * np.pi) 
    x = R * theta
    y = R
    x2 = x_points[i]
    y2 = y_points[i]
    time_text.set_text(r'$\theta$ = %.2f $\pi$' % (theta / np.pi))
    cx, cy = circle(x, y, R)
    circle_line.set_data(cx, cy)
    line.set_data((x, x2), (y, y2))
    trace.set_data((x_points[:i], y_points[:i]))
    return circle_line, line, trace, time_text,

def circle(a, b, r):
    # center at (a,b) with radius r
    theta = np.linspace(0, 2*np.pi, 100)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)
    return x, y

ani = FuncAnimation(fig, animate, interval=10, blit=True, frames=instep)

plt.show()