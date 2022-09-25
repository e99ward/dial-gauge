import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos, radians, sqrt

#-----------------------------------#
R = 5 #float(input(" R = "))
r = 1
delta_x, delta_y = [1.0, 0.5] # displacement
#-----------------------------------#

def get_trace():
    x_pos = []
    y_pos = []
    for s in range(0, circumference):
        # cycloid
        # x = (R + r) * cos(radians(s)) - r * cos(radians(((R + r)/r)*s))
        # y = (R + r) * sin(radians(s)) - r * sin(radians(((R + r)/r)*s))
        x = (R + r) * cos(radians(s))
        y = (R + r) * sin(radians(s))
        x_pos.append(x)
        y_pos.append(y)
    return x_pos, y_pos

def make_parallel_move(x_pos, y_pos):
    x_new = [x + delta_x for x in x_pos]
    y_new = [y + delta_y for y in y_pos]
    return x_new, y_new

def get_probe(x_pos, y_pos):
    dist = []
    for s in range(0, circumference):
        db = (x_pos[s])**2 + (y_pos[s])**2
        dist.append(sqrt(db) - r)
    return dist

circumference = 360
x_points, y_points = get_trace()
x_points, y_points = make_parallel_move(x_points, y_points)
distance = get_probe(x_points, y_points) 

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.set_aspect("equal")
ax1.set_xlim(-1.5*(R + r), 1.5*(R + r))
ax1.set_ylim(-1.5*(R + r), 1.5*(R + r))
ax1.set_title('Dial Gauge')
ax2.set_xlim(0, 360)
ax2.set_ylim(0, 10)
ax2.set_title('Gauge Distance')
ax2.set_xlabel('Angle, deg')
ax2.set_xticks(45*np.arange(8))
ax2.grid()

trace, = ax1.plot([], [], 'b-')
circleR, = ax1.plot([], [], 'k-')
circler, = ax1.plot([], [], 'g-', linewidth=1)
wand, = ax1.plot([], [], '.y-', linewidth=1)
probe, = ax2.plot([], [], 'r-')

line_org1 = plt.Line2D([-1,1], [0,0], color='k', linewidth=0.5)
line_org2 = plt.Line2D([0,0], [-1,1], color='k', linewidth=0.5)
ax1.add_artist(line_org1)
ax1.add_artist(line_org2)

def init():
    cx, cy = circle_drawing(delta_x, delta_y, R)
    circleR.set_data(cx, cy)
    return circleR,

def animate(i):
    x = x_points[i]
    y = y_points[i]
    phi = radians(i) * R / r
    x2 = x - r * cos(phi)
    y2 = y - r * sin(phi)
    cx, cy = circle_drawing(x, y, r)
    circler.set_data(cx, cy)
    wand.set_data((x, x2), (y, y2))
    trace.set_data((x_points[:i], y_points[:i]))
    probe.set_data((np.arange(i), distance[:i]))
    return circler, circleR, trace, wand, probe,

def circle_drawing(c_x, c_y, radius):
    theta = np.linspace(0, 2*np.pi, 100)
    x = c_x + radius * np.cos(theta)
    y = c_y + radius * np.sin(theta)
    return x, y

anim = FuncAnimation(fig, animate, interval=10, blit=True, repeat=False,
                    init_func=init,
                    frames=circumference)

#plt.grid()
plt.show()