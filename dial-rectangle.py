import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos, tan, atan, radians, degrees, sqrt

#-----------------------------------#
R = 5 #float(input(" R = "))
r = 1
delta_x, delta_y = [0.0, 0.5] # displacement
angle = 10 # rotation
#-----------------------------------#

def get_trace():
    x_pos = []
    y_pos = []
    for s in range(0, circumference):
        lim1 = degrees(atan(R/(R+r)))
        lim2 = degrees(atan((R+r)/R))
        lim3 = 180 - lim2
        lim4 = 180 - lim1
        lim5 = 180 + lim1
        lim6 = 180 + lim2
        lim7 = 180 + lim3
        lim8 = 180 + lim4
        if s >= lim1 and s < lim2:
            x = solve_quadratic_1(s)
            y = x * tan(radians(s))
            x_pos.append(x)
            y_pos.append(y)
        elif s >= lim2 and s < lim3:
            y = R + r
            x = y * tan(radians(90-s))
            x_pos.append(x)
            y_pos.append(y)
        elif s >= lim3 and s < lim4:
            x = solve_quadratic_2(s)
            y = x * tan(radians(s))
            x_pos.append(x)
            y_pos.append(y)
        elif s >= lim4 and s < lim5:
            x = -1 * (R + r)
            y = x * tan(radians(s))
            x_pos.append(x)
            y_pos.append(y)
        elif s >= lim5 and s < lim6:
            x = solve_quadratic_3(s)
            y = x * tan(radians(s))
            x_pos.append(x)
            y_pos.append(y)
        elif s >= lim6 and s < lim7:
            y = -1 * (R + r)
            x = y * tan(radians(90-s))
            x_pos.append(x)
            y_pos.append(y)
        elif s >= lim7 and s < lim8:
            x = solve_quadratic_4(s)
            y = x * tan(radians(s))
            x_pos.append(x)
            y_pos.append(y)
        else:
            x = R + r
            y = x * tan(radians(s))
            x_pos.append(x)
            y_pos.append(y)
    return x_pos, y_pos

def solve_quadratic_1(theta):
    A = (1 + tan(radians(theta))**2)
    B = R * (1 + tan(radians(theta)))
    C = 2*R*R - r*r
    quad = B*B - A*C
    ans = (B + sqrt(quad)) / A
    return ans

def solve_quadratic_2(theta):
    A = (1 + tan(radians(theta))**2)
    B = R * (-1 + tan(radians(theta)))
    C = 2*R*R - r*r
    quad = B*B - A*C
    ans = (B - sqrt(quad)) / A
    return ans

def solve_quadratic_3(theta):
    A = (1 + tan(radians(theta))**2)
    B = R * (-1 - tan(radians(theta)))
    C = 2*R*R - r*r
    quad = B*B - A*C
    ans = (B - sqrt(quad)) / A
    return ans

def solve_quadratic_4(theta):
    A = (1 + tan(radians(theta))**2)
    B = R * (1 - tan(radians(theta)))
    C = 2*R*R - r*r
    quad = B*B - A*C
    ans = (B + sqrt(quad)) / A
    return ans

def make_rotate(x_pos, y_pos, angle):
    x_new = []
    y_new = []
    for s in range(0, circumference):
        x = x_pos[s] * cos(radians(angle)) - y_pos[s] * sin(radians(angle))
        y = x_pos[s] * sin(radians(angle)) + y_pos[s] * cos(radians(angle))
        x_new.append(x)
        y_new.append(y)
    return x_new, y_new

def make_parallel_move(x_pos, y_pos, delta_x, delta_y):
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
x_points, y_points = make_rotate(x_points, y_points, angle)
x_points, y_points = make_parallel_move(x_points, y_points, delta_x, delta_y)
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
circler, = ax1.plot([], [], 'g-', linewidth=1)
probe, = ax2.plot([], [], 'r-')

def rect_drawing(delta_x, delta_y, angle, R):
    x_rect = [R,-R,-R,R,R]
    y_rect = [R,R,-R,-R,R]
    x_new = []
    y_new = []
    for s in np.arange(len(x_rect)):
        x = x_rect[s] * cos(radians(angle)) - y_rect[s] * sin(radians(angle))
        y = x_rect[s] * sin(radians(angle)) + y_rect[s] * cos(radians(angle))
        x_new.append(x)
        y_new.append(y)
    x_pos = [x + delta_x for x in x_new]
    y_pos = [y + delta_y for y in y_new]
    line = plt.Line2D(x_pos, y_pos, color='k', linewidth=2)
    return line

line_org1 = plt.Line2D([-1,1], [0,0], color='k', linewidth=0.5)
line_org2 = plt.Line2D([0,0], [-1,1], color='k', linewidth=0.5)
line_org3 = rect_drawing(delta_x, delta_y, angle, R)
ax1.add_artist(line_org1)
ax1.add_artist(line_org2)
ax1.add_artist(line_org3)

def animate(i):
    x = x_points[i]
    y = y_points[i]
    cx, cy = circle_drawing(x, y, r)
    circler.set_data(cx, cy)
    trace.set_data((x_points[:i], y_points[:i]))
    probe.set_data((np.arange(i), distance[:i]))
    return circler, trace, probe,

def circle_drawing(c_x, c_y, radius):
    theta = np.linspace(0, 2*np.pi, 100)
    x = c_x + radius * np.cos(theta)
    y = c_y + radius * np.sin(theta)
    return x, y

anim = FuncAnimation(fig, animate, interval=10, blit=True, repeat=False,
                    frames=circumference)

#plt.grid()
plt.show()