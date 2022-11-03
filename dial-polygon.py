import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import degrees, radians, sin, cos, tan, sqrt, atan
from SegTR import Rotate, ParallelMove

#-----------------------------------#
r = 2 # radius of gauge ball
T1 = 0.01 # tolerance for adjust
T2 = 0.01 # tolerance for acceptance
delta_x, delta_y = [1.1, 0.5] # displacement
angle = 10 # rotation
#-----------------------------------#

A = 0.0
B = 0.0
C = 0.0
circumference = 360
seg_org = np.array([[5,0],[5,2.5],[5.6,5],[5,5.6],[2.5,5],[0,5],[-2.5,5],[-5,5.6],[-5.6,5],[-5,2.5],[-5,0],
                    [-5,-2.5],[-5.6,-5],[-5,-5.6],[-2.5,-5],[0,-5],[2.5,-5],[5,-5.6],[5.6,-5],[5,-2.5],[5,0]])
# seg_org = np.array([[5,0],[5,2.5],[5,5],[2.5,5],[0,5],[-2.5,5],[-5,5],[-5,2.5],[-5,0],
#                     [-5,-2.5],[-5,-5],[-2.5,-5],[0,-5],[2.5,-5],[5,-5],[5,-2.5],[5,0]])
seg_rot = Rotate(seg_org, angle)
segments = ParallelMove(seg_rot, delta_x, delta_y)

angles = []
trace_x = []
trace_y = []
gauge_x = []
gauge_y = []
distance = []

def get_segment(idx):
    pos1 = segments[idx,:]
    pos2 = segments[idx+1,:]
    return pos1, pos2

def set_segment_constants(pt_1, pt_2):
    x1, y1 = pt_1
    x2, y2 = pt_2
    global A
    A = y1 - y2
    global B
    B = x2 - x1
    global C
    C = x1*y2 - x2*y1

def get_angle_of_point(pt):
    if pt[0] == 0:
        if pt[1] > 0:
            angle = 90
        else:
            angle = 270
    elif pt[0] > 0:
        # region 1 or 4
        angle = degrees(atan(pt[1] / pt[0]))
        if angle < 0:
            angle += 360
    else:
        # region 2 or 3
        angle = degrees(atan(pt[1] / pt[0]))
        angle += 180
    return angle

def get_foot_position(pt): #from pt to line
    x0, y0 = pt
    x_h = (B*B*x0 - A*B*y0 - A*C) / (A**2 + B**2)
    y_h = (A*A*y0 - A*B*x0 - B*C) / (A**2 + B**2)
    pt_h = [x_h, y_h]
    return pt_h

def get_foot_position_angle(pt):
    pt_h = get_foot_position(pt)
    angle = get_angle_of_point(pt_h)
    return angle

def get_foot_position_angle_explicit_ABC(pt, A, B, C):
    x0, y0 = pt
    x_h = (B*B*x0 - A*B*y0 - A*C) / (A**2 + B**2)
    y_h = (A*A*y0 - A*B*x0 - B*C) / (A**2 + B**2)
    pt_h = [x_h, y_h]
    angle = get_angle_of_point(pt_h)
    return angle

def get_distance(pt): #from pt to line
    x0, y0 = pt
    frac_up = A*x0 + B*y0 + C
    frac_dn = A**2 + B**2
    dist = abs(frac_up) / sqrt(frac_dn)
    return dist

def get_distance_by_points(pt_1, pt_2, pt_c): # from pt_c to line pt_1 - pt_2
    x0, y0 = pt_c
    x1, y1 = pt_1
    x2, y2 = pt_2
    frac_up = (x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)
    frac_dn = (x2-x1)**2 + (y2-y1)**2
    dist = abs(frac_up) / sqrt(frac_dn)
    return dist

def get_distance_btw_points(pt_1, pt_2):
    x1, y1 = pt_1
    x2, y2 = pt_2
    sqrt_in = (x2-x1)**2 + (y2-y1)**2
    return sqrt(sqrt_in)

def assert_angle_at_x_region(theta, angle):
    if theta < 45:
        if angle > 315:
            angle -= 360
    elif theta > 315:
        if angle < 45:
            angle += 360
    return angle

def judgment_of_touch_with_neighbor(pt_in, theta, index):
    if index < 0:
        pt_1 = segments[-2,:]
        pt_2 = segments[-1,:]
        idx_lw = len(angles)-2
        idx_up = len(angles)-1
    elif index >= len(angles)-1:
        pt_1 = segments[0,:]
        pt_2 = segments[1,:]
        idx_lw = 0
        idx_up = 1
    else:
        pt_1, pt_2 = get_segment(index)
        idx_lw = index
        idx_up = index+1
    x1, y1 = pt_1
    x2, y2 = pt_2
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    pt_x, pt_y = pt_in
    dist = get_distance_by_points(pt_1, pt_2, pt_in)
    ifcusp = False
    while(True):
        if (dist > r-T2 and dist < r+T2):
            angle = get_foot_position_angle_explicit_ABC([pt_x, pt_y], A, B, C)
            if angle > angles[idx_up] or angle < angles[idx_lw]:
                ifcusp = True
            break
        elif (dist < r):
            if theta >= 45 and theta < 135:
                pt_x += T1*tan(radians(90-theta))
                pt_y += T1
            elif theta >= 135 and theta < 225:
                pt_x -= T1
                pt_y -= T1*tan(radians(theta))
            elif theta >= 225 and theta < 315: 
                pt_x -= T1*tan(radians(90-theta))
                pt_y -= T1
            else:
                pt_x += T1
                pt_y += T1*tan(radians(theta))
            dist = get_distance_by_points(pt_1, pt_2, [pt_x, pt_y])
        else:
            if theta >= 45 and theta < 135:
                pt_x -= T1*tan(radians(90-theta))
                pt_y -= T1
            elif theta >= 135 and theta < 225:
                pt_x += T1
                pt_y += T1*tan(radians(theta))
            elif theta >= 225 and theta < 315: 
                pt_x += T1*tan(radians(90-theta))
                pt_y += T1
            else:
                pt_x -= T1
                pt_y -= T1*tan(radians(theta))
            dist = get_distance_by_points(pt_1, pt_2, [pt_x, pt_y])
    pt_out = [pt_x, pt_y]
    return pt_out, ifcusp

def judgment_of_touch_x(pt, theta, index):
    pt_1, pt_2 = get_segment(index)
    pt_c = [pt[0], pt[1]]
    dist = get_distance(pt_c)
    ifcusp = False
    while(True):
        # 0-45 < angle < 0+45
        if (dist < r-T2):
            pt_c[0] += T1
            pt_c[1] += T1*tan(radians(theta))
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            angle = assert_angle_at_x_region(theta, angle)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x(pt_c, pt_1, theta)
                break
        elif (dist > r+T2):
            pt_c[0] -= T1
            pt_c[1] -= T1*tan(radians(theta))
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            angle = assert_angle_at_x_region(theta, angle)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x(pt_c, pt_1, theta)
                break
        else: # if (dist > r-T2 and dist < r+T2):
            angle = get_foot_position_angle(pt_c)
            angle = assert_angle_at_x_region(theta, angle)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x(pt_c, pt_1, theta)
                break
            break
    return pt_c

def judgment_of_touch_y(pt, theta, index):
    pt_1, pt_2 = get_segment(index)
    pt_c = [pt[0], pt[1]]
    dist = get_distance(pt_c)
    ifcusp = False
    while(True):
        # 90-45 < angle < 90+45
        if (dist < r-T2):
            pt_c[0] += T1*tan(radians(90-theta))
            pt_c[1] += T1
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_1, theta)
                break
        elif (dist > r+T2):
            pt_c[0] -= T1*tan(radians(90-theta))
            pt_c[1] -= T1
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_1, theta)
                break
        else: # (dist > r-T2 and dist < r+T2):
            angle = get_foot_position_angle(pt_c)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_1, theta)
                break
            break
    return pt_c

def judgment_of_touch_x_neg(pt, theta, index):
    pt_1, pt_2 = get_segment(index)
    pt_c = [pt[0], pt[1]]
    dist = get_distance(pt_c)
    ifcusp = True
    while(True):
        # 180-45 < angle < 180+45
        if (dist < r-T2):
            pt_c[0] -= T1
            pt_c[1] -= T1*tan(radians(theta))
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x_neg(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x_neg(pt_c, pt_1, theta)
                break
        elif (dist > r+T2):
            pt_c[0] += T1
            pt_c[1] += T1*tan(radians(theta))
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x_neg(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x_neg(pt_c, pt_1, theta)
                break
        else: # if (dist > r-T2 and dist < r+T2):
            angle = get_foot_position_angle(pt_c)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x_neg(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_x_neg(pt_c, pt_1, theta)
                break
            break
    return pt_c

def judgment_of_touch_y_neg(pt, theta, index):
    pt_1, pt_2 = get_segment(index)
    pt_c = [pt[0], pt[1]]
    dist = get_distance(pt_c)
    ifcusp = 0
    while(True):
        # 270-45 < angle < 270+45
        if (dist < r-T2):
            pt_c[0] -= T1*tan(radians(90-theta))
            pt_c[1] -= T1
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y_neg(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y_neg(pt_c, pt_1, theta)
                break
        elif (dist > r+T2):
            pt_c[0] += T1*tan(radians(90-theta))
            pt_c[1] += T1
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y_neg(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y_neg(pt_c, pt_1, theta)
                break
        else: # if (dist > r-T2 and dist < r+T2):
            angle = get_foot_position_angle(pt_c)
            if angle > angles[index+1]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y_neg(pt_c, pt_2, theta)
                break
            elif angle < angles[index]:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, theta, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y_neg(pt_c, pt_1, theta)
                break
            break
    return pt_c

def adjust_gauge_to_cusp_x(pt_in, pt_s, theta):
    pt_c = [pt_in[0], pt_in[1]]
    dist = get_distance_btw_points(pt_c, pt_s)
    while(True):
        if (dist > r-T2 and dist < r+T2):
            break
        elif (dist < r):
            pt_c[0] += T1
            pt_c[1] += T1*tan(radians(theta))
            dist = get_distance_btw_points(pt_c, pt_s)
        else:
            pt_c[0] -= T1
            pt_c[1] -= T1*tan(radians(theta))
            dist = get_distance_btw_points(pt_c, pt_s)
    return pt_c

def adjust_gauge_to_cusp_y(pt_in, pt_s, theta):
    pt_c = [pt_in[0], pt_in[1]]
    dist = get_distance_btw_points(pt_c, pt_s)
    while(True):
        if (dist > r-T2 and dist < r+T2):
            break
        elif (dist < r):
            pt_c[0] += T1*tan(radians(90-theta))
            pt_c[1] += T1
            dist = get_distance_btw_points(pt_c, pt_s)
        else:
            pt_c[0] -= T1*tan(radians(90-theta))
            pt_c[1] -= T1
            dist = get_distance_btw_points(pt_c, pt_s)
    return pt_c

def adjust_gauge_to_cusp_x_neg(pt_in, pt_s, theta):
    pt_c = [pt_in[0], pt_in[1]]
    dist = get_distance_btw_points(pt_c, pt_s)
    while(True):
        if (dist > r-T2 and dist < r+T2):
            break
        elif (dist < r):
            pt_c[0] -= T1
            pt_c[1] -= T1*tan(radians(theta))
            dist = get_distance_btw_points(pt_c, pt_s)
        else:
            pt_c[0] += T1
            pt_c[1] += T1*tan(radians(theta))
            dist = get_distance_btw_points(pt_c, pt_s)
    return pt_c

def adjust_gauge_to_cusp_y_neg(pt_in, pt_s, theta):
    pt_c = [pt_in[0], pt_in[1]]
    dist = get_distance_btw_points(pt_c, pt_s)
    while(True):
        if (dist > r-T2 and dist < r+T2):
            break
        elif (dist < r):
            pt_c[0] -= T1*tan(radians(90-theta))
            pt_c[1] -= T1
            dist = get_distance_btw_points(pt_c, pt_s)
        else:
            pt_c[0] += T1*tan(radians(90-theta))
            pt_c[1] += T1
            dist = get_distance_btw_points(pt_c, pt_s)
    return pt_c

def build_up_data(pt_c):
    global trace_x
    trace_x.append(pt_c[0])
    global trace_y
    trace_y.append(pt_c[1])

def do_region_positive_x(index): # 1x (0-45) = 4x (315-360)
    pt_1, pt_2 = get_segment(index)
    set_segment_constants(pt_1, pt_2)
    pt_c = [pt_1[0], pt_1[1]]
    for theta in range(angles[index], angles[index+1]):
        pt_c[1] = pt_c[0] * tan(radians(theta))
        pt_c = judgment_of_touch_x(pt_c, theta, index)
        build_up_data(pt_c)

def do_region_positive_y(index): # 1y (45-90) = 2y (90-135)
    pt_1, pt_2 = get_segment(index)
    set_segment_constants(pt_1, pt_2)
    pt_c = [pt_1[0], pt_1[1]]
    for theta in range(angles[index], angles[index+1]):
        pt_c[0] = pt_c[1] * tan(radians(90-theta))
        pt_c = judgment_of_touch_y(pt_c, theta, index)
        build_up_data(pt_c)

def do_region_negative_x(index): # 2x (135-180) = 3x (180-225)
    pt_1, pt_2 = get_segment(index)
    set_segment_constants(pt_1, pt_2)
    pt_c = [pt_1[0], pt_1[1]]
    for theta in range(angles[index], angles[index+1]):
        pt_c[1] = pt_c[0] * tan(radians(theta))
        pt_c = judgment_of_touch_x_neg(pt_c, theta, index)
        build_up_data(pt_c)

def do_region_negative_y(index): # 3y (225-270) = 4y (270-315)
    pt_1, pt_2 = get_segment(index)
    set_segment_constants(pt_1, pt_2)
    pt_c = [pt_1[0], pt_1[1]]
    for theta in range(angles[index], angles[index+1]):
        pt_c[0] = pt_c[1] * tan(radians(90-theta))
        pt_c = judgment_of_touch_y_neg(pt_c, theta, index)
        build_up_data(pt_c)

# set the angles for the segments
for item in segments:
    angle = get_angle_of_point(item) # [item[0], item[1]]
    if angles and angle < angles[-1]:
        angle += 360
    angles.append(int(angle))
print(segments)
print(angles)

# classify the regions
for idx, angle in enumerate(angles[:-1]):
    if 45 <= angle < 135: # if angle >= 45 and angle < 135:
        do_region_positive_y(idx)
    elif 135 <= angle < 225:
        do_region_negative_x(idx)
    elif 225 <= angle < 315: 
        do_region_negative_y(idx)
    else:
        do_region_positive_x(idx)

# print(len(gauge_x))
# print(len(trace_x))

for i in range(len(trace_x)-1):
    dist = get_distance_btw_points([trace_x[i], trace_y[i]], [0, 0])
    distance.append(dist - r)
distance.append(distance[0])

# plot

fig0, ax0 = plt.subplots()
ax0.set_xlim(0, 360)
ax0.set_ylim(0, 10)
ax0.set_xticks(30*np.arange(12))
probefixed, = ax0.plot(np.arange(circumference), distance, 'r-')
ax0.grid()

fig1, (ax1, ax2) = plt.subplots(ncols=2)
ax1.set_aspect("equal")
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.set_title('Dial Gauge')
ax2.set_xlim(0, 360)
ax2.set_ylim(0, 10)
ax2.set_title('Gauge Distance')
ax2.set_xlabel('Angle, deg')
ax2.set_xticks(45*np.arange(8))
ax2.grid()

trace, = ax1.plot([], [], 'b-')
circle, = ax1.plot([], [], 'g-', linewidth=1)
punct, = ax1.plot([], [], '.k-', linewidth=1)
probe, = ax2.plot([], [], 'r-')

line_seg = plt.Line2D(segments[:,0], segments[:,1], color='k')
ax1.add_artist(line_seg)
line_org1 = plt.Line2D([-1,1], [0,0], color='k', linewidth=0.5)
line_org2 = plt.Line2D([0,0], [-1,1], color='k', linewidth=0.5)
ax1.add_artist(line_org1)
ax1.add_artist(line_org2)

def animate(i):
    x = trace_x[i]
    y = trace_y[i]
    cx, cy = circle_drawing(x, y, r)
    circle.set_data(cx, cy)
    trace.set_data((trace_x[:i], trace_y[:i]))
    punct.set_data((delta_x, delta_y))
    probe.set_data((np.arange(i), distance[:i]))
    return circle, trace, punct, probe,

def circle_drawing(c_x, c_y, radius):
    theta = np.linspace(0, 2*np.pi, 100)
    x = c_x + radius * np.cos(theta)
    y = c_y + radius * np.sin(theta)
    return x, y

anim = FuncAnimation(fig1, animate, interval=20, blit=True, frames=circumference)

plt.show()