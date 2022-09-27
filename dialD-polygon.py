import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import degrees, radians, sin, cos, tan, sqrt, atan
from SegTRD import Rotate, ParallelMove, RotatePoint

#-----------------------------------#
r = 2 # radius of gauge ball
T1 = 0.01 # tolerance for adjust
T2 = 0.01 # tolerance for acceptance
delta_x, delta_y = [1.1, 0.5] # displacement
#angle = 10 # rotation
#-----------------------------------#

A = 0.0
B = 0.0
C = 0.0
circumference = 360
seg_org = np.array([[5,0],[5,2.5],[5.6,5],[5,5.6],[2.5,5],[0,5],[-2.5,5],[-5,5.6],[-5.6,5],[-5,2.5],[-5,0],
                  [-5,-2.5],[-5.6,-5],[-5,-5.6],[-2.5,-5],[0,-5],[2.5,-5],[5,-5.6],[5.6,-5],[5,-2.5],[5,0]])
# seg_org = np.array([[5,0],[5,2.5],[5,5],[2.5,5],[0,5],[-2.5,5],[-5,5],[-5,2.5],[-5,0],
#                     [-5,-2.5],[-5,-5],[-2.5,-5],[0,-5],[2.5,-5],[5,-5],[5,-2.5],[5,0]])
seg_init = ParallelMove(seg_org, delta_x, delta_y)

trace_y1 = []
trace_y2 = []
segments = []
rotation = []
distance = []

def get_segment(segments, idx):
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
    # pt_x0 = 0
    x_h = -1 * (A*B*pt + A*C) / (A**2 + B**2)
    y_h = (A*A*pt - B*C) / (A**2 + B**2)
    pt_h = [x_h, y_h]
    return pt_h

def get_foot_position_angle(pt):
    pt_h = get_foot_position(pt)
    angle = get_angle_of_point(pt_h)
    return angle

def get_foot_position_angle_explicit_ABC(pt, A, B, C):
    # pt_x0 = 0
    x_h = -1 * (A*B*pt + A*C) / (A**2 + B**2)
    y_h = (A*A*pt - B*C) / (A**2 + B**2)
    pt_h = [x_h, y_h]
    angle = get_angle_of_point(pt_h)
    return angle

def get_distance(pt): #from pt to line
    # pt_x0 = 0
    frac_up = B*pt + C
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

def judgment_of_touch_with_neighbor(pt_in, segments, index):
    if index < 0:
        pt_1 = segments[-2,:]
        pt_2 = segments[-1,:]
        idx_lw = len(segments)-2
        idx_up = len(segments)-1
    elif index >= len(segments)-1:
        pt_1 = segments[0,:]
        pt_2 = segments[1,:]
        idx_lw = 0
        idx_up = 1
    else:
        pt_1, pt_2 = get_segment(segments, index)
        idx_lw = index
        idx_up = index+1
    x1, y1 = pt_1
    x2, y2 = pt_2
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    pt_y = pt_in
    dist = get_distance_by_points(pt_1, pt_2, [0, pt_y])
    ang_lw = get_angle_of_point(pt_1)
    ang_up = get_angle_of_point(pt_2)
    ifcusp = False
    while(True):
        if (dist > r-T2 and dist < r+T2):
            angle = get_foot_position_angle_explicit_ABC(pt_y, A, B, C)
            if angle > ang_up or angle < ang_lw:
                ifcusp = True
            break
        elif (dist < r):
            if pt_y > 0:
                pt_y += T1
            else:
                pt_y -= T1
            dist = get_distance_by_points(pt_1, pt_2, [0, pt_y])
        else:
            if pt_y < 0:
                pt_y -= T1
            else:
                pt_y += T1
            dist = get_distance_by_points(pt_1, pt_2, [0, pt_y])
    return pt_y, ifcusp

def judgment_of_touch_y_positive(pt_1, pt_2, segments, index):
    pt_c = (pt_1[1] + pt_2[1]) / 2
    dist = get_distance(pt_c)
    ang_lw = get_angle_of_point(pt_1)
    ang_up = get_angle_of_point(pt_2)
    ifcusp = False
    while(True):
        if (dist < r-T2):
            pt_c += T1
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > ang_up:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_2)
                break
            elif angle < ang_lw:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_1)
                break
        elif (dist > r+T2):
            pt_c -= T1
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > ang_up:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_2)
                break
            elif angle < ang_lw:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_1)
                break
        else: # (dist > r-T2 and dist < r+T2):
            angle = get_foot_position_angle(pt_c)
            if angle > ang_up:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_2)
                break
            elif angle < ang_lw:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_1)
                break
            break
    return pt_c

def judgment_of_touch_y_negative(pt_1, pt_2, segments, index):
    pt_c = (pt_1[1] + pt_2[1]) / 2
    dist = get_distance(pt_c)
    ang_lw = get_angle_of_point(pt_1)
    ang_up = get_angle_of_point(pt_2)
    ifcusp = 0
    while(True):
        # 270-45 < angle < 270+45
        if (dist < r-T2):
            pt_c -= T1
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > ang_up:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_2)
                break
            elif angle < ang_lw:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_1)
                break
        elif (dist > r+T2):
            pt_c += T1
            dist = get_distance(pt_c)
            angle = get_foot_position_angle(pt_c)
            if angle > ang_up:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_2)
                break
            elif angle < ang_lw:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_1)
                break
        else: # if (dist > r-T2 and dist < r+T2):
            angle = get_foot_position_angle(pt_c)
            if angle > ang_up:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index+1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_2)
                break
            elif angle < ang_lw:
                pt_c, ifcusp = judgment_of_touch_with_neighbor(pt_c, segments, index-1)
                if ifcusp == True:
                    pt_c = adjust_gauge_to_cusp_y(pt_c, pt_1)
                break
            break
    return pt_c

def adjust_gauge_to_cusp_y(pt_in, pt_s):
    pt_c = [0, pt_in]
    dist = get_distance_btw_points(pt_c, pt_s)
    while(True):
        if (dist > r-T2 and dist < r+T2):
            break
        elif (dist < r):
            if pt_c[1] > 0:
                pt_c[1] += T1
            else:
                pt_c[1] -= T1
            dist = get_distance_btw_points(pt_c, pt_s)
        else:
            if pt_c[1] > 0:
                pt_c[1] -= T1
            else:
                pt_c[1] += T1
            dist = get_distance_btw_points(pt_c, pt_s)
    return pt_c[1]

def build_up_data(pt_c1, pt_c2):
    global trace_y1
    trace_y1.append(pt_c1)
    global trace_y2
    trace_y2.append(pt_c2)

# rotate the segments by 1 deg
for angle in range(circumference):
    seg = Rotate(seg_init, angle)
    rot = RotatePoint([delta_x, delta_y], angle)
    pt_c1 = 0
    pt_c2 = 0
    for idx in range(len(seg)-1):
        # find the segment that corrosing y-axis
        if seg[idx,1] > 0 and seg[idx,0] >=0 and seg[idx+1,0] < 0:
            pt_1, pt_2 = get_segment(seg, idx)
            set_segment_constants(pt_1, pt_2)
            pt_c1 = judgment_of_touch_y_positive(pt_1, pt_2, seg, idx)
        elif seg[idx,1] < 0 and seg[idx,0] <=0 and seg[idx+1,0] > 0:
            pt_1, pt_2 = get_segment(seg, idx)
            set_segment_constants(pt_1, pt_2)
            pt_c2 = judgment_of_touch_y_negative(pt_1, pt_2, seg, idx)
    build_up_data(pt_c1, pt_c2)
    segments.append(seg)
    rotation.append(rot)

segments = np.array(segments).reshape(circumference,len(seg_init),2)
rotation = np.array(rotation)

for i in range(len(trace_y1)-1):
    dist = trace_y1[i] - trace_y2[i]
    distance.append(dist - 2*r)
distance.append(distance[0])

# plot
fig0, ax0 = plt.subplots()
ax0.set_xlim(0, 360)
ax0.set_ylim(8, 16)
ax0.set_xticks(30*np.arange(12))
probefixed, = ax0.plot(np.arange(circumference), distance, 'r-')
ax0.grid()

fig1, (ax1, ax2) = plt.subplots(ncols=2)
ax1.set_aspect("equal")
ax1.set_xlim(-13, 13)
ax1.set_ylim(-13, 13)
ax1.set_title('Dial Gauge')
ax2.set_xlim(0, 360)
ax2.set_ylim(8, 16)
ax2.set_title('Gauge Distance')
ax2.set_xlabel('Angle, deg')
ax2.set_xticks(45*np.arange(8))
ax2.grid()

# trace, = ax1.plot([], [], 'b-')
seg_draw, = ax1.plot([], [], 'k-')
circle1, = ax1.plot([], [], 'g-', linewidth=1)
circle2, = ax1.plot([], [], 'g-', linewidth=1)
wand, = ax1.plot([], [], '.y-', linewidth=1)
punct, = ax1.plot([], [], '.k-', linewidth=1)
probe, = ax2.plot([], [], 'r-')

line_org1 = plt.Line2D([-1,1], [0,0], color='k', linewidth=0.5)
line_org2 = plt.Line2D([0,0], [-1,1], color='k', linewidth=0.5)
ax1.add_artist(line_org1)
ax1.add_artist(line_org2)
#punct = plt.Circle([2,2], 0.1, color="b")
#ax1.add_artist(punct)

def animate(i):
    y1 = trace_y1[i]
    y2 = trace_y2[i]
    cx1, cy1 = circle_drawing(0, y1, r)
    circle1.set_data(cx1, cy1)
    cx2, cy2 = circle_drawing(0, y2, r)
    circle2.set_data(cx2, cy2)
    seg_draw.set_data((segments[i,:,0], segments[i,:,1]))
    wand.set_data((0, 0), (y1, y2))
    punct.set_data((rotation[i,0], rotation[i,1]))
    probe.set_data((np.arange(i), distance[:i]))
    return circle1, circle2, seg_draw, wand, punct, probe,

def circle_drawing(c_x, c_y, radius):
    theta = np.linspace(0, 2*np.pi, 100)
    x = c_x + radius * np.cos(theta)
    y = c_y + radius * np.sin(theta)
    return x, y

anim = FuncAnimation(fig1, animate, interval=20, blit=True, frames=circumference)

plt.show()