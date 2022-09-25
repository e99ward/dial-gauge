import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import degrees, radians, sin, cos, tan, sqrt, atan

#-----------------------------------#
R = 5 #float(input(" R = "))
r = 1
#-----------------------------------#

circumference = 360
T1 = 0.01 # tolerance
T2 = 0.02 # tolerance
#pt_c = [0.0, 0.0]
segments = np.array([[5,0],[5,5],[0,5],[-5,5],[-5,0],[-5,-5],[0,-5],[5,-5],[5,0]])

def parallel_move(delta):
    new_position = segments + delta
    return new_position

def parallel_rotate(theta):
    x_new = segments[:,0] * cos(radians(theta)) - segments[:,1] * sin(radians(theta))
    y_new = segments[:,0] * sin(radians(theta)) + segments[:,1] * cos(radians(theta))
    new_position = np.concatenate((x_new, y_new), axis=None).reshape(2,len(x_new))
    return new_position.transpose()

def segment_reset(seg_input):
    for idx in range(len(seg_input)-1):
        if seg_input[idx,1] < 0 and seg_input[idx+1,1] > 0:
            new_new = bisect_this_segment(seg_input, idx)
    return new_new

def bisect_this_segment(seg_input, idx):
    x1, y1 = seg_input[idx,:]
    x2, y2 = seg_input[idx+1,:]
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    x3 = -1 * C / A
    #new_segments = np.delete(seg_input, idx+1, axis=0)
    new_segments = np.insert(seg_input, idx+1, [x3,0], axis=0)
    #print(new_segments)
    seg_to_front = new_segments[idx+1:,:]
    seg_to_back = new_segments[1:idx+2,:]
    new_new = np.append(seg_to_front, seg_to_back, axis=0)
    #print(new_new)
    return new_new

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.set_aspect("equal")
ax1.set_xlim(-8, 8)
ax1.set_ylim(-8, 8)
ax2.set_aspect("equal")
ax2.set_xlim(-8, 8)
ax2.set_ylim(-8, 8)

trace, = ax1.plot([], [], 'b-')
circle, = ax1.plot([], [], 'g-', linewidth=1)
wand, = ax1.plot([], [], '.y-', linewidth=1)
probe, = ax2.plot([], [], 'r-')
# probe, = ax2.plot(np.arange(360), distance, 'r-')
# for idx in range(len(angles)-1):
#     x, y = get_segment(idx)
#     line_seg = plt.Line2D(x, y)
#     ax.add_artist(line_seg)

line_seg = plt.Line2D(segments[:,0], segments[:,1], color='k')
ax1.add_artist(line_seg)
# trace, = ax.plot(trace_x, trace_y)

seg_new2 = parallel_rotate(-25)
print(seg_new2)
# line_seg2 = plt.Line2D(seg_new[:,0], seg_new[:,1], color='k')
# ax2.add_artist(line_seg2)

seg_new = segment_reset(seg_new2)
print(seg_new)
line_seg2 = plt.Line2D(seg_new[:,0], seg_new[:,1], color='k')
ax2.add_artist(line_seg2)

plt.show()