import numpy as np
from math import radians, sin, cos

#segments = np.array([[5,0],[5,5],[0,5],[-5,5],[-5,0],[-5,-5],[0,-5],[5,-5],[5,0]])

def ParallelMove(segments, delta_x, delta_y):
    x_new = segments[:,0] + delta_x
    y_new = segments[:,1] + delta_y
    new_position = np.concatenate((x_new, y_new), axis=None).reshape(2,len(x_new))
    seg_new = _segment_reset(new_position.transpose())
    return seg_new
    
def Rotate(segments, theta):
    x_new = segments[:,0] * cos(radians(theta)) - segments[:,1] * sin(radians(theta))
    y_new = segments[:,0] * sin(radians(theta)) + segments[:,1] * cos(radians(theta))
    new_position = np.concatenate((x_new, y_new), axis=None).reshape(2,len(x_new))
    seg_new = _segment_reset(new_position.transpose())
    return seg_new

def _segment_reset(seg_input):
    new_new = seg_input
    for idx in range(len(seg_input)-1):
        if seg_input[idx,1] < 0 and seg_input[idx+1,1] > 0:
            new_new = _bisect_this_segment(seg_input, idx)
    return new_new

def _bisect_this_segment(seg_input, idx):
    x1, y1 = seg_input[idx,:]
    x2, y2 = seg_input[idx+1,:]
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    x3 = -1 * C / A
    #new_segments = np.delete(seg_input, idx+1, axis=0)
    new_segments = np.insert(seg_input, idx+1, [x3,0], axis=0)
    seg_to_front = new_segments[idx+1:,:]
    seg_to_back = new_segments[1:idx+2,:]
    new_new = np.append(seg_to_front, seg_to_back, axis=0)
    return new_new
