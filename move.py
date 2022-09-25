import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = 4,3
from matplotlib.animation import FuncAnimation

#-----------------------------------#
r = 1.2 # radius of circle
h = 0.1 # distance of circle flying
#-----------------------------------#

# create a figure with an axes
fig, ax = plt.subplots()
# set the axes limits
ax.set_xlim(-2*h-r, r+2*h)
ax.set_ylim(-2*h-r, r+2*h)
# set equal aspect such that the circle is not shown as ellipse
ax.set_aspect("equal")
# create a point in the axes
phi = np.linspace(0, 2*np.pi, 360)
x = (r - h) * np.cos(phi)
y = (r - h) * np.sin(phi)

line, = ax.plot(x, y, "k-")
point, = ax.plot(0, 1, marker="o")

# Updating function, to be repeatedly called by the animation
def update(phi):
    # obtain point coordinates 
    x,y = circle(phi)
    # set point's coordinates
    point.set_data([x],[y])
    return point,

def circle(phi):
    return np.array([r*np.cos(phi), r*np.sin(phi)])

# create animation with 10ms interval, which is repeated,
# provide the full circle (0,2pi) as parameters
ani = FuncAnimation(fig, update, interval=10, blit=True, repeat=True,
                    frames=np.linspace(0, 2*np.pi, 360, endpoint=False))

#plt.grid()
plt.show()