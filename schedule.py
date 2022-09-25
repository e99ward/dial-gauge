from socket import timeout
import numpy as np
import matplotlib.pyplot as plt
import math
from time import sleep

fig = plt.figure(figsize=(5, 10))

ax1 = plt.subplot(2, 1, 1)
xs = np.arange(7)
ys = xs**2
line1, = ax1.plot(xs, ys, 'ro')

ax2 = plt.subplot(2, 1, 2, projection='polar')
rads = np.arange(0, (2 * np.pi), 0.01)
for rad in rads:
    ax2.plot(rad, rad, 'g.')

# x = np.linspace(0, 10*np.pi, 100)
# y = np.sin(x)
  
# plt.ion()
# fig = plt.figure()
# ax2 = plt.subplot(2, 1, 2)
# line1, = ax2.plot(x, y, 'b-')
  
# for phase in np.linspace(0, 10*np.pi, 100):
#     line1.set_ydata(np.sin(0.5 * x + phase))
#     fig.canvas.draw()
#     fig.canvas.flush_events()

plt.draw()
plt.pause(2)
ax1.remove()
ax1 = plt.subplot(2, 1, 1)
x = np.linspace(-1.0, 1.0, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 0.6
plt.contour(X,Y,F,[0])
plt.gca().set_aspect('equal')
plt.draw()



plt.show()