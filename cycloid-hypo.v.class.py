from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Hypocycloid:

    def __init__(self, R = 3, r = 1, frames = 100, ncycles = 1):
        self.frames = frames
        self.ncycles = ncycles
        self.R = R
        self.r = r
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')

        ## main circle: R
        theta = np.linspace(0, 2*np.pi, 100)
        x = R * np.cos(theta)
        y = R * np.sin(theta)

        self.circle_main, = self.ax.plot(x, y, 'b-')

        ## moving circle: r
        x = r * np.cos(theta) + R - r
        y = r * np.sin(theta)
        self.circle_move, = self.ax.plot(x, y, 'k-')

        ##line and dot:
        self.line, = self.ax.plot([R-r,0], [0,0], 'k-')
        self.dot, = self.ax.plot([R-r], [0], 'ko', ms=5)
        ##hypocycloid:
        self.hypocycloid, = self.ax.plot([], [], 'r-')

        self.animation = FuncAnimation(
            self.fig, self.animate,
            frames=self.frames*self.ncycles,
            interval=50, blit=False,
            repeat_delay=2000,
        )

    def update_moving_circle(self, phi):
        theta = np.linspace(0, 2*np.pi, 100)
        x = (self.R-self.r)*np.cos(phi) + self.r*np.cos(theta)
        y = (self.R-self.r)*np.sin(phi) + self.r*np.sin(theta)
        self.circle_move.set_data(x, y)

    def update_hypocycloid(self, phis):
        x = (self.R-self.r)*np.cos(phis) + self.r*np.cos((self.R-self.r)/self.r*phis)
        y = (self.R-self.r)*np.sin(phis) - self.r*np.sin((self.R-self.r)/self.r*phis)
        self.hypocycloid.set_data(x, y)

        center = [(self.R-self.r)*np.cos(phis[-1]), (self.R-self.r)*np.sin(phis[-1])]

        self.line.set_data([center[0],x[-1]], [center[1],y[-1]])
        self.dot.set_data([center[0]], [center[1]])

    def animate(self, frame):
        frame = frame + 1
        phi = 2*np.pi*frame/self.frames
        self.update_moving_circle(phi)
        self.update_hypocycloid(np.linspace(0, phi, frame))


hypo = Hypocycloid(R=4, r=1, frames=100, ncycles=4)

##un-comment the next line, if you want to save the animation as gif:
##hypo.animation.save('hypocycloid.gif', writer='imagemagick', fps=10, dpi=75)

plt.show()