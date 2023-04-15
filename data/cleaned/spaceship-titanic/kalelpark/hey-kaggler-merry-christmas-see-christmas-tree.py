import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
k = 360
z = 1
x = [math.cos(i) * 7 for i in range(1, 360)]
y = [math.sin(i) * 7 for i in range(1, 360)]
plt.scatter(x, y, z)
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
for s in reversed(range(0, 300, 25)):
    z = s
    x = [math.cos(i) * (300 - s) for i in range(1, 360, 5)]
    y = [math.sin(i) * (300 - s) for i in range(1, 360, 5)]
    ax.scatter(x, y, z, c='green')
    if s % 50 == 0:
        z = s
        x = [math.cos(i) * (300 - s) for i in range(1, 360, 5)]
        y = [math.sin(i) * (300 - s) for i in range(1, 360, 5)]
        ax.scatter(x, y, z, c='red')
ax.scatter(0, 0, 301, s=250, c='blue', marker='*')
z = np.random.randint(0, 300, 100)
x = np.random.randint(-500, 500, 100)
y = np.random.randint(-500, 500, 100)
ax.scatter(x, y, z, c='gray')
plt.xlim(-500, 500)
plt.ylim(-500, 500)
plt.axis('off')