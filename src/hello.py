#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

print("Hello, Nerine!")
print("on")

x = np.arange(5)
y = np.arange(5)

print(x)

plt.plot(x,y)
plt.savefig('/net/labdata/nerine/space-time-clouds/figure.png')
plt.show()
