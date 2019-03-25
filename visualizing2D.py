import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import pylab

# make values from -5 to 5, for this example

UNKNOWN_FLOW_THRESH = 0.5

def readFlowFile(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    # if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # data = np.fromfile(f, np.float32, count=2*w*h)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()

    u = flow[: , : , 0]
    v = flow[: , : , 1]
    smaller_u = np.where(abs(u) < UNKNOWN_FLOW_THRESH)
    smaller_v = np.where(abs(v) < UNKNOWN_FLOW_THRESH)
    u[smaller_u] = -100
    u[smaller_v] = -100
    v[smaller_u] = -100
    v[smaller_v] = -100
    flow[: , : , 0] = u
    flow[: , : , 1] = v
    tmp = []
    for row in flow:
      for element in row:
        tmp.append(element)

    return tmp


fig, ax = plt.subplots(tight_layout=True)

for i in range(100, 301):
  print('../flow/flow%d.flo' % i)
  arr = readFlowFile('../flow/flow%d.flo' % i)

  x = []
  y = []
  for element in arr:
    x.append(element[0])
    y.append(element[1])

  hist = ax.hist2d(x, y, bins=100, range=[[-30, 30],[-30, 30]])
  # plt.savefig('./plt/plt%d.png' % i)
  plt.pause(0.001)

plt.show()


