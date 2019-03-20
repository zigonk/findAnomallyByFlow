import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import pylab
import cv2

# make values from -5 to 5, for this example


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

    tmp = []
    for row in flow:
      for element in row:
        tmp.append(element)

    return tmp


fig, ax = plt.subplots(tight_layout=True)
plt.xlim(-50, 50)
plt.ylim(-50, 50)

flowfileFolder = './flow'

list = os.listdir(flowfileFolder)  # dir is your directory path
number_files = len(list)
print(number_files)

video_name = 'video.avi'

images = []

width = 0
height = 0

for i in range(number_files):
  print('./flow/flow%d.flo' % i)
  arr = readFlowFile('./flow/flow%d.flo' % (i+100))

  x = []
  y = []
  for element in arr:
    x.append(element[0])
    y.append(element[1])

  hist = ax.hist2d(x, y, bins=100, range=[[-0.1, 0.1], [-0.1, 0.1]])
  # plt.savefig('./plt/plt%d.png' % i)
  fig = plt.figure()
  height, width = fig.shape
  images.append(fig)

video = cv2.VideoWriter(video_name, 0, 1, [width, height])

for image in images:
  video.write(image)
video.release()
