# TO DO: extract background again with dupplicating video
# Increase FRAME_DISTANCE to about 60

import cv2
import os
import numpy as np

FRAME_DISTANCE = 15
DIFFERENCE_THRESH = 0.7
CONTINUOUS_FRAME_DIFF_THRESH = 0.7
BLOCK_WIDTH = 25
BLOCK_HEIGHT = 25
HEIGHT = 410
WIDTH = 800

output = open("output.txt","w+")

def isDifferent(image, X, Y):
	count = 0
	for i in range(X, min(X + BLOCK_HEIGHT, HEIGHT)):
		for j in range(Y, min(Y + BLOCK_WIDTH, WIDTH)):
			count += (image[i][j] == 255)
	if (count / float(BLOCK_HEIGHT * BLOCK_WIDTH) >= DIFFERENCE_THRESH):
		return 1
	else:
		return 0

def calculate_difference(videoNo, file):
	src = "/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/thresh/%s" % (videoNo, file)
	if not os.path.exists(src):
		return False
	image = cv2.imread(src)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	diff = np.zeros((HEIGHT, WIDTH))
	for i in range(0, HEIGHT, BLOCK_HEIGHT):
		for j in range(0, WIDTH, BLOCK_WIDTH):
			diff[int(i / BLOCK_HEIGHT)][(int)(j / BLOCK_WIDTH)] = isDifferent(image, i, j)
	difference.append(diff)


def same(frame, X, Y):
	count = 0
	for i in range(frame, min(frame + FRAME_DISTANCE, len(difference))):
		count += difference[i][X][Y]
	if (count / FRAME_DISTANCE >= 1.0 - CONTINUOUS_FRAME_DIFF_THRESH):
		return True
	else :
		return False
	
def differ(frame, X, Y):
	return not(same(frame, X, Y))

def annotate_anomaly(video, frame, X, Y):
	file  = "%05d.jpg" % (frame * 30)
	x = X * BLOCK_HEIGHT
	y = Y * BLOCK_WIDTH
	src = "/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/thresh/%s" % (video, file)
	img = cv2.imread(src)
	cv2.rectangle(img,(x, y),(x + BLOCK_HEIGHT, y + BLOCK_WIDTH),(255, 0, 0), 3)
	outsrc = "/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/anomaly/%s" % (video, file)
	cv2.imwrite(outsrc, img)


def findAnomaly(video):
	for i in range(0, HEIGHT / BLOCK_HEIGHT):
		for j in range(0, (WIDTH / BLOCK_WIDTH) + bool(WIDTH % BLOCK_WIDTH)):
			for f in range(FRAME_DISTANCE, len(difference)):
				if (same(f - FRAME_DISTANCE, i, j) and differ(f, i, j) and same(f + FRAME_DISTANCE, i, j)):
					print("[ANOMALY] Video %d, frame %d, block (%d, %d)" % (video, f, i, j))
					annotate_anomaly(video, f, i, j)
					output.write("%d %d" % (video, f))


def createDirectory(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

global difference

for video in range(1, 101):
	print("Processing %d" % video)
	previousFile = None

	createDirectory("/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/anomaly" % video)
	thresh_directory = "/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/thresh" % video	

	files = os.listdir(thresh_directory)
	number_files = len(files)

	difference = []

	for i in range(0, number_files):
		currentFile = "%05d.jpg" % (i * 30)
		calculate_difference(video, currentFile)
	findAnomaly(video)

output.close()