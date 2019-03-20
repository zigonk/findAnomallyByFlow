#   compute colored image to visualize optical flow file .flo

#   According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
#   Contact: dqsun@cs.brown.edu
#   Contact: schar@middlebury.edu

#   Author: Johannes Oswald, Technical University Munich
#   Contact: johannes.oswald@tum.de
#   Date: 26/04/2017

#	For more information, check http://vision.middlebury.edu/flow/

import cv2
import sys
import numpy as np
import argparse
import os

TAG_FLOAT = 202021.25

def makeColorwheel():

	#  color encoding scheme

	#   adapted from the color circle idea described at
	#   http://members.shaw.ca/quadibloc/other/colint.htm

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3])  # r g b

	col = 0
	#RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
	col += RY

	#YG
	colorwheel[col:YG+col, 0] = 255 - np.floor(255*np.arange(0, YG, 1)/YG)
	colorwheel[col:YG+col, 1] = 255
	col += YG

	#GC
	colorwheel[col:GC+col, 1] = 255
	colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
	col += GC

	#CB
	colorwheel[col:CB+col, 1] = 255 - np.floor(255*np.arange(0, CB, 1)/CB)
	colorwheel[col:CB+col, 2] = 255
	col += CB

	#BM
	colorwheel[col:BM+col, 2] = 255
	colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
	col += BM

	#MR
	colorwheel[col:MR+col, 2] = 255 - np.floor(255*np.arange(0, MR, 1)/MR)
	colorwheel[col:MR+col, 0] = 255
	return colorwheel


def computeColor(u, v):

	colorwheel = makeColorwheel()
	nan_u = np.isnan(u)
	nan_v = np.isnan(v)
	nan_u = np.where(nan_u)
	nan_v = np.where(nan_v)

	u[nan_u] = 0
	u[nan_v] = 0
	v[nan_u] = 0
	v[nan_v] = 0

	ncols = colorwheel.shape[0]
	radius = np.sqrt(u**2 + v**2)
	a = np.arctan2(-v, -u) / np.pi
	fk = (a+1) / 2 * (ncols-1)  # -1~1 maped to 1~ncols
	k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
	k1 = k0+1
	k1[k1 == ncols] = 0
	f = fk - k0

	img = np.empty([k1.shape[0], k1.shape[1], 3])
	ncolors = colorwheel.shape[1]
	for i in range(ncolors):
		tmp = colorwheel[:, i]
		col0 = tmp[k0]/255
		col1 = tmp[k1]/255
		col = (1-f)*col0 + f*col1
		idx = radius <= 1
		col[idx] = 1 - radius[idx]*(1-col[idx])  # increase saturation with radius
		col[~idx] *= 0.75  # out of range
		img[:, :, 2-i] = np.floor(255*col).astype(np.uint8)

	return img.astype(np.uint8)

global vTable, lTable, cTable
vTable = [[[0, 0]] * 1000 for i in range(1001)]
avgTable = [[[0, 0]] * 1000 for i in range(1001)]
cTable = [[0] * 1000 for i in range(1001)]

def computeImg(flow):

	eps = sys.float_info.epsilon
	UNKNOWN_FLOW_THRESH = 1e9
	UNKNOWN_FLOW = 1e10

	u = flow[:, :, 0]
	v = flow[:, :, 1]

	maxu = -999
	maxv = -999

	minu = 999
	minv = 999

	maxrad = -1
	#fix unknown flow
	greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
	greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
	u[greater_u] = 0
	u[greater_v] = 0
	v[greater_u] = 0
	v[greater_v] = 0

	maxu = max([maxu, np.amax(u)])
	minu = min([minu, np.amin(u)])

	maxv = max([maxv, np.amax(v)])
	minv = min([minv, np.amin(v)])
	rad = np.sqrt(np.multiply(u, u)+np.multiply(v, v))
	maxrad = max([maxrad, np.amax(rad)])

	u = u/(maxrad+eps)
	v = v/(maxrad+eps)
	img = computeColor(u, v)
	return img


MOVING_THRESHOLD = 0.00
ANOMALY_THRESHOLD = 0.1
STALLING_PIXELS_COUNT_THRESHOLD = 10
ANOMALY_PIXELS_COUNT_THRESHOLD = 10
MIN_PIXELS_IN_GROUP = 5
GROUP_SIZE = 2 # 10x10
RANGE_L =	600
RANGE_R = 900

def processInputFlow(flow):
	height = flow.shape[0]
	width = flow.shape[1]

	for i in range(height):
		for j in range(width):
			sum = [0, 0]
			count = 0
			for k1 in range(max(0, i - GROUP_SIZE), min(height, i + GROUP_SIZE + 1)):
				for k2 in range(max(0, j - GROUP_SIZE), min(width, j + GROUP_SIZE + 1)):
					# if (np.linalg.norm(flow[i][j]) >= MOVING_THRESHOLD):
					sum[0] += flow[k1][k2][0]
					sum[1] += flow[k1][k2][1]
					count += 1
					sum[0] = sum[0] / float(max(1.0, count))
					sum[1] = sum[1] / float(max(1.0, count))
				if (np.linalg.norm(sum) >= MOVING_THRESHOLD):
					vTable[i][j][0] += sum[0]
					vTable[i][j][1] += sum[1]
					cTable[i][j] += 1
				# vTable[i][j] += flow[i][j]
				# vTable[i][j] = [x + y for x, y in zip(vTable[i][j], flow[i][j])]
				# cTable[i][j] += 1

def preprocess(flowfileFolder):
	list = os.listdir(flowfileFolder)  # dir is your directory path

	# for i in range(number_files):
	for i in range(RANGE_L, RANGE_R):
		# index = (i + 1) * 2
		index = i
		if (index % 50 == 0):
			print('Preprocessing %d . . .' % index)
		flowPath = os.path.join(flowfileFolder, 'flow%d.flo' % index)
		flow = readFlowFile(flowPath)

		processInputFlow(flow)

def calculateAvgTable():
	print('Calculating avgTable . . .')

	stalling_pixels = 0

	for i in range(1000):
		for j in range(1000):
			avgTable[i][j] = [(float(x) / float(max(cTable[i][j], 1.0))) for x in vTable[i][j]]
			if (np.linalg.norm(avgTable[i][j]) >= MOVING_THRESHOLD and cTable[i][j] <= 15):
				stalling_pixels += 1

	if (stalling_pixels >= STALLING_PIXELS_COUNT_THRESHOLD):
		print('[ANOMALY]: stalling vehicle')

def detectAnomaly(flow, frameIndex):
	height = flow.shape[0]
	width = flow.shape[1]
	anomaly_pixels_count = 0

	for i in range(height):
		for j in range(width):
			sum = [0, 0]
			count = 0
			for k1 in range(max(0, i - GROUP_SIZE), min(height, i + GROUP_SIZE + 1)):
				for k2 in range(max(0, j - GROUP_SIZE), min(width, j + GROUP_SIZE + 1)):
					# if (np.linalg.norm(flow[i][j]) >= MOVING_THRESHOLD):
					sum[0] += flow[k1][k2][0]
					sum[1] += flow[k1][k2][1]
					count += 1
				# if (count >= MIN_PIXELS_IN_GROUP):
					sum[0] = sum[0] / float(max(1.0, count))
					sum[1] = sum[1] / float(max(1.0, count))
				if (np.linalg.norm(sum) >= MOVING_THRESHOLD):
					diff = [x - y for x, y in zip(avgTable[i][j], sum)]
					if (np.linalg.norm(diff) >= ANOMALY_THRESHOLD):
						anomaly_pixels_count += 1
	print('[FRAME %d] Pixels count: %d' % (frameIndex, anomaly_pixels_count))
	if (anomaly_pixels_count >= ANOMALY_PIXELS_COUNT_THRESHOLD):
		print('[ANOMALY] Frame index: ', frameIndex)


def findingAnomaly(flowfileFolder):
	list = os.listdir(flowfileFolder)  # dir is your directory path
	number_files = len(list)

	# for i in range(number_files):
	for i in range(RANGE_L, RANGE_R):
		# index = (i + 1) * 2
		index = i
		if (index % 50 == 0):
				print('Running %d . . .' % index)
		flowPath = os.path.join(flowfileFolder, 'flow%d.flo' % index)
		flow = readFlowFile(flowPath)

		detectAnomaly(flow, index)


def readFlowFile(file):
	assert type(file) is str, "file is not str %r" % str(file)
	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	f = open(file, 'rb')
	flo_number = np.fromfile(f, np.float32, count=1)[0]
	assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
	w = np.fromfile(f, np.int32, count=1)
	h = np.fromfile(f, np.int32, count=1)
	# if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	# data = np.fromfile(f, np.float32, count=2*w*h)
	data = np.fromfile(f, np.float32, count=2*w[0]*h[0])

	# Reshape data into 3D array (columns, rows, bands)
	flow = np.resize(data, (int(h), int(w), 2))
	f.close()

	return flow


def main():
	flowfileFolder = '/content/drive/My Drive/PWC-Net/flow/%d' % 91
	# flowfileFolder = './flow'
	preprocess(flowfileFolder)
	calculateAvgTable()
	findingAnomaly(flowfileFolder)


main()

# a group of pixel differ from avg 10x10
# a group pixels of only movement at that pixels 10x10
