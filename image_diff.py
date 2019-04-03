# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True,
#                 help="first input image")
# ap.add_argument("-s", "--second", required=True,
#                 help="second")
# args = vars(ap.parse_args())

def calculate_confidence(diff, x, y, w, h):
	sumSimilarity = 0
	for i in range(y, y + h):
		for j in range(x, x + w):
			sumSimilarity += diff[i][j]
	return float(sumSimilarity) / float(255 * w * h)



def calculate_diff(videoNo, A, B):
	# load the two input images
	srcA = "/content/drive/My Drive/AIC_2019_Train_Cut/cut_video_bg_frames/%d/%s" % (videoNo, A)
	srcB = "/content/drive/My Drive/AIC_2019_Train_Cut/cut_video_bg_frames/%d/%s" % (videoNo, B)
	if not os.path.exists(srcA):
		return
	if not os.path.exists(srcB):
		return
	imageA = cv2.imread(srcA)
	imageB = cv2.imread(srcB)
	# x = 690
	# y = 100
	# h = 25
	# w = 25
	# imageA = imageA[y:y+h, x:x+w]
	# imageB = imageB[y:y+h, x:x+w]

	# convert the images to grayscale
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	# print("SSIM: {}".format(score))

	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	# thresh = cv2.threshold(diff, 0, 255,
	#                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = cv2.threshold(diff, 50, 255,
												cv2.THRESH_BINARY_INV)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
													cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# compute the bounding box of the contour and then draw the
		# bounding box on both input images to represent where the two
		# images differ
		(x, y, w, h) = cv2.boundingRect(c)
		confidence = calculate_confidence(diff, x, y, w, h)
		cv2.putText(imageB,"%f" % confidence, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))
		cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# show the output images
	# cv2.imshow("Original", imageA)
	# cv2.imshow("Modified", imageB)
	# cv2.imshow("Diff", diff)
	# cv2.imshow("Thresh", thresh)
	# cv2.waitKey(0)
	
	srcOriginal = "/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/original/%s" % (videoNo, B)
	srcDiff = "/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/diff/%s" % (videoNo, B)
	srcThresh = "/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/thresh/%s" % (videoNo, B)
	cv2.imwrite(srcOriginal, imageB)
	cv2.imwrite(srcDiff, diff)
	cv2.imwrite(srcThresh, thresh)

def createDirectory(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

FRAME_DISTANCE = 15

for video in range(1, 101):
	print("Processing %d" % video)
	previousFile = None
	directory = "/content/drive/My Drive/AIC_2019_Train_Cut/cut_video_bg_frames/%d" % video
	createDirectory("/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d" % video)
	createDirectory("/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/original" % video)
	createDirectory("/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/diff" % video)
	createDirectory("/content/drive/My Drive/AIC_2019_Train_Cut/difference/%d/thresh" % video)

	files = os.listdir(directory)
	number_files = len(files)

	for i in range(FRAME_DISTANCE + 1, number_files):
		previousFile = "%05d.jpg" % ((i - FRAME_DISTANCE) * 30)
		currentFile = "%05d.jpg" % (i * 30)
		calculate_diff(video, previousFile, currentFile)
