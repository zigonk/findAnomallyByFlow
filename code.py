import numpy as np
import cv2 as cv

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.2,
                       minDistance = 50,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

global vTable, lTable, cTable
vTable = [[[0, 0]] * 1000 for i in range(1001)]
lTable = [[[]] * 1000 for i in range(1001)]
cTable = [[0] * 1000 for i in range(1001)]

# table to store velocity and number of vector through all pixels

run_cycle = 30 * 100

def preprocess(video_name):
    cap = cv.VideoCapture(video_name)

    # Take first frame and find corners in it

    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    count = 1
    while(count <= run_cycle):
        print(count / 30)
        count = count + 1
        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p0new = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        for i in p0new:
            found = 0
            for j in p0:
                # if np.array_equal(i, j):
                if np.linalg.norm(i - j) <= 50:
                    found = 1
            if (found == 0):
                p0 = np.concatenate((p0, np.array([i])))
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color[i % 100].tolist(), 2)
            frame = cv.circle(frame, (a, b), 5, color[i % 100].tolist(), -1)
            currentV = new - old
            if (np.linalg.norm(currentV) >= 1):
                iC = (int)(c)
                iD = (int)(d)
                vTable[iC][iD] = vTable[iC][iD] + currentV
                lTable[iC][iD].append(currentV)
                cTable[iC][iD] = cTable[iC][iD] + 1
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv.destroyAllWindows()
    cap.release()
    for i in range(1, 999):
        for j in range(1, 999):
            if (cTable[i][j] > 0):
                vTable[i][j][:] = [x / cTable[i][j] for x in vTable[i][j]]

def is_anomaly(video_name):
    show_abnormal = False
    abnormal_center = None
    abnormal_radius = 50
    abnormal_time = False
    abnormal_threshold = 7

    cap = cv.VideoCapture(video_name)

    # Take first frame and find corners in it

    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    count  = 1
    while(count <= run_cycle):
        print(count / 30)
        count = count + 1
        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p0new = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        for i in p0new:
            found = 0
            for j in p0:
                # if np.array_equal(i, j):
                if np.linalg.norm(i - j) <= 50:
                    found = 1
            if (found == 0):
                p0 = np.concatenate((p0, np.array([i])))

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color[i % 100].tolist(), 2)
            frame = cv.circle(frame, (a, b), 5, color[i % 100].tolist(), -1)
            currentV = new - old
            iC = (int)(c)
            iD = (int)(d)
            if (not(show_abnormal) and np.linalg.norm(currentV) >= 1):
                if (np.linalg.norm(currentV - vTable[iC][iD]) >= abnormal_threshold):
                    print("Abnormal")
                    print(currentV)
                    print(vTable[iC][iD], cTable[iC][iD])
                    abnormal_center = [iC, iD]
                    show_abnormal = True
        # img = cv.add(frame, mask)
        img = frame
        if (show_abnormal):
            cv.circle(img, (abnormal_center[0], abnormal_center[1]), abnormal_radius, (0,255,0), thickness=1, lineType=8, shift=0)
        cv.imshow('show_abnormal_frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv.destroyAllWindows()
    cap.release()
    return show_abnormal

preprocess("51c.mp4")
print("preprocess done")
is_anomaly("51c.mp4")
