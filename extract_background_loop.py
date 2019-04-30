import cv2, os

#loop time
loop_time = 120
# time between frame ( time * frame rate = 30)
time_rate = 5

rt = '/content/drive/My Drive/AIC_2019_Train_Cut_abs(3)/video'
videos = os.listdir(rt)
#path for original frames
wrt_ori = '/content/drive/My Drive/AIC_2019_Train_Cut_abs(3)/orig_frames'
#path for background frames
wrt_bg = '/content/drive/My Drive/AIC_2019_Train_Cut_abs(3)/bg_frames'
# wrt_bg = './data/all_imgs/bg'
if not os.path.exists(wrt_ori):
    os.mkdir(wrt_ori)
if not os.path.exists(wrt_bg):
    os.mkdir(wrt_bg)


left = 51
right = 55
index = 0

for video in videos:
    index += 1
    if (index < left or index > right):
        continue
    print (video)
    if not os.path.exists(os.path.join(wrt_bg, video.split('.')[0])):
        os.mkdir(os.path.join(wrt_bg, video.split('.')[0]))
    if not os.path.exists(os.path.join(wrt_ori, video.split('.')[0])):
        os.mkdir(os.path.join(wrt_ori, video.split('.')[0]))

    #read video
    cap = cv2.VideoCapture(os.path.join(rt, video))
    ret, frame = cap.read()

    #build MOG2 model
    bs = cv2.createBackgroundSubtractorMOG2(history=120)
    bs.setHistory(120)

    count = 1
    frame_store = [frame]
    while ret:
        fg_mask = bs.apply(frame)
        bg_img = bs.getBackgroundImage()
        #filter frame 
        if (count % time_rate == 0) and (count > 30*loop_time) :
            cv2.imwrite(os.path.join(wrt_bg, video.split('.')[0], str(int(count)).zfill(5)+'.jpg'), bg_img)
        
        ret, frame = cap.read()
        if (count < 30 * loop_time) :
            frame_store.append(frame)
        
        # when enough frame for loop, loop and print background after loop
        if (count == 30 * loop_time) :
            loop_count = 1
            for loop_frame in frame_store :
                fg_mask = bs.apply(loop_frame)
                bg_img = bs.getBackgroundImage()
                if (loop_count % time_rate == 0) :
                    cv2.imwrite(os.path.join(wrt_bg, video.split('.')[0], str(int(loop_count)).zfill(5)+'.jpg'), bg_img)
                loop_count += 1
        count += 1

