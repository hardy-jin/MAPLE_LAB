import cv2
import os
import numpy as np

# load video form  your path
filename = 'clip_017284'
vc = cv2.VideoCapture('./' + filename + '.mp4')
save_pth = './surgical_tool_video/'
video_fold = save_pth + filename
if not os.path.exists(video_fold):
    os.makedirs(video_fold)

c = 0
# rval = vc.isOpened()
rval = True
while rval:
    c = c + 1
    rval, frame = vc.read()
    if not rval:
        continue
    cropImg = frame[55:675,195:1085,:]
    cv2.imwrite(video_fold + '/' + filename + '_' + '%04d' % c + '.jpg', cropImg)


vc.release()

labels = np.zeros(14)
labels[2] = 1
labels[8] = 1
labels[3] = 1
# labels[2] = 1
# 0-13 appear == 1 else == 0 (num_obj == 1, 2 or 3)
np.save('./surgical_tool_video/labels/' + filename + '_label.npy', labels)
