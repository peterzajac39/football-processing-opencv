import cv2
import numpy as np
# script creates perspective transformation view from video

def resizeimage(frame):
    height, width, layers = frame.shape
    new_h = 270
    new_w = 430
    frame = cv2.resize(frame, (new_w, new_h))
    return frame


def getPerspectiveTransformation(frame):
    rows, cols, ch = frame.shape
    pts1 = np.float32([[60, 20], [380, 20], [0, 100], [430, 100]])
    pts2 = np.float32([[0, 0], [430, 0], [0, 270], [430, 270]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(frame, M, (430, 270))
    return dst

def getPerspectiveTransformationMiddle(frame):
    rows, cols, ch = frame.shape
    pts1 = np.float32([[30, 75], [400, 75], [0, 130], [430, 130]])
    pts2 = np.float32([[0, 0], [430, 0], [0, 270], [430, 270]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(frame, M, (430, 270))
    return dst


img = cv2.imread('images/ball.png')
img1 = cv2.imread('images/ball.png')

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1290, 270))

cap1 = cv2.VideoCapture('filmrole/filmrole1.avi')
cap2 = cv2.VideoCapture('filmrole/filmrole3.avi')
cap3 = cv2.VideoCapture('filmrole/filmrole5.avi')
cap4 = cv2.VideoCapture('filmrole/filmrole2.avi')
cap5= cv2.VideoCapture('filmrole/filmrole4.avi')
cap6 = cv2.VideoCapture('filmrole/filmrole6.avi')

vis = np.concatenate((img, img1), axis = 1)
cv2.imshow('out', vis)

while(1):
    ret1, frame1 = cap1.read()
    if ret1:
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        ret1, frame4 = cap4.read()
        ret2, frame5 = cap5.read()
        ret3, frame6 = cap6.read()

        frame1 = resizeimage(frame1)
        frame2 = resizeimage(frame2)
        frame3 = resizeimage(frame3)
        original = np.concatenate((frame3, frame2, frame1), axis=1)

        cv2.imshow('original frames merged', original)
        cv2.imwrite('merged_original.jpg', original);

        frame4 = resizeimage(frame4)
        frame5 = resizeimage(frame5)
        frame6 = resizeimage(frame6)

        frame3 = getPerspectiveTransformation(frame3)
        frame2 = getPerspectiveTransformation(frame2)
        frame1 = getPerspectiveTransformation(frame1)

        frame4 = getPerspectiveTransformation(frame4)
        frame5 = getPerspectiveTransformation(frame5)
        frame6 = getPerspectiveTransformation(frame6)

        frame5 = cv2.flip(frame5, 1)

        result = np.concatenate((frame3, frame2, frame1), axis=1)

        # totalresult = np.concatenate((result, result2), axis=0)
        cv2.imshow('result', result)
        cv2.imwrite('merged_transformed.jpg', result);

        # write the flipped frame
        out.write(result)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cap1.release()
cap2.release()
cap3.release()
out.release()
cv2.destroyAllWindows()
