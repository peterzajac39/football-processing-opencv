import numpy as np
import cv2
# script tracks movement of the player by MeanShift algoritm and mask of players label
cap = cv2.VideoCapture('filmrole/filmrole3.avi')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')


# take first frame of the video
ret,frame = cap.read()

height, width, layers = frame.shape
new_h = int(height / 2)
new_w = int(width / 2)

frame = cv2.resize(frame, (new_w, new_h))

out = cv2.VideoWriter('tracking_ez.avi', fourcc, 30.0, (new_w, new_h))

# setup initial location of window
# r,h,c,w - region of image
#           simply hardcoded the values
# r,h,c,w = 240,50,240,30
# r,h,c,w = 255,60,330,40
# overlapping - very good
r,h,c,w = 220,50,435,30
track_window = (c,r,w,h)
lower_blue = np.array([32,0,224])
upper_blue = np.array([225,255,255])

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)

# maskpitch = cv2.inRange(hsv_roi, np.array((36., 0.,0.)), np.array((86.,255.,255.)))
# cv2.bitwise_not(mask, mask)
cv2.imshow('mask', mask)
cv2.imwrite('white_mask.jpg', mask)
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    height, width, layers = frame.shape
    new_h = int(height / 2)
    new_w = int(width / 2)

    frame = cv2.resize(frame, (new_w, new_h))

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        #print bounding box of tracking object
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
        cv2.imwrite('meanshift_track.jpg', img2)
        out.write(img2)
        k = cv2.waitKey(60) & 0xff
        if k == 32:
            cv2.waitKey(0)
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()
out.release()