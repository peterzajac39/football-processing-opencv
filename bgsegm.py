import numpy as np
import cv2
# script segments and detects players and referees in video

# function filters contours and removes small and big ones and returns those which represent player
def filtercontours(contours):
    playercontours = list()
    for c in contours:
        rect = cv2.boundingRect(c)
        # if contour is too big or too small its not player
        if (rect[2] < 7 or rect[3] < 20) or (rect[2] > 60 or rect[3] > 100): continue
        playercontours.append(c)
    return playercontours


# function classifies contours to 2 lists based on color
def classifycontours(contours):
    classifiedObjects = {}
    ateamplayers = list()
    bteamplayers = list()

    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        crop_img = bigFrame[y:y + h, x:x + w]
        # compute mean color with mask of background for better reults
        meanColor = cv2.mean(crop_img, mask =mask)
        # comparison of Green color range is threshold for labels
        if meanColor[1] > 100:
            ateamplayers.append(c)
        else:
            bteamplayers.append(c)

    classifiedObjects['ateam'] = ateamplayers
    classifiedObjects['bteam'] = bteamplayers
    return classifiedObjects


cap = cv2.VideoCapture('filmrole/filmrole3.avi')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernelBig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

ret,testFrame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

height, width, layers = testFrame.shape
new_h = int(height / 2)
new_w = int(width / 2)

frame = cv2.resize(testFrame, (new_w, new_h))
out = cv2.VideoWriter('segmentation.avi', fourcc, 20.0, (new_w, new_h))

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
fgbg1 = cv2.bgsegm.createBackgroundSubtractorCNT(isParallel=True)

# define range of green color in HSV
lower_green = np.array([36, 0, 0])
upper_green = np.array([86, 255, 255])
# define range of orange color in HSV
ORANGE_MIN = np.array([10, 0, 0], np.uint8)
ORANGE_MAX = np.array([205, 255, 255], np.uint8)

# read video by frames
while(1):
    ret, bigFrame = cap.read()

    height, width, layers = bigFrame.shape
    new_h = int(height / 2)
    new_w = int(width / 2)

    frame = cv2.resize(bigFrame, (new_w, new_h))
    bigFrame=frame

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # create mask for referees orange color
    refMask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
    cv2.bitwise_not(refMask, refMask)

    refMask = cv2.erode(refMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations=1)
    refMask = cv2.dilate(refMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Bitwise-AND mask and original image
    cv2.bitwise_not(mask, mask)

    # fg mask from backg substraction
    fgmask = fgbg.apply(frame)

    # closing
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # opening
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)


    clonedFrame = fgmask

    ret, thresh1 = cv2.threshold(clonedFrame, 100, 255, cv2.THRESH_BINARY)

    fgmask = cv2.bitwise_and(fgmask, mask)
    reffgMask= cv2.bitwise_and(fgmask, refMask)
    # create mask where are no referees
    notrefmask = cv2.bitwise_not(reffgMask)
    cv2.imwrite('ref_ext.jpg', reffgMask)
    # remove from mask referees
    fgmask = cv2.bitwise_and(fgmask, notrefmask)

    # find contours for referees and players
    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im2, refcontours, hierarchy = cv2.findContours(reffgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colorArray = []



    contours = filtercontours(contours)
    refcontours = filtercontours(refcontours)
    dict = classifycontours(contours)

    # print out contours
    for c in dict['ateam']:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(bigFrame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    for c in dict['bteam']:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(bigFrame, (x, y), (x + w, y + h), (0, 0, 0), 2)

    for c in refcontours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        crop_img = bigFrame[y:y + h, x:x + w]
        cv2.rectangle(bigFrame, (x, y), (x + w, y + h), (0,0,255), 2)

    # unused kmeans algoritm
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 2
    # data = np.array(colorArray)
    # data = np.float32(data)
    # if data.shape[0] >= K:
    #     ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #     print(label)
    #     i = 0
    #     for c in contours:
    #         rect = cv2.boundingRect(c)
    #         if rect[2] < 10 or rect[3] < 10: continue
    #         x, y, w, h = rect
    #         color = ()
    #         if label[i][0] == 1:
    #             color = (int(center[label[i][0]][0]), int(center[label[i][0]][1]),int(center[label[i][0]][2]))
    #         else:
    #             color = (int(center[label[i][0]][0]), int(center[label[i][0]][1]), int(center[label[i][0]][2]))
    #         cv2.rectangle(bigFrame, (x, y), (x + w, y + h), color, 2)
    #         i = i + 1

    cv2.imshow('frame',fgmask)
    cv2.imwrite("mean_col.jpg", bigFrame)
    cv2.imshow("original", bigFrame)
    # cv2.imshow("hsv", notrefmask)
    k = cv2.waitKey(30) & 0xff
    if k == 32:
        cv2.waitKey(0)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
out.release()
