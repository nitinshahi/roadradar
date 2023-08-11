import cv2
from speed_tracker import *
import numpy as np
end = 0

tracker = EuclideanDistTracker()
#cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture("night.mov")
# cap = cv2.VideoCapture("clip2.mp4")
# cap = cv2.VideoCapture("clip3.mp4")   `
cap = cv2.VideoCapture("clip4.mp4")
# cap = cv2.VideoCapture("traffic4.mp4")

fps = 60 #60 frames per second
wait_time = int(1000/(fps)) #1 second divide by fps


object_detector = cv2.createBackgroundSubtractorMOG2(history=None,varThreshold=None)

#KERNALS
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernal_e = np.ones((5,5),np.uint8)

while True:
    ret,frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5) # Resize the frame
    roi = frame[20:1720, 0:1980]

    fgmask = fgbg.apply(roi) #applies background substraction
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) #white greater than 200 pixel
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp) #morphologial operation to reduce noises
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    e_img = cv2.erode(mask2, kernal_e)


    contours,_ = cv2.findContours(e_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #THRESHOLD
        if area > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            # Draw bounding rectangle around the detected object
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h])

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id


        if(tracker.getsp(id)<tracker.limit()):
            cv2.putText(roi,str(id)+" "+str(tracker.getsp(id)),(x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255,0,0), 3)
        else:
            cv2.putText(roi,str(id)+ " "+str(tracker.getsp(id)),(x, y-15),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0,0,255), 3)

        s = tracker.getsp(id)
        if (tracker.f[id] == 1 and s != 0):
            tracker.capture(roi, x, y, h, w, s, id)
    # DrawingLINES
    cv2.line(roi, (0, 430), (960, 430), (0, 0, 0), 1)
    cv2.line(roi, (0, 255), (960, 255), (0, 0, 0), 1)
    #cv2.line(roi, (0, 590), (960, 590), (0, 0, 0), 1)
    #cv2.line(roi, (0, 255), (960, 255), (0, 0, 0), 1)


    cv2.imshow("Erode", e_img)
    cv2.imshow("ROI", roi)

    # Check for key press
    key = cv2.waitKey(wait_time-10)
    # If ESC key is pressed, end the tracker
    if key==27:
        tracker.end()
        end=1
        break
if(end!=1):
    tracker.end()
cap.release()
cv2.destroyAllWindows()