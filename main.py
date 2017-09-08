
# 24/7 CRAZZZZZZZZZZZZZZZZZZY
# import the necessary packages
import numpy as np
import sys
import cv2
import pyttsx
def getAngle(p0, p1=np.array([0,0]), p2=None):

    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    if angle < 0:
        angle = 360 + np.degrees(angle)
    else:
        angle = np.degrees(angle)
    return angle

def track(image):

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(image, (5,5),0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image for only green colors
    lower_green = np.array([50,100,100])
    upper_green = np.array([70,255,255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Blur the mask
    bmask = cv2.GaussianBlur(mask, (5,5),0)

    # Take the moments to get the centroid
    moments = cv2.moments(bmask)
    m00 = moments['m00']
    centroid_x, centroid_y = None, None
    if m00 != 0:
        centroid_x = int(moments['m10']/m00)
        centroid_y = int(moments['m01']/m00)

    # Assume no centroid
    ctr = (-1,-1)

    # Use centroid if it exists
    if centroid_x != None and centroid_y != None:

        ctr = (centroid_x, centroid_y)

        # Put black circle in at centroid in image
        cv2.circle(image, ctr, 10, (100,0,0), 5)

    # Display full-color image

    # Force image display, setting centroid to None on ESC key input
    if cv2.waitKey(1) & 0xFF == 27:
        ctr = None
    if(ctr == (-1, -1)):
        print ""
        print "There is no sufficient light. Please turn on the lights."
        print ""
        sys.exit(0)

    # Return coordinates of centroid
    return list(ctr)

number = None
colour = None
centerWheel = [360,238]
cap = cv2.VideoCapture(0)
for i in range(20):
    s, im = cap.read() # captures image
cv2.imwrite("test2.jpg",im) # writes image test.bmp to disk

img_rgb = cv2.imread('test2.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('justheball.bmp',0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.7
loc = np.where( res >= threshold)
count = 0
runningTotalY = 0
runningTotalX = 0

for pt in zip(*loc[::-1]):
    x1, y1 = pt
    x2, y2 = (pt[0] + w, pt[1] + h)
    cX = (x1 + x2) / 2
    cY = (y1 + y2) / 2
    runningTotalX += cX
    runningTotalY += cY

    count += 1


try:

    averageX = runningTotalX / count
    averageY = runningTotalY /  count
except(ZeroDivisionError):
    print ""
    print "Ball couldn't be found.Please re-spin the wheel."
    print ""
    sys.exit(0)


cv2.circle(img_rgb, (averageX, averageY), 7, (0, 0, 255), -1)
cv2.putText(img_rgb, "center", (averageX - 20, averageY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


print ""

cv2.imwrite('Detected.jpg',img_rgb)

print ""

finalAngle = getAngle(track(img_rgb), centerWheel, [averageX, averageY])


numbers = [32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
colour = ["Red", "Black"] * 18

finalNumber = None
finalColour = None

startRange = 4.864864864864865
endRange = 14.594594594594595

if (finalAngle >= 0 and finalAngle <= 4.864864864864865 and finalNumber == None) :
    finalNumber = 0
    finalColour = "Green"

for i in range(36) :
    if (finalAngle >= startRange and finalAngle <= endRange and finalNumber == None) :
        finalNumber = numbers[i]
        finalColour = colour[i]
        break
    startRange += 9.72972972972973
    endRange += 9.72972972972973

if (finalAngle >= 355.135135135135135 and finalAngle <= 360 and finalNumber == None) :
    finalNumber = 0
    finalColour = "Green"


print "Your number is " + str(finalNumber) + " " + str(finalColour)
print ""
engine = pyttsx.init()

engine.say("Your number is " + str(finalNumber) + " " + str(finalColour))
engine.runAndWait()


