from threading import Thread
import numpy as np
import imutils
import time
import cv2
import string

global imagecount
imagecount = 0


def vert_image_angle (y):
    vertical_angle_of_view = 48.8 # degrees
    vertical_resolution = 240 # pixels
    pix_per_degree = vertical_resolution / vertical_angle_of_view

    vertical_angle = (y / pix_per_degree)
    vertical_angle = np.radians(vertical_angle)
    

    return (vertical_angle)

def horiz_image_angle (x):
    horiz_angle_of_view = 62.2 # degrees
    horiz_resolution = 320 # pixels
    pix_per_degree = horiz_resolution / horiz_angle_of_view
    
    horiz_angle = np.radians(x / pix_per_degree)

    return (horiz_angle)

def new_bearing(locA, locB):

    na = locA[0]
    ea = locA[1]
    nb = locB[0]
    eb = locB[1]

    deltan = nb - na
    deltae = eb - ea

    alphaAB = np.arctan2(deltae,deltan)

    return (alphaAB)

def preparemask (hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper);
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask;

def meldmask (mask_0, mask_1):
    mask = cv2.bitwise_or(mask_0, mask_1)
    return mask;


def detectline3(vs, vehicle, height):

    global imagecount
    
    imagecount = imagecount + 1
    if imagecount > 99:
        imagecount  = 0

    xres = 320
    yres = 240
    xColour1 = xRed = 0.0

    coordA_Good = False
    red1Good = False
    coordB_Good = False
    red2Good = False

    coordA = [0,0]
    coordB = [0,0]
    xRed1 = xRed2 = 0
    yRed1 = yRed2 = 0                

    frame = vs.read()
    frame = imutils.resize(frame, width=320)

    roidepth = 20   
    roiymin = 40   
    roiymintop = roiymin - roidepth
    roiymax = yres - roidepth -1   
    sensitivity = 20

    lower_red_0 = np.array([0, 100, 100]) 
    upper_red_0 = np.array([sensitivity, 255, 255])
    
    lower_red_1 = np.array([180 - sensitivity, 100, 100]) 
    upper_red_1 = np.array([180, 255, 255])


    y3 = roiymax
    y4 = y3 + roidepth

    while y3 > roiymin:

        roihsv2 = frame[y3:y4, 0:(xres-1)]
        blurred2 = cv2.GaussianBlur(roihsv2, (11, 11), 0)
        roihsv2 = cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)
    

        maskr_2 = preparemask (roihsv2, lower_red_0 , upper_red_0)
        maskr_3 = preparemask (roihsv2, lower_red_1 , upper_red_1 )
        maskr2 = meldmask ( maskr_2, maskr_3)

        cnts_red2 = cv2.findContours(maskr2.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center2 = None

        if len(cnts_red2) > 0:

            c_red2 = max(cnts_red2, key=cv2.contourArea)
            ((x_red2, y_red2), radius_red2) = cv2.minEnclosingCircle(c_red2)
            M_red2 = cv2.moments(c_red2)

            cx_red2 = int(M_red2["m10"] / M_red2["m00"])
            cy_red2 = int(M_red2["m01"] / M_red2["m00"])
            

            cy_red2 = cy_red2 + y3

            if radius_red2 > 5:
                coordA_Good = True
                cv2.circle(frame, (cx_red2, cy_red2), int(radius_red2),
                (0, 0, 255), 2)                            

                xRed2 = cx_red2 - (xres/2) 
                yRed2 = (yres/2) - cy_red2 
                coordA = [xRed2, yRed2]

                break

        y3 = y3 - roidepth
        y4 = y3 + roidepth

    y1 = 0
    y2 = y1 + roidepth

    while y2 < y3: 

        roihsv1 = frame[y1:y2, 0:(xres-1)]
        blurred1 = cv2.GaussianBlur(roihsv1, (11, 11), 0)
        roihsv1 = cv2.cvtColor(blurred1, cv2.COLOR_BGR2HSV)

        maskr_0 = preparemask (roihsv1, lower_red_0 , upper_red_0)
        maskr_1 = preparemask (roihsv1, lower_red_1 , upper_red_1 )
        maskr1 = meldmask ( maskr_0, maskr_1)

        cnts_red1 = cv2.findContours(maskr1.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        center1 = None


        if len(cnts_red1) > 0:

            c_red1 = max(cnts_red1, key=cv2.contourArea)
            ((x_red1, y_red1), radius_red1) = cv2.minEnclosingCircle(c_red1)
            M_red1 = cv2.moments(c_red1)

            cx_red1 = int(M_red1["m10"] / M_red1["m00"])
            cy_red1 = int(M_red1["m01"] / M_red1["m00"])
            

            cy_red1 = cy_red1 + y1

            if radius_red1 > 5:
                coordB_Good = True
                cv2.circle(frame, (cx_red1, cy_red1), int(radius_red1),
                (0, 0, 255), 2)

                xRed1 = cx_red1 - (xres/2) #
                yRed1 = (yres/2) - cy_red1 
                coordB = [xRed1, yRed1]

                break

        y1 = y1 + roidepth
        y2 = y1 + roidepth


    cv2.line(frame, (0, y1), (xres, y1), (255,0,0))
    cv2.line(frame, (0, y2), (xres, y2), (255,0,0))
    cv2.line(frame, (0, y3), (xres, y3), (255,0,0))
    cv2.line(frame, (0, y4), (xres, y4), (255,0,0))

    
    if (coordA_Good == True) and (coordB_Good == True) :
        # Draw line to show bearing
        cv2.line(frame,(cx_red1,cy_red1),(cx_red2,cy_red2),(0,0,255),5)

    camera_pitch = np.radians(-45) 
    if coordA_Good == True:

        pitch_from_image = vert_image_angle(coordA[1])
        total_pitch = (camera_pitch + pitch_from_image) 
        total_pan = horiz_image_angle(coordA[0])
   
        mult = height / np.tan(total_pitch)
        coordA_north = mult * np.cos(total_pan)
        coordA_east = mult * np.sin(total_pan)
        locA = (coordA_north, coordA_east) 
    else:
        locA = (0,0)
        

    if coordB_Good == True:

        pitch_from_image = vert_image_angle(coordB[1])
        total_pitch = (camera_pitch + pitch_from_image) 

        total_pan = horiz_image_angle(coordB[0])
     
        mult = height / np.tan(total_pitch)
        coordB_north = mult * np.cos(total_pan)
        coordB_east = mult * np.sin(total_pan)
        locB = (coordB_north, coordB_east) 
    else:
        locB = (0,0)

    if coordA_Good == True and coordB_Good == False:
        locB = locA
        locA = (0,0)
        coordB_Good == True

    
    if coordA_Good == True and coordB_Good == True:     

        bearingAB = (new_bearing(locA,locB))
        angle = np.degrees(bearingAB)
        lineFound = True

    else:
        angle = 0.0
        lineFound = False

    offset = (2.0 * coordA[0]) / xres

    return frame, (angle, offset, lineFound)


def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True