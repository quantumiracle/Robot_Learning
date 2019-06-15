"""
Image Processing
uisng blob detection for testing one image
https://www.learnopencv.com/blob-detection-using-opencv-python-c/
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

def image_processing(img):
    ''' read image and transfer to color image '''
    # cv2.imshow('origin',img)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    ''' get threshold, for contour detection in next step '''
    ret,thresh_img = cv2.threshold(img,160,255,cv2.THRESH_TOZERO)  # lower than low-threshold to be black
    # cv2.imshow('Threshold', thresh_img)

    ''' get canny edge image, for the large circle detection with HoughCricles '''
    edge_detected_image = cv2.Canny(img, 60, 200)
    # cv2.imshow('Canny', edge_detected_image)

    ''' find image contours and plot '''
    image, contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cimg = cv2.drawContours(cimg, contours, -1, (0,0,255), 1)
    # cv2.imshow('Contour', cimg)

    contour_centers=[]  # detected contour centers list
    min_countour_size = 5  # minimal number of pixels for one contour, to exclude the very small wrong points
    for i in range (len(contours)):
        if len(contours[i])>min_countour_size:
            contour_centers.append(np.average(contours[i], axis = 0).reshape(2))  # reshape for reducing dimension

    return cimg, edge_detected_image, contour_centers


def large_circle_detect(cimg, edge_detected_image):
    ''' detect the big circle with houghcircles '''
    big_circle = cv2.HoughCircles(edge_detected_image,cv2.HOUGH_GRADIENT,2,30,  # dp, min_distance
                                param1=10,param2=10,minRadius=100,maxRadius=256)
    # pixel size of the image
    wid=len(cimg)
    hei=len(cimg[0])
    offset=50 # tolerance for offset of the center of big circle
    xshift=0  # x shift of center of large circle
    if big_circle is not None:
        big_circle = np.uint16(np.around(big_circle[0,:]))

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in big_circle:
            # draw the circle in the output image
            if x<wid*0.5+xshift+offset and x> wid*0.5+xshift-offset and y<hei*0.5+offset and y>hei*0.5-offset:
                cv2.circle(cimg, (x, y), r, (255, 0, 0), 4)
                cv2.circle(cimg,(x,y),4,(255,0,0),2)  # draw the center
                break
        # print('Radius: {}, Center: ({},{})'.format(r, x, y))
    return cimg

def contour_center_check(contour_centers, cimg, NUM_PINS=127):
    last_point=[[0,0]] 
    min_mse = 80 # minimum of mse for consecutive points


    for point in contour_centers:
        round_point=np.round(point).astype("int") # blob positions
        ''' if two consecutive points are too close, remove it'''
        mse = (np.square(round_point - last_point)).mean() 
        # print(mse)

        if mse>min_mse:  # larger distance than the distance lower bound counts one real point
            cv2.circle(cimg,(round_point[0],round_point[1]),1,(0,255,0),2)
        last_point = round_point
    # print('Num of pins: ', len(contour_centers))  # supposed to be 127
    if len(contour_centers)==NUM_PINS:
        VALID_DETECT=True
    else:
        VALID_DETECT=False
    return cimg, VALID_DETECT

def CenterRegister(centers):
    ''' register the pins centers to be in order: 7,8,9,10,11,12,13,12,11,10,9,8,7 '''
    num_list = [ 7,8,9,10,11,12,13,12,11,10,9,8,7]  # pins displacement, totally 127 pins
    contour_centers=np.array(centers)
    ordered_centers= [[0,0]] 
    for i in range(len(num_list)):
        indx=np.argsort(contour_centers[:,-1])[:num_list[i]]  # sorted on y-axis, select top-jth smallest, return index
        # print('ind: ', indx)
        y_contour = contour_centers[indx.tolist()]  # selected num_list[i] smallest y-axis centers
        x_contour = y_contour[y_contour[:, 0].argsort()]  # sorted on x-axis, return centers list
        # print('x: ', x_contour)
        ordered_centers=np.concatenate((ordered_centers, x_contour))
        contour_centers=np.delete(contour_centers, indx.tolist(), axis=0)  # delete sorted centers, deal with the rest
        # print('len: ', len(contour_centers))
    ordered_centers = ordered_centers[1:] # remove the first [0,0]
    # print(len(ordered_centers))

    return ordered_centers


if __name__ == '__main__':

    path = './img256f_r30'
    save_path = './img256f_r30con/'
    NUM_PINS=127
    # img_width=256
    # img_height=256
    pins_x=[]
    pins_y=[]
    for filename in os.listdir(path):
        print(filename) 
        img = cv2.imread(os.path.join(path,filename),0)
        cimg, edge_detected_image, contour_centers = image_processing(img)
        cimg = large_circle_detect(cimg, edge_detected_image)
        cimg, VALID_DETECT = contour_center_check(contour_centers, cimg, NUM_PINS=NUM_PINS)
        cv2.imwrite(save_path+str(filename),cimg)
        contour_centers = CenterRegister(contour_centers)


        if VALID_DETECT:  # pins detection correct
            reshape_contour_centers = np.array(contour_centers).transpose()
            pins_x.append(reshape_contour_centers[0])
            pins_y.append(reshape_contour_centers[1])


    reshape_pins_x = np.array(pins_x).transpose()
    plt.figure(1)
    for i in range(NUM_PINS):
        plt.subplot(211)
        plt.plot(np.arange(len(pins_x)), reshape_pins_x[i])
        plt.title('Position')
        plt.subplot(212)
        plt.plot(np.arange(len(pins_x)), reshape_pins_x[i]-reshape_pins_x[i][0])
        plt.title('Displacement')
    plt.savefig('./pins.png')
