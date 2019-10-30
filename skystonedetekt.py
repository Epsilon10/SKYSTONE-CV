import cv2
import numpy as np
import math
from enum import Enum
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

import timeit

IMAGE_PATH = "test_image.png"

class VisionPipeline:
    """
    """
    def __init__(self, show_plot=False):
        """initializes all values to presets or None if need to be set
        """

        self.show_plot = show_plot

        self.__hsv_threshold_hue = [11.864406779661016, 30.254668930390498]
        self.__hsv_threshold_saturation = [161.73510344266967, 255.0]
        self.__hsv_threshold_value = [116.51958094541313, 255.0]

        self.hsv_threshold_output = None


        self.__cv_erode_src = self.hsv_threshold_output
        self.__cv_erode_kernel = None
        self.__cv_erode_anchor = (-1, -1)
        self.__cv_erode_iterations = 10.0
        self.__cv_erode_bordertype = cv2.BORDER_CONSTANT
        self.__cv_erode_bordervalue = (-1)

        self.cv_erode_output = None

        self.__cv_dilate_src = self.cv_erode_output
        self.__cv_dilate_kernel = None
        self.__cv_dilate_anchor = (-1, -1)
        self.__cv_dilate_iterations = 20.0
        self.__cv_dilate_bordertype = cv2.BORDER_CONSTANT
        self.__cv_dilate_bordervalue = (-1)

        self.cv_dilate_output = None

        self.__find_contours_input = self.cv_dilate_output
        self.__find_contours_external_only = False

        self.find_contours_output = None

        self.__filter_contours_contours = self.find_contours_output
        self.__filter_contours_min_area = 300.0
        self.__filter_contours_min_perimeter = 1000.0
        self.__filter_contours_min_width = 500.0
        self.__filter_contours_max_width = 300000.0
        self.__filter_contours_min_height = 0.0
        self.__filter_contours_max_height = 1000.0
        self.__filter_contours_solidity = [0, 100]
        self.__filter_contours_max_vertices = 1000000.0
        self.__filter_contours_min_vertices = 0.0
        self.__filter_contours_min_ratio = 0.0
        self.__filter_contours_max_ratio = 1000.0

        self.filter_contours_output = None

        self.x_min = self.y_min = int(1e10)
        self.x_max = self.y_max = int(-1.0 * 1e10)

        self.rects = []

    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step HSV_Threshold0:
        self.__hsv_threshold_input = source0
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step CV_erode0:
        self.__cv_erode_src = self.hsv_threshold_output
        (self.cv_erode_output) = self.__cv_erode(self.__cv_erode_src, self.__cv_erode_kernel, self.__cv_erode_anchor, self.__cv_erode_iterations, self.__cv_erode_bordertype, self.__cv_erode_bordervalue)

        # Step CV_dilate0:
        self.__cv_dilate_src = self.cv_erode_output
        (self.cv_dilate_output) = self.__cv_dilate(self.__cv_dilate_src, self.__cv_dilate_kernel, self.__cv_dilate_anchor, self.__cv_dilate_iterations, self.__cv_dilate_bordertype, self.__cv_dilate_bordervalue)

        # Step Find_Contours0:
        self.__find_contours_input = self.cv_dilate_output
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)

        # Step Filter_Contours0:
        self.__filter_contours_contours = self.find_contours_output
        (self.filter_contours_output) = self.__filter_contours(self.__filter_contours_contours, self.__filter_contours_min_area, self.__filter_contours_min_perimeter, self.__filter_contours_min_width, self.__filter_contours_max_width, self.__filter_contours_min_height, self.__filter_contours_max_height, self.__filter_contours_solidity, self.__filter_contours_max_vertices, self.__filter_contours_min_vertices, self.__filter_contours_min_ratio, self.__filter_contours_max_ratio)
        """
        if self.x_min < 1e5:
            print(self.x_min, self.x_max, self.y_min, self.y_max, "hi")
            cv2.rectangle(source0, (self.x_min,self.y_min + 150), (self.x_max, self.y_max), (255,0,0), 2)
        """
        print(type(self.filter_contours_output))

        self.detector_pos = None
        
        
        for c in self.filter_contours_output:
            x,y,w,h = cv2.boundingRect(c)
            center_x = (x + (x + w)) / 2
            center_y = (y + (y+h)) / 2 
            self.rects.append((center_x, center_y))
            print(x,y,w,h)
            #cv2.rectangle(self.source0,(x,y),(x+w,y+h),(0,255,0),2)
        self.source0 = source0[self.y_min + 150:self.y_max + 60, :]
        self.get_orientation()
        print(self.detector_pos)
        #self.source0 = self.source0[self.y_min + 150:self.y_max, :]
        
        self.ycrcb = cv2.cvtColor(self.source0, cv2.COLOR_BGR2YCrCb)

        y, cr, cb = cv2.split(self.ycrcb)
        self.cb = cb
        """
        plt.matshow(self.cb)
        plt.colorbar()
        plt.show()
        """
        
        ret,thresh1 = cv2.threshold(self.cb,115,140,cv2.THRESH_BINARY_INV)
        self.thresh1 = thresh1
        
        self.__find_contours_input = self.thresh1
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)
        self.reset_rectangle()
        self.__filter_contours_contours = self.find_contours_output
        (self.filter_contours_output) = self.__filter_contours(self.__filter_contours_contours, self.__filter_contours_min_area, self.__filter_contours_min_perimeter, self.__filter_contours_min_width, self.__filter_contours_max_width, self.__filter_contours_min_height, self.__filter_contours_max_height, self.__filter_contours_solidity, self.__filter_contours_max_vertices, self.__filter_contours_min_vertices, self.__filter_contours_min_ratio, self.__filter_contours_max_ratio)        

        cv2.drawContours(self.source0, self.filter_contours_output, -1, (0, 255, 0), 3) 
        
        x,y,w,h = cv2.boundingRect(self.filter_contours_output[0])
        # cv2.rectangle(self.source0,(x,y),(x+w,y+h),(0,255,0),2)
        self.thresh1 = self.thresh1[self.y_min:self.y_max, self.x_min:self.x_max]
        #ret,thresh1 = cv2.threshold(self.cb,105,140,cv2.THRESH_BINARY)
        #self.thresh1 = thresh1

        x=self.get_max_dropoff_indices()
        print(x)

    def __hsv_threshold(self, input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.x_min, x_max, y_min, y_max)
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturatix_min, x_max, y_min, y_max)
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    def __cv_erode(self, src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of lower value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for erosion. A numpy.ndarray.
           iterations: the number of times to erode.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after erosion.
        """
        return cv2.erode(src, kernel, anchor, iterations = (int) (iterations +0.5),borderType = border_type, borderValue = border_value)

    def __cv_dilate(self, src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of higher value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for dilation. A numpy.ndarray.
           iterations: the number of times to dilate.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after dilation.
        """
        print('chump')
        return cv2.dilate(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)

    def __find_contours(self, input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if(external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        contours, hierarchy =cv2.findContours(input, mode=mode, method=method)
        return contours

    def __filter_contours(self, input_contours, min_area, min_perimeter, min_width, max_width,
                        min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                        min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        output = []
        # self.reset_rectangle()
        for contour in input_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if (w < min_width or w > max_width):
                continue
            if (h < min_height or h > max_height):
                continue
            area = cv2.contourArea(contour)
            if (area < min_area):
                continue
            if (cv2.arcLength(contour, True) < min_perimeter):
                continue
            hull = cv2.convexHull(contour)
            solid = 100 * area / cv2.contourArea(hull)
            if (solid < solidity[0] or solid > solidity[1]):
                continue
            if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                continue
            ratio = (float)(w) / h
            if (ratio < min_ratio or ratio > max_ratio):
                continue
            output.append(contour)
            self.x_min = x if x < self.x_min else self.x_min
            self.y_min = y if y < self.y_min else self.y_min 
            self.x_max = x + w if x + w > self.x_max else self.x_max 
            self.y_max = y + h if y + h > self.y_max else self.y_max
        return output
    
    def get_orientation(self):
        print(self.rects)
        source_len = len(self.source0[0])
        print(self.rects[0][0] / source_len)

    def get_max_dropoff_indices(self):
        colsums = (self.thresh1 / 140).sum(axis=0)
        if (self.show_plot)
            plt.plot(np.arange(len(self.thresh1[0])), colsums)
        #plt.plot(np.arange(len(self.thresh1[0])), np.gradient(colsums))
            plt.show()

        for i in range(len(colsums)):
            if colsums[i] < 150:
                if i+100 > len(colsums):
                    break
                if self.is_large_peak(colsums[i:i+100]):
                    cv2.circle(self.thresh1, (i + 250, len(self.source0) // 2),100 ,(0,0,255), 10)
                    return i
        return 0

    def is_large_peak(self, slice, base=150, thresh=25):
        for val in slice:
            if val > base:
                return False
        return np.amax(np.abs(np.diff(slice))) < thresh
    
    def reset_rectangle(self):
        self.x_min = self.y_min = int(1e8)
        self.y_max = self.x_max = int(-1.0 * 1e8)
        

pipeline = VisionPipeline()
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.namedWindow('o', cv2.WINDOW_NORMAL)
cv2.resizeWindow('o', 600,600)
cv2.namedWindow('cb', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cb', 600,600)


im = cv2.imread(IMAGE_PATH)
pipeline.process(im)
cv2.imshow('o', pipeline.source0)
cv2.imshow('cb', pipeline.cb)
cv2.imshow('image', pipeline.thresh1)
cv2.waitKey(0)

#cap.release()
cv2.destroyAllWindows()


