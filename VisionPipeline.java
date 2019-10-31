package org.firstinspires.ftc.teamcode.vision;

import android.util.Log;

import org.firstinspires.ftc.teamcode.CSVWriter;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class VisionPipeline extends OpenCvPipeline{
    // THIS DETECTOR RETURNS THE PIXEL LOCATION OF THE LEFT MOST BOUNDARY OF THE BLACK TARGET
    // YOU CAN EASILY MODIFY IT TO GET YOU THE CENTER
    // IT IS YOUR JOB TO DETERMINE WHERE YOU WANT TO GO BASED ON THIS VALUE

    // SOMETIMES THERE IS SOME RANDOM CRASH THAT HAPPENS IF UR CAMERA WOBBLES A LOT
    // I AM LOOKING INTO A FIX, BUT AS LONG AS UR CAMERA DOESN"T FLAIL WILDLY IN MATCH U R GOOD

    // IF YOU SEE BUGS PLEASE MESSAGE ME ON DISCORD at Epsilon#0036
    // THIS IS NO MEANS ROBUST, BUT IT WORKS WELL FOR ME

    // THIS DETECTOR SACRIFICES SPEED FOR A LOT OF VERSATILITY
    // IT FUNCTIONS WITH LOTS OF NOISE BY PERFORMING LOTS OF FILTERS
    // IF YOU FEEL THAT SOME PARTS OF THIS PIPELINE ARENT NEEDED, THEN REMOVE THEM
    // TO IMPROVE FRAMERATE


    private final Scalar minHSV = new Scalar(11.8, 161.7, 116.5);
    private final Scalar maxHSV = new Scalar(30.3, 255.0, 255.0);

    private final Point anchor = new Point(-1,-1);
    private final int erodeIterations = 10;

    private final int dilateIterations = 20;

    // THESE NEED TO BE TUNED BASED ON YOUR DISTANCE FROM THE BLOCKS
    private final double minContourArea = 300.0;
    private final double minContourPerimeter = 1000.0;
    private final double minContourWidth = 300.0;
    private final double minContourHeight = 0.0;

    private final double cbMin = 105;
    private final double cbMax = 140;

    private int minX, minY = Integer.MAX_VALUE;
    private int maxX, maxY = -1 * Integer.MAX_VALUE;

    // TUNE THESE THEY WILL VARY BASED ON WEBCAM PLACEMENT!!!!!

    private final int maxVumarkValue = 80; // used to be 150
    private final int valleyLength = 40;

    private int vumarkLeftBoundary = -1;

    private Mat mask = new Mat();
    private Mat kernel = new Mat();


    private Mat hierarchy = new Mat();

    // private CSVWriter csvWriter = new CSVWriter(new File("colsums.java"));


    private final int INDEX_ERROR = -2; // index error code

    public Mat processFrame(Mat input) {
        Mat workingMat = input.clone();
        Imgproc.cvtColor(workingMat,workingMat,Imgproc.COLOR_RGB2HSV); // convert to HSV space

        Core.inRange(workingMat, minHSV, maxHSV, mask); // apply yellow filter
        Imgproc.erode(mask, mask, kernel, anchor, erodeIterations); // basically a faster blur
        Imgproc.dilate(mask, mask, kernel, anchor, dilateIterations); // remove noise

        List<MatOfPoint> stoneContours = new ArrayList<>();
        Imgproc.findContours(mask, stoneContours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE); // find block contours

        // remove noisy contours
        List<MatOfPoint> outputContours = new ArrayList<>();
        filterContours(stoneContours,outputContours, minContourArea, minContourPerimeter, minContourWidth, minContourHeight);

        workingMat.release();

        // draw stones (but not skystones)
        if (minX > 1e5 || maxX < 0 || minY > 1e5 || maxY < 0) return input;

        Rect r = new Rect(new Point(minX, minY + 150), new Point(maxX,  maxY));
        Imgproc.rectangle(input, r, new Scalar(0,0,255));

        // crop vertically

        Mat cbMat = crop(input.clone(),new Point(0, minY + 150), new Point(input.width() - 1,  maxY));
        // input = crop(input,new Point(0, minY + 150), new Point(input.width() - 3,  maxY));

        Imgproc.cvtColor(cbMat,cbMat,Imgproc.COLOR_RGB2YCrCb); // convert to ycrcb
        Core.extractChannel(cbMat, cbMat, 2); // extract cb channel

        Imgproc.threshold(cbMat, cbMat, cbMin, cbMax, Imgproc.THRESH_BINARY_INV); // binary mask to find the vumark

        List<MatOfPoint> stoneContours2 = new ArrayList<>();
        Imgproc.findContours(cbMat, stoneContours2, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE); // draw contours around the stones
        List<MatOfPoint> finalOutputContours = new ArrayList<>();
        resetRectangle();
        // build the max bounding rect
        filterContours(stoneContours2,finalOutputContours, minContourArea, minContourPerimeter, minContourWidth, minContourHeight);
        if (finalOutputContours.size() == 0) return input;

        // crop so we only see the stones
        Rect allStonesRect = getMaxRectangle();
        Mat cbCrop = crop(cbMat, allStonesRect);

        // crop so we only have the stones in the image
        // Imgproc.drawContours(input, finalOutputContours, -1, new Scalar(0,0,255), 10);

        Rect oBoundingRect = getMaxRectangle();
        Rect correctedRect = new Rect(new Point(oBoundingRect.x + minX, r.y), new Point(oBoundingRect.x +oBoundingRect.width + minX, r.y + r.height));
        //Imgproc.rectangle(input, correctedRect, new Scalar(0,255,0), 8);


        vumarkLeftBoundary = getMaxDropoff(cbCrop);
        Log.d("VUMARK: ", vumarkLeftBoundary+"");

        // this just estimates where the marker is, dont rely on this
        Imgproc.circle(input,new Point(vumarkLeftBoundary + 200, input.height() / 2), 60, new Scalar(255,0,0), 10);
        resetRectangle();

        /*
        if (vumarkLeftBoundary < (2 * input.width() / 7))
            Log.d("DIR: ", "left");
        else if (vumarkLeftBoundary > ( 5 * input.width() / 7))
            Log.d("DIR: ", "right");
        else
            Log.d("DIR: ", "center");
         */
        return input;
    }

    private void filterContours(List<MatOfPoint> contours, List<MatOfPoint> outputContours, double minContourArea, double minContourPerimeter, double minContourWidth,
    double minContourHeight) {
        //resetRectangle();
        Log.d("NumContours", contours.size() + "");
        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);
            int x = rect.x;
            int y = rect.y;
            int w = rect.width;
            int h = rect.height;

            if (w < minContourWidth)
                continue;
            if (rect.area() < minContourArea)
                continue;
            if ((2 * w + 2 * h) < minContourPerimeter)
                continue;
            if (h < minContourHeight)
                continue;
            outputContours.add(contour);

            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x + w > maxX) maxX = x + w;
            if (y + h> maxY) maxY = y + h;
        }

    }

    private Mat crop(Mat image, Point topLeftCorner, Point bottomRightCorner) {
        Rect cropRect = new Rect(topLeftCorner, bottomRightCorner);
        return new Mat(image, cropRect);
    }

    private Mat crop(Mat image, Rect rect) {
        return new Mat(image, rect);
    }

    private int getMaxDropoff(Mat image) {

        Mat colsums = new Mat();
        Core.reduce(image, colsums, 0, Core.REDUCE_SUM, 4);

        colsums.convertTo(colsums, CvType.CV_32S);

        int[] colsumArray = new int[(int)(colsums.total()*colsums.channels())];
        colsums.get(0,0,colsumArray);
        for (int i = 0; i < colsumArray.length; i++) {
            colsumArray[i] /= 140;
        }


        for (int i = 0; i < colsumArray.length; i++) {

            if (colsumArray[i] < maxVumarkValue) {
                // Log.d("colsum", i+"");
                if (i + valleyLength > colsumArray.length - 1) return INDEX_ERROR;
                int[] slice = Arrays.copyOfRange(colsumArray, i, i+valleyLength);
                if (isLargeValley(slice, maxVumarkValue, 25)) {

                    return i;
                }
            }
        }
        return -1;
    }

    private boolean isLargeValley(int[] slice, double maxValue, double thresh) {
        List<Integer> differences = new ArrayList<>(slice.length - 1);
        for (int i : slice) {
            if (i > maxValue) return false;

        }
        for (int i = 0; i < slice.length - 1; i++)
            differences.add(Math.abs(slice[i+1] - slice[i]));

        return Collections.max(differences) < thresh;
    }

    public int getVumarkLeftBoundary() { return vumarkLeftBoundary; }

    private void resetRectangle() {
        maxX = maxY = -1 * (int) 1e8;
        minX = minY = (int) 1e8;
    }

    private Rect getMaxRectangle() { return new Rect(new Point(minX, minY), new Point(maxX,maxY)); }

}
