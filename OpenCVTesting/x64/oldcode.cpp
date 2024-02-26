//Using C++ version 98

#define _USE_MATH_DEFINES

#include <iostream>
#include <string>
#include <filesystem>
#include <ctime>
#include <vector>
#include <math.h>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dirent.h"

using namespace cv;
using namespace std;

//namespace fs = std::filesystem;

Mat img, imgGray, imgBlur, imgCanny, imgDil, imgErode, imgWarp;

template <typename T> class Vector2D
{
private:
    T x;
    T y;

public:
    explicit Vector2D(const T& x = 0, const T& y = 0) : x(x), y(y) {}
    Vector2D(const Vector2D<T>& src) : x(src.x), y(src.y) {}
    virtual ~Vector2D() {}

    // Accessors
    inline T X() const { return x; }
    inline T Y() const { return y; }
    inline T X(const T& x) { this->x = x; }
    inline T Y(const T& y) { this->y = y; }

    // Vector arithmetic
    inline Vector2D<T> operator-() const
    {
        return Vector2D<T>(-x, -y);
    }

    inline Vector2D<T> operator+() const
    {
        return Vector2D<T>(+x, +y);
    }

    inline Vector2D<T> operator+(const Vector2D<T>& v) const
    {
        return Vector2D<T>(x + v.x, y + v.y);
    }

    inline Vector2D<T> operator-(const Vector2D<T>& v) const
    {
        return Vector2D<T>(x - v.x, y - v.y);
    }

    inline Vector2D<T> operator*(const T& s) const
    {
        return Vector2D<T>(x * s, y * s);
    }

    // Dot product
    inline T operator*(const Vector2D<T>& v) const
    {
        return x * v.x + y * v.y;
    }

    // l-2 norm
    inline T norm() const { return sqrt(x * x + y * y); }

    // inner angle (radians)
    static T angle(const Vector2D<T>& v1, const Vector2D<T>& v2)
    {
        return acos((v1 * v2) / (v1.norm() * v2.norm()));
    }
};

Mat CropImage(Mat img)
{
    /*
    Get the relevant image bounds
    Crop the image
    */

    //Original crop
    int xMin = 890;
    int xMax = 1008;
    int yMin = 410;
    int yMax = 692;
    float boundsPercentage = 0.1;

    //Add a border to crop
    int xMinAdjusted = xMin - round((xMax - xMin) * boundsPercentage);
    int xMaxAdjusted = xMax + round((xMax - xMin) * boundsPercentage);
    int yMinAdjusted = yMin - round((yMax - yMin) * boundsPercentage);
    int yMaxAdjusted = yMax + round((yMax - yMin) * boundsPercentage);

    //Need to ensure these don't go outside of image bounds
    if (xMinAdjusted < 1) { xMinAdjusted = 0; }
    if (xMaxAdjusted > img.size().width) { xMinAdjusted = img.size().width; }
    if (yMinAdjusted < 1) { yMinAdjusted = 0; }
    if (yMaxAdjusted > img.size().height) { yMaxAdjusted = img.size().height; }

    Mat imgc = img(Range(yMinAdjusted, yMaxAdjusted), Range(xMinAdjusted, xMaxAdjusted));

    return imgc;
}


void GetContours(Mat imgDil, Mat img)
{


    vector<vector<Point>> contours;
    vector<Vec4i> heirarchy;

    findContours(imgDil, contours, heirarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

    vector<vector<Point>>conPoly(contours.size());


    for (int i = 0; i < contours.size(); i++)
    {
        int area = contourArea(contours[i]);
        //cout << area << endl;

        if (area > 3000.0f)
        {
            if (conPoly[i].size() < 50.0f)
            {
                float peri = arcLength(contours[i], true);
                approxPolyDP(contours[i], conPoly[i], 0.001 * peri, true);
                //drawContours(img, contours, i, Scalar(255, 0, 255), 4);
                //drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);
                drawContours(img, conPoly, i, Scalar(255, 0, 255), FILLED);
                cout << conPoly[i].size() << endl;
                //boundingRect(conPoly[i]);
            }
        }
    }

}

int main(int argc, const char** argv) {

    //Load in color
    img = imread("test3.jpg");

    if (__cplusplus == 202101L) std::cout << "C++23";
    else if (__cplusplus == 202002L) std::cout << "C++20";
    else if (__cplusplus == 201703L) std::cout << "C++17";
    else if (__cplusplus == 201402L) std::cout << "C++14";
    else if (__cplusplus == 201103L) std::cout << "C++11";
    else if (__cplusplus == 199711L) std::cout << "C++98";
    else std::cout << "pre-standard C++." << __cplusplus;
    std::cout << "\n";

    //Mat denoised;
    //fastNlMeansDenoisingColored(img, denoised, 30, 7, 21,10);


    namedWindow("Control", WINDOW_AUTOSIZE); //create a window called "Control"

    int iLowH = 0;
    int iHighH = 179;

    int iLowS = 0;
    int iHighS = 255;

    int iLowV = 0;
    int iHighV = 255;

    int iHoleFillSize = 1;
    //int iLowHoleFillSize = 100;

    //Create trackbars in "Control" window
    createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    createTrackbar("HighH", "Control", &iHighH, 179);

    createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    createTrackbar("HighS", "Control", &iHighS, 255);

    createTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
    createTrackbar("HighV", "Control", &iHighV, 255);

    createTrackbar("Hole fill size", "Control", &iHoleFillSize, 5); //Hole fill size (1 - 100)

    Point2f test1src[4] = { {892,411},{1006,411},{890,689},{1006,961} };
    Point2f test5src[4] = { {838,142},{1013,188},{826,810},{1006,799} };

    float goalHeight = 550;
    float goalWidth = 230;
    Point2f goalPostsrc[4] = { {0.0f,0.0f},{goalWidth,0.0f},{0.0f,goalHeight},{goalWidth,goalHeight} };

    Mat matrix = getPerspectiveTransform(test5src, goalPostsrc);
    warpPerspective(img, imgWarp, matrix, Point(goalWidth, goalHeight));

    //Mat imgc = CropImage(img);

    /*
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(img, imgBlur, Size(3, 3), 3, 0);
    Canny(imgBlur, imgCanny, 25, 100);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDil, kernel);
    GetContours(imgDil, img);

    */

    Mat imgHSV;

    //Image has been read as BGR
    cvtColor(img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

    Mat imgThresholded;
    Mat imgMask;

    //yellow
    //inRange(imgHSV, Scalar(35, 0, 0), Scalar(146, 155, 100), imgMask);
    //for test image3
    inRange(imgHSV, Scalar(23, 141, 85), Scalar(38, 255, 139), imgMask);


    //cvtColor(imgMask, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgMask, imgBlur, Size(3, 3), 3, 0);
    Canny(imgBlur, imgCanny, 25, 100);
    //Canny(imgMask, imgCanny, 25, 100);
    //Size will be based on zoom
    Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
    dilate(imgCanny, imgDil, kernel);
    imgMask = imgDil;

    /*
    erode(imgCanny, imgDil, kernel);
    dilate(imgDil, imgDil, kernel);

    //morphological closing (fill small holes in the foreground)
    dilate(imgDil, imgDil, kernel);
    erode(imgDil, imgDil, kernel);


    imgMask = imgDil;


    //Mat binary_image;
    //threshold(imgMask, binary_image, 100, 255, THRESH_BINARY);
    Mat invertedimgMask;
    bitwise_not(imgMask, invertedimgMask);
    imgMask = invertedimgMask;*/


    GetContours(imgMask, img);

    /*
    Mat dst, cdst, cdstP;
    dst = img;
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();

    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(imgMask, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    */

    while (1) {

        //white goalpost (assuming that we have a clear image with good lighting
        //inRange(imgHSV, Scalar(0, 0, 200), Scalar(255, 30, 255), imgMask);

        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

        //Range check
        if (iHoleFillSize < 1)
        {
            iHoleFillSize = 1;
        }

        //morphological opening (remove small objects from the foreground)
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(iHoleFillSize, iHoleFillSize)));
        dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(iHoleFillSize, iHoleFillSize)));

        //morphological closing (fill small holes in the foreground)
        dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(iHoleFillSize, iHoleFillSize)));
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(iHoleFillSize, iHoleFillSize)));


        //imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
        //imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);


        imshow("Thresholded Image", imgThresholded); //show the thresholded image
        imshow("Mask Image", imgMask); //show the thresholded image
        imshow("Original", img); //show the original image
        //imshow("FoundContours", imgDil); //show the original image
        //imshow("Image Warped", imgWarp); //show the original image

        if (waitKey(30) == 27) { //if esc is press break the loop//
            break;
        }
    }
    waitKey(0);

    destroyAllWindows();
}

void ProcessMultipleImages()
{
    //select the loading path
    string loadingPath = "D:\TestData\AFLGoalSequence6\*.*";
    string savingPath = "D:\OpenCVPython\PythonApplicationTest\PythonApplicationTest\MulitpleImageSaving6";

    //Number used for iterating file output
    int i = 0;

    struct dirent* dp;
    char* fullpath;
    const char* path = "C:\\test\\"; // Directory target
    DIR* dir = opendir(path); // Open the directory - dir contains a pointer to manage the dir
    while (dp = readdir(dir)) // if dp is null, there's no more content to read
    {
        fullpath = pathcat(path, dp->d_name);
        printf("%s\n", fullpath);

        time_t startTime = time(NULL);

        Mat image_read = imread("fullpath");

        img = ProcessGoalposts(image_read);

        imwrite(savingPath + "processedImage" + '_' + to_string(i) + ".png", img);

        time_t totalTime = time(NULL) - startTime;
        string totalTimeS = std::to_string(roundf(totalTime * 10000) / 10000);
        cout << "Total time taken to open and then process image: " + totalTimeS;

        free(fullpath);
        i += 1;

        waitKey(0);
    }
    closedir(dir); // close the handle (pointer)

}

Mat ProcessGoalposts(Mat toProcess)
{
    //process two images
    string testNumber = "14";

    //img = cv.imread(cv.samples.findFile(args.input1))

    //Image has been read as BGR
    //Convert the captured frame from BGR to HSV

    //Image has been read as BGR
    Mat imgHSV;
    cvtColor(img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV


    //inRange(imgHSV, Scalar(0, 0, 200), Scalar(255, 30, 255), imgMask);

    //isolate the white
    int sensitivity = 50;
    Scalar lower_white = Scalar(0, 0, 255 - sensitivity);
    Scalar upper_white = Scalar(255, sensitivity, 255);

    Mat imgMask;
    inRange(imgHSV, lower_white, upper_white, imgMask); //Threshold the image

    //Skeletonize
    Mat imgSkel = Skeletonize(imgMask);

    //imshow('Masked white', imgMask)

    Mat kernel = getStructuringElement(MORPH_RECT, Size(4, 4));
    //kernel = np.ones((4, 4), np.uint8)

    //imgBlur = GaussianBlur(imgSkel, (3, 3), 0)
    //imgCanny = Canny(imgSkel, 25, 100)

    Mat imgDil;
    dilate(imgSkel, imgDil, kernel, Point(-1, -1), 2, 1, 1);


    Mat final = ProbabilisticHoughLines(imgDil);

    //imgDil = convert(imghsv, imgMask)

    //get contours
    //contours, hierarchy = cv.findContours(imgDil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    //np_contours = np.array(contours, dtype = object)

    //print(len(np_contours))

    //comparison = cv.cvtColor(imgDil, cv.COLOR_GRAY2RGB)

    //areas = []
    //j = 0
    //while j < len(np_contours) :
    // cnt = np_contours[j]
    // area = cv.contourArea(cnt)

    // if area > 200:
    //   perimeter = cv.arcLength(cnt, True)
    //   epsilon = 0.001 * cv.arcLength(cnt, True)
    //   approx = cv.approxPolyDP(cnt, epsilon, True)
    //cv.drawContours(img, [approx], -1, (255, 0, 255), thickness = cv.FILLED)
    //   cv.drawContours(comparison, [approx], -1, (255, 0, 255), 8)

    //   areas.append(area)

    //j += 1

    //isolate the pink
    Scalar lower_pink = Scalar(0, 255, 0);
    Scalar upper_pink = Scalar(0, 255, 0);
    Mat extrafinal;

    inRange(final, lower_pink, upper_pink, extrafinal);
    imshow("Probablistic Hough Lines", extrafinal);

    //cv.waitKey()
    //Test(imgMask)
    return extrafinal;
}

Mat Skeletonize(Mat img)
{
    /*
    https://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
    Get all the lines as 1 pixel wide
    */

    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
    Mat skel(img.size(), CV_8UC1, cv::Scalar(0));

    // Repeat steps 2 - 4
    while (true) {

        //Step 2: Open the image
        Mat opening;
        morphologyEx(img, opening, MORPH_OPEN, element);

        //Step 3: Substract open from the original image
        Mat subtracted;
        subtract(img, subtracted, opening);

        //Step 4: Erode the original image and refine the skeleton
        Mat eroded;
        erode(img, eroded, element);

        bitwise_or(skel, skel, subtracted);

        img = eroded;

        // Step 5: If there are no white pixels left ie..the image has been completely eroded, quit the loop
        if (countNonZero(img) == 0)
        {
            break;
        }
    }
    return skel;
}

Mat ProbabilisticHoughLines(Mat img)
{
    cout << ("Finding probablistic hough lines...");
    Mat gray = img;

    Mat comparison = img;
    cvtColor(comparison, comparison, COLOR_GRAY2RGB);

    float rho = 0.1;
    float theta = M_PI / 180;
    int threshold = 30;

    int minLineLength = 40;
    int maxLineGap = 5;

    vector<Vec4i> lines;
    HoughLinesP(gray, lines, HOUGH_PROBABILISTIC, theta, threshold, minLineLength, maxLineGap);

    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        int x1 = l[0];
        int y1 = l[1];
        int x2 = l[2];
        int y2 = l[3];

        Vector2D<double> a;
        Vector2D<double> b;
        Vector2D<double> c;

        if (x1 > x2)
        {
            int tempx = x1;
            int tempy = y1;
            x1 = x2;
            y1 = y2;
            x2 = tempx;
            y2 = tempy;

            float straight = 0;

            if (x1 != x2)
            {

                //check the angle of line
                if (y1 == 0)
                {
                    a = Vector2D<double>(x1, y1);
                    b = Vector2D<double>(x2, y2);
                    c = Vector2D<double>(x2, 0);
                }
                else
                {
                    b = Vector2D<double>(x1, y1);
                    a = Vector2D<double>(x2, y2);
                    c = Vector2D<double>(x1, 0);
                }

                Vector2D<double> ba = a - b;
                Vector2D<double> bc = c - b;

                float cosine_angle = (ba * bc) / (ba.norm() * bc.norm());
                //float cosine_angle = dotProduct(ba, bc) / (norm(ba) * norm(bc));
                float angle = acos(cosine_angle);

                //angle = np.degrees(angle);
                //convert radians to degrees
                angle = angle * (180.0 / M_PI);

                float angledec = angle / 90;
                float anglewhole = angle; // 90

                angle = (angledec - anglewhole) * 90;
            }
            else
            {

                //print(np.degrees(angle))
                float angle = 0;
                float angleFilter = 10;
                float minimumPostLength = 200;
                float postLength;

                if ((y1 - y2) > 0)
                {
                    postLength = y1 - y2;
                }
                else
                {
                    postLength = y2 - y1;
                }

                if (postLength > minimumPostLength)
                {
                    //print(' degrees = ' + str(angle))

                    if ((angle < angleFilter) || (angle > 90 - angleFilter))
                    {
                        //if angle > 90:
                        // angle = 180 - angle
                        cout << ("Points : (" + std::to_string(x1) + ", " + std::to_string(y1) + "), (" + std::to_string(x2) + ", " + std::to_string(y2) + ") with angle : " + std::to_string(angle));

                        line(comparison, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2, LINE_AA);
                    }
                }
            }
        }
    }

    //cv.imshow('Probablistic Hough Lines', comparison)
    return comparison;
}


char* pathcat(const char* str1, char* str2)
{
    char* res;
    size_t strlen1 = strlen(str1);
    size_t strlen2 = strlen(str2);
    int i, j;
    //need to cast to assign to a void pointer
    res = (char*)(malloc((strlen1 + strlen2 + 1) * sizeof * res));
    strcpy(res, str1);

    for (i = strlen1, j = 0; ((i < (strlen1 + strlen2)) && (j < strlen2)); i++, j++)
    {
        res[i] = str2[j];
    }
    res[strlen1 + strlen2] = '\0';
    return res;
}