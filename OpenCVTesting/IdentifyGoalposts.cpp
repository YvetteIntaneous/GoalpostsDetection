#include "IdentifyGoalposts.h"

using namespace cv;
using namespace std;

std::list<FieldData> fieldData;
std::list<RecordingData> recordData;
std::list<string> imageNames;

//Screenspace points
vector<vector<int>> foundPoints;
vector<vector<int>> generatedPoints;

//Supplied goalpost points in 3D space`
vector<vector<float>> goalpostPoints;

//For iterating through camera data frames in prerecorded data
int currentCameraFrame = 0;
RecordingData currentCameraData;

// Testing mode 0 = readomg from realtime file
// Testing mode 1 = reading from premade file, moving camera, generating gl, image manipulation
// testing mode 2 = reading from premade file. only moving camera and generating GL
//Testing mode 3 = single GL frame
int testingMode = 2;
int dilateGL = 10;


/*
void ComparePoints(vector<vector<int>> points1, vector<vector<int>> points2)
{
    for (int i = 0; i < points1.size(); i++)
    {
        for (int j = 0; j < points2.size(); j++)
        {
            points1[]
        }
    }
}
*/
/*Reading input data*/
void ReadDataField(string filePath)
{
    //read data
    io::CSVReader<5> in(filePath);
    std::cout << "Reading field data file..." << std::endl;

    // identfier as an integer, x, y, z, identifier as a string point
    std::string intIdentifier;
    double xPos;
    double yPos;
    double zPos;
    std::string pointIdentifier;

    //iterator for debugging
    while (in.read_row(intIdentifier, xPos, yPos, zPos, pointIdentifier))
    {
        FieldData entry;
        entry.intIdentifier = intIdentifier;
        entry.xPos = xPos;
        entry.yPos = yPos;
        entry.zPos = zPos;
        entry.pointIdentifier = pointIdentifier;

        fieldData.push_back(entry);
    }
}

void ReadImageNames(const char* imagesPath)
{
    struct dirent* dp;

    std::string pathString;
    std::string fileString;
    std::string fullpaths;

    const char* path = imagesPath; // Directory target
    DIR* dir = opendir(path); // Open the directory - dir contains a pointer to manage the dir

    while (dp = readdir(dir)) // if dp is null, there's no more content to read
    {
        pathString = path;
        fileString = dp->d_name;

        //Checking if we have a valid file... Should possibly do this for specific file types?
        if (fileString.length() > 4)
        {
            fullpaths = pathString + "\\" + fileString;
            imageNames.push_back(fullpaths);
        }
    }
    closedir(dir); // close the handle (pointer)
}

void ReadData(string filePath)
{
    //read data
    io::CSVReader<18> in(filePath);
    std::cout << "Reading camera data file..." << std::endl;
    //in.read_header(io::ignore_no_column, "epochTime", "timeStamp", "roll", "pitch", "yaw", "xPos", "yPos", "zPos", "focalLength");

    // Epoch Time, Timestamp, R, P, Y, X, Y, Z, Focal Length, K1, K2, K3, P1, P2, Fx, Fy, Cx, Cy
    double epochTime;
    std::string timeStamp;
    double roll;
    double pitch;
    double yaw;
    double xPos;
    double yPos;
    double zPos;
    double focalLength;
    double k1;
    double k2;
    double k3;
    double p1;
    double p2;
    // Camera focal lengths
    double fx;
    double fy;
    // Optical centres
    double cx;
    double cy;

    float imageWidth = 1920;
    float imageHeight = 1080;

    //creeate a list
    //std::list<RecordingData> recordData;

    //iterator for debugging
    int i = 0;
    while (in.read_row(epochTime, timeStamp, roll, pitch, yaw, xPos, yPos, zPos, focalLength, k1, k2, k3, p1, p2, fx, fy, cx, cy))
    {
        i++;

        // do stuff with the data
        //std::cout << "Row test " + std::to_string(i) + " " + timeStamp << std::endl;

        RecordingData entry;
        entry.epochTime = epochTime;
        entry.timeStamp = timeStamp;
        entry.roll = roll;
        entry.pitch = pitch;
        entry.yaw = yaw;
        entry.xPos = xPos;
        entry.yPos = yPos;
        entry.zPos = zPos;
        entry.focalLength = focalLength;
        entry.k1 = k1;
        entry.k2 = k2;
        entry.k3 = k3;
        entry.p1 = p1;
        entry.p2 = p2;
        //https://answers.opencv.org/question/17076/conversion-focal-distance-from-mm-to-pixels/
        entry.fx = fx * imageWidth;
        entry.fy = fy * imageHeight;
        entry.cx = cx * imageWidth;
        entry.cy = cy * imageHeight;

        recordData.push_back(entry);

    }
    //std::cout << "Row test: " << i << std::endl;
}

/* Goalpost generation functions*/


void AddGoalpostPoints()
{
    ReadDataField("D:\\Testing\\OpenCVTesting\\OpenCVTesting\\231123_puntroad.csv");
}


FieldData GetPostDataEntry(int i)
{
    list<FieldData>::iterator itr = fieldData.begin();
    std::advance(itr, i);
    FieldData foundEntry = *itr;

    return foundEntry;

}

vector<double> MakePoint(double x, double y, double z)
{
    vector<double> point;

    point.at(0) = x;
    point.at(1) = y;
    point.at(2) = z;

    return point;
}

void AdjustPositionsForOpenGL(vector<vector<double>> postPoints)
{
    double tempY;
    double tempZ;

    //Open GL swaps the Y and Z (Y refers to vertical dimension in OpenGL)
    for (int i = 0; i < 8; i++)
    {
        tempY = postPoints.at(i).at(1);
        tempZ = postPoints.at(i).at(2);

        postPoints.at(i) = MakePoint(postPoints.at(i).at(0), postPoints.at(i).at(2), postPoints.at(i).at(1));
    }
}

int FindClosestPointMatch(int j)
{
    //FInds the closest point to the specified point we provide

    double calculatedDistance = 0;
    double shortestDistance = 1000;
    int point = 0;

    for (int i = 0; i < foundPoints.size(); i++)
    {
        calculatedDistance = sqrt(pow(foundPoints[j].at(0) - foundPoints[i].at(0), 2) + pow(foundPoints[j].at(1) - foundPoints[i].at(1), 2) * 1.0);

        if (calculatedDistance < shortestDistance)
        {
            shortestDistance = calculatedDistance;
            point = i;
        }
    }

    return point;
}

Mat ProbabilisticHoughLines(Mat imgToUse)
{
    std::cout << "  Finding probablistic hough lines..." << std::endl;
    Mat gray = imgToUse;

    Mat comparison = imgToUse;
    cvtColor(comparison, comparison, COLOR_GRAY2RGB);

    double rho = 0.1;
    double theta = M_PI / 180;
    int threshold = 30;

    int minimumDetectionLength = 50;
    int minimumPostLength = 150;
    int maxLineGap = 10;

    vector<Vec4i> lines;
    vector<Vec4i> validLines;
    HoughLinesP(gray, lines, HOUGH_PROBABILISTIC, theta, threshold, minimumDetectionLength, maxLineGap);

    int x1;
    int y1;
    int x2;
    int y2;
    int tempx;
    int tempy;

    double angle;
    //double angledec;
    //double anglewhole;

    for (int i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i]; 
        x1 = l[0];
        y1 = l[1];
        x2 = l[2];
        y2 = l[3];

        Vector2D<double> a;
        Vector2D<double> b;
        Vector2D<double> c;

        if (x1 > x2)
        {
            tempx = x1;
            tempy = y1;
            x1 = x2;
            y1 = y2;
            x2 = tempx;
            y2 = tempy;
        }

        double straight = 0;

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

            double cosine_angle = (ba * bc) / (ba.norm() * bc.norm());
            //float cosine_angle = dotProduct(ba, bc) / (norm(ba) * norm(bc));
            angle = acos(cosine_angle);

            //angle = np.degrees(angle);
            //convert radians to degrees
            angle = angle * (180.0 / M_PI);

            //Making sure we only get a value between 0 and 90 for angle
            if (angle < 0)
            {
                angle = angle * -1;
            }

            while (angle > 90)
            {
                angle -= 90;
            }
        }
        else
        {
            angle = 0;
        }
            
        //cout << to_string(angle);
        double angleFilter = 10;
        int postLength;

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
                //cout << ("Points : (" + std::to_string(x1) + ", " + std::to_string(y1) + "), (" + std::to_string(x2) + ", " + std::to_string(y2) + ") with angle : " + std::to_string(angle));

                line(comparison, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2, LINE_AA);

            }
        }             
    }

    //Compare all the lines, then draw
    
    //imshow("Probablistic Hough Lines", comparison); 
    //waitKey();
    return comparison;
}

Mat Mask(Mat imageToMask, Mat mask)
{
    Mat maskedImage;

    imageToMask.copyTo(maskedImage, mask);

    return maskedImage;
}

Mat Skeletonize(Mat imgToUse)
{
    /*
    https://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
    Get all the lines as 1 pixel wide
    */

    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
    Mat skel(imgToUse.size(), CV_8UC1, cv::Scalar(0));
    Mat opening;
    Mat subtracted;
    Mat eroded;
    
    int i = 0;

    // Repeat steps 2 - 4
    while (true)
    {
        //Step 2: Open the image
        morphologyEx(imgToUse, opening, MORPH_OPEN, element);

        //Step 3: Substract open from the original image
        subtract(imgToUse, opening, subtracted);

        //Step 4: Erode the original image and refine the skeleton
        erode(imgToUse, eroded, element);

        bitwise_or(skel, subtracted, skel);

        eroded.copyTo(imgToUse);

        i++;

        // Step 5: If there are no white pixels left ie..the image has been completely eroded, quit the loop
        if (countNonZero(imgToUse) == 0)
        {
            break;
        }
    }

    //imshow("Masked white", skel);
    std::cout << "Skeleton refinements = " << to_string(i) << std::endl;

    return skel;
}

Mat ProcessGoalposts(Mat img, Scalar lower_color, Scalar upper_color)
{
    //process two images

    //Image has been read as BGR
    //Convert the captured frame from BGR to HSV

    //Image has been read as BGR
    Mat imgHSV;
    Mat imgMask;
    Mat generatedLines;
    
    cvtColor(img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

    inRange(imgHSV, lower_color, upper_color, imgMask); //Threshold the image

    //Skeletonize
    Mat imgSkel;
    imgSkel = Skeletonize(imgMask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(4, 4));

    Mat imgDil;
    dilate(imgSkel, imgDil, kernel, Point(-1, -1), 2, 1, 1);

    Mat imgHoughLines = ProbabilisticHoughLines(imgDil);

    //isolate the pink
    Scalar lower_pink = Scalar(0, 255, 0);
    Scalar upper_pink = Scalar(0, 255, 0);


    inRange(imgHoughLines, lower_pink, upper_pink, generatedLines);
    //cv::imshow("Probablistic Hough Lines", generatedLines);
    
    //return extrafinal;
    return generatedLines;
}

Mat CropImage(Mat img)
{
    /*
    Get the relevant image bounds
    Crop the image
    */

    //Original crop
    double xMin = 890;
    double xMax = 1008;
    double yMin = 410;
    double yMax = 692;
    double boundsPercentage = 0.1;

    //Add a border to crop
    int xMinAdjusted = (int)(xMin - round((xMax - xMin) * boundsPercentage));
    int xMaxAdjusted = (int)(xMax + round((xMax - xMin) * boundsPercentage));
    int yMinAdjusted = (int)(yMin - round((yMax - yMin) * boundsPercentage));
    int yMaxAdjusted = (int)(yMax + round((yMax - yMin) * boundsPercentage));

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
                drawContours(img, conPoly, i, Scalar(255, 0, 255), FILLED);
                std::cout << conPoly[i].size() << endl;
            }
        }
    }   
}

void MyDistortPoints(
    const std::vector<cv::Point2d>& src,
    std::vector<cv::Point2d>& dst,
    cv::InputArray camMat,
    cv::InputArray distMat)
{
    dst.clear();

    cv::Mat cameraMatrix = camMat.getMat();
    cv::Mat distorsionMatrix = distMat.getMat();

    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double ux = cameraMatrix.at<double>(0, 2);
    double uy = cameraMatrix.at<double>(1, 2);

    double k1 = distorsionMatrix.at<double>(0);
    double k2 = distorsionMatrix.at<double>(1);
    double p1 = distorsionMatrix.at<double>(2);
    double p2 = distorsionMatrix.at<double>(3);
    double k3 = distorsionMatrix.at<double>(4);

    //https://web.archive.org/web/20170607081328/https://code.opencv.org/issues/1387

    dst.clear();

    //BOOST_FOREACH(const cv::Point2d &p, src)
    for (unsigned int i = 0; i < src.size(); i++)
    {
        const cv::Point2d& p = src[i];
        double x = p.x;
        double y = p.y;
        double xCorrected, yCorrected;
        //Step 1 : correct distorsion
        {
            double r2 = x * x + y * y;
            //radial distorsion
            xCorrected = x * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
            yCorrected = y * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

            //tangential distorsion
            //http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
            xCorrected = xCorrected + (2. * p1 * x * y + p2 * (r2 + 2. * x * x));
            yCorrected = yCorrected + (p1 * (r2 + 2. * y * y) + 2. * p2 * x * y);
        }
        //Step 2 : ideal coordinates => actual coordinates
        {
            xCorrected = xCorrected * fx + ux;
            yCorrected = yCorrected * fy + uy;
        }
        dst.push_back(cv::Point2d(xCorrected, yCorrected));
    }
}

/*
void MyDistortPoints(const std::vector<cv::Point2d>& src, std::vector<cv::Point2d>& dst,
    const cv::Matx33d& cameraMatrix, const cv::Matx<double, 1, 5>& distorsionMatrix)
{
    cv::Mat cameraMatrix2(cameraMatrix);
    cv::Mat distorsionMatrix2(distorsionMatrix);
    return MyDistortPoints(src, dst, cameraMatrix2, distorsionMatrix2);
}*/

Matx33d CameraMatrixSet(double fx, double fy, double ux, double uy)
{
    Matx33d cameraMatrix;

    cameraMatrix(0, 0) = fx;
    cameraMatrix(0, 1) = 0;
    cameraMatrix(0, 2) = ux;

    cameraMatrix(1, 0) = 0;
    cameraMatrix(1, 1) = fy;
    cameraMatrix(1, 2) = uy;

    cameraMatrix(2, 0) = 0;
    cameraMatrix(2, 1) = 0;
    cameraMatrix(2, 2) = 1;

    return cameraMatrix;
}

cv::Matx<double, 1, 5> DistortionMatrixSet(double k1, double k2, double k3, double p1, double p2)
{
    cv::Matx<double, 1, 5> distortionMatrix;

    distortionMatrix(0, 0) = k1;
    distortionMatrix(0, 1) = k2;
    distortionMatrix(0, 2) = p1;
    distortionMatrix(0, 3) = p2;
    distortionMatrix(0, 4) = k3;

    return distortionMatrix;
}

Mat UnDistort(std::string imageName, double fx, double fy, double ux, double uy, double k1, double k2, double k3, double p1, double p2)
{

    //https://web.archive.org/web/20170607081328/https://code.opencv.org/issues/1387
    std::cout << std::string("Using values : ") <<
        std::string(", fx = ") << fx <<
        std::string(", fy = ") << fy <<
        std::string(", ux = ") << ux <<
        std::string(", uy = ") << uy <<
        std::string(", k1 = ") << k1 <<
        std::string(", k2 = ") << k2 <<
        std::string(", k3 = ") << k3 <<
        std::string(", p1 = ") << p1 <<
        std::string(", p2 = ") << p2 << 
        std::endl;
    
    Matx33d cameraMatrix = CameraMatrixSet(fx, fy, ux, uy);
    cv::Matx<double, 1, 5> distortionMatrix = DistortionMatrixSet(k1, k2, k3, p1, p2);

    std::vector<cv::Point2d> distortedPoints;
    std::vector<cv::Point2d> undistortedPoints;
    std::vector<cv::Point2d> redistortedPoints;

    /*
    distortedPoints.push_back(cv::Point2d(324., 249.));// equals to optical center
    distortedPoints.push_back(cv::Point2d(340., 200));
    distortedPoints.push_back(cv::Point2d(785., 345.));
    distortedPoints.push_back(cv::Point2d(0., 0.));*/

    distortedPoints.push_back(cv::Point2d(0, 0));// equals to optical center
    distortedPoints.push_back(cv::Point2d(1920, 0));
    distortedPoints.push_back(cv::Point2d(0., 1080));
    distortedPoints.push_back(cv::Point2d(1920, 1080));

    cv::undistortPoints(distortedPoints, undistortedPoints, cameraMatrix, distortionMatrix);
    MyDistortPoints(undistortedPoints, redistortedPoints, cameraMatrix, distortionMatrix);
    cv::undistortPoints(redistortedPoints, undistortedPoints, cameraMatrix, distortionMatrix);

    //std::cout << std::string("DistortedPoints: ") << distortedPoints << std::endl;
    //std::cout << std::string("UnDistortedPoints: ") << undistortedPoints << std::endl;
    //std::cout << std::string("ReDistortedPoints: ") << redistortedPoints << std::endl;

    //Testing
    std::string loadingPath = imageName;
    Mat originalImage = imread(loadingPath);

    vector< Point2f> dst_corners(4);
    dst_corners[0].x = 0;
    dst_corners[0].y = 0;
    dst_corners[1].x = 1920;
    dst_corners[1].y = 0;
    dst_corners[2].x = dst_corners[1].x;
    dst_corners[2].y = 1080;
    dst_corners[3].x = 0;
    dst_corners[3].y = dst_corners[2].y;

    /*
    dst_corners[0].x = 0;
    dst_corners[0].y = 0;
    dst_corners[1].x = 1920;
    dst_corners[1].y = 0;
    dst_corners[2].x = dst_corners[1].x;
    dst_corners[2].y = 1080;
    dst_corners[3].x = 0;
    dst_corners[3].y = dst_corners[2].y;
*/

    Size warped_image_size = Size(cvRound(dst_corners[2].x), cvRound(dst_corners[2].y));

    //undistort
    Mat newCameraMatrix;
    Mat undistorted;
    Mat map1, map2;
    Size imageSize(cv::Size(originalImage.cols, originalImage.rows));

    newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distortionMatrix, imageSize, 1, imageSize, 0);
    cv::undistort(originalImage, undistorted, newCameraMatrix, distortionMatrix, newCameraMatrix);

    //Alternate method to undistort
    //cv::initUndistortRectifyMap(cameraMatrix, distorsionMatrix, cv::Mat(), cv::getOptimalNewCameraMatrix(newCameraMatrix, distorsionMatrix, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
    //remap(original_image, undistorted, map1, map2, cv::INTER_LINEAR);

    //crop the image
        //x, y, w, h = roi
        //dst = dst[y:y + h, x : x + w]

    //Unit test ensuring we have an accuracy that is better than 0.001 pixel
    for (unsigned int i = 0; i < undistortedPoints.size(); i++)
    {
        cv::Point2d dist = redistortedPoints[i] - distortedPoints[i];
        double norm = sqrt(dist.dot(dist));
        std::cout << "norm = " << norm << std::endl;
        assert(norm < 1E-3);
    } 

    //try
    //Mat M = getPerspectiveTransform(distortedPoints, undistortedPoints);
    //Mat warpedImage;
    //warpPerspective(originalImage, warpedImage, M, warped_image_size); // do perspective transformation*/
    //imshow("warped", warpedImage);

    return undistorted;
}


float CalculateFocalPixel(float focalLength, float sensorDimension, float pixelDimension)
{
    //focal_pixel = (focal_mm / sensor_width_mm) * image_width_in_pixels
    float value = (focalLength / sensorDimension) * pixelDimension;

    return value;
}

void ProcessData(int i, std::string savePath) 
{
    std::cout << "Processing image..." << std::endl;

    //iterate to get the matcfhing record data associated with each image

    list<RecordingData>::iterator itr = recordData.begin();
    std::advance(itr, i);
    RecordingData foundEntry = *itr;

    list<std::string>::iterator itrNames = imageNames.begin();
    std::advance(itrNames, i);
    std::string  distortedImagePath = *itrNames;

    //std::cout << foundEntry.timeStamp << std::endl;

    Mat distortedImage = imread(distortedImagePath);
    Mat undistortedImage;

    undistortedImage = UnDistort(distortedImagePath, foundEntry.fx, foundEntry.fy, foundEntry.cx, foundEntry.cy, foundEntry.k1, foundEntry.k2, foundEntry.k3, foundEntry.p1, foundEntry.p2);
    
    Scalar lower_color, upper_color;
    double sensitivity;

    //Check for the white
    sensitivity = 30.0;
    lower_color = Scalar(0, 0, 255.0 - sensitivity);
    upper_color = Scalar(255, sensitivity, 255);
    

    /*
    //Check for the blue
    sensitivity = 50.0;
    lower_color = Scalar(130 - sensitivity, 255, 255.0 - sensitivity);
    upper_color = Scalar(130 + sensitivity, sensitivity, 255);
    */ 

    //Save undistorted
     
    //Save distorted

    Mat processedUndistortedImage;
    Mat processedDistortedImage;

    //Raw image processed
    processedDistortedImage = ProcessGoalposts(distortedImage, lower_color, upper_color);
    //Distorted image processed
    processedUndistortedImage = ProcessGoalposts(undistortedImage, lower_color, upper_color);



    //Save the images
    std::string finalPath;

    //Combined images for testing
    Mat combined = distortedImage;
    Mat convertedColorspace;

    Mat matDst(Size(combined.cols * 2, combined.rows * 2), combined.type(), Scalar::all(0));

    Mat matRoi = matDst(Rect(0, 0, combined.cols, combined.rows));
    combined.copyTo(matRoi);

    matRoi = matDst(Rect(combined.cols, 0, combined.cols, combined.rows));
    undistortedImage.copyTo(matRoi);

    matRoi = matDst(Rect(0, combined.rows, combined.cols, combined.rows));
    cvtColor(processedDistortedImage, convertedColorspace, COLOR_RGB2BGR);
    convertedColorspace.copyTo(matRoi);

    matRoi = matDst(Rect(combined.cols, combined.rows, combined.cols, combined.rows));
    cvtColor(processedUndistortedImage, convertedColorspace, COLOR_RGB2BGR);
    convertedColorspace.copyTo(matRoi);


    // Display big mat
    //cv::imshow("Images", matDst);
    finalPath = savePath + "CombinedComparison\\" + to_string(i) + ".png";
    cv::imwrite(finalPath, matDst);

    //cv::imwrite("D:\\Testing\\OpenCVTesting\\OpenCVTesting\\afterprocessingImage.png", undistorted);
    std::cout << finalPath << std::endl;
}

/*
int main(int argc, const char** argv) 
{
    RunImageProcessing();
}*/

void RunImageProcessing()
{
    if (__cplusplus == 202101L)
        std::cout << "C++23" << std::endl;
    else if (__cplusplus == 202002L) std::cout << "C++20" << std::endl;
    else if (__cplusplus == 201703L) std::cout << "C++17" << std::endl;
    else if (__cplusplus == 201402L) std::cout << "C++14" << std::endl;
    else if (__cplusplus == 201103L) std::cout << "C++11" << std::endl;
    else if (__cplusplus == 199711L) std::cout << "C++98" << std::endl;
    else std::cout << "pre-standard C++." << __cplusplus << std::endl;

    std::string savePath = "D:\\Testing\\OpenCVTesting\\OpenCVTesting\\";

    // Read the data file
    ReadData("D:\\TestData\\PTZ Movetest 17Jan\\17 Jan 2024, 12755 pm.csv");

    if (testingMode == 0)
    {

    }
    else if (testingMode == 1)
    {
        // Read the data file
        ReadData("D:\\TestData\\PTZ Movetest 17Jan\\17 Jan 2024, 12755 pm.csv");

        // Read the image files
        ReadImageNames("D:\\TestData\\PTZ Movetest 17Jan");

        //Go through loop distorting each image
        for (int i = 0; i < imageNames.size(); i += 100)
        {
            //int imageNumbertoTest = 1500;
            ProcessData(i, savePath);
        }

        std::cout << "Successfully processed images" << std::endl;

        //waitKey(0);
        destroyAllWindows();
    }
    else if (testingMode == 2)
    {
        // Read the data file
        ReadData("D:\\TestData\\PTZ Movetest 17Jan\\17 Jan 2024, 12755 pm.csv");
    }
}

void IterateNextFrame()
{
    currentCameraFrame += 1;
    std::cout << "Generating frame: " + std::to_string(currentCameraFrame) << std::endl;
    list<RecordingData>::iterator itr = recordData.begin();
    std::advance(itr, currentCameraFrame);
    currentCameraData = *itr;
}

void CompareFrames()
{

}

void ProcessGLFrame(Mat m, int counter)
{

    //get lines
    Mat imgSkel = m;
    cvtColor(m, imgSkel, cv::COLOR_RGB2GRAY);

    cv::threshold(imgSkel, imgSkel, 127, 255, cv::THRESH_BINARY);
    imgSkel = Skeletonize(imgSkel);


    //get lines

    Mat imgHoughLines = ProbabilisticHoughLines(imgSkel);

    //get point positions and store


    //dilate to make mask
    Mat dilated;
    
    //We probably want the kernal size to correlate to camera position and zoom relative to the goalposts.
    //Some calculation ideally neeeded for this.
    Mat kernel = getStructuringElement(MORPH_RECT, Size(dilateGL, dilateGL));

    dilate(m, dilated, kernel, Point(-1, -1), 2, 1, 1);
    
    //We now have the mask. Apply this to the raw frame.

    //Mat masked
    //img1.copyTo(r, img2);

    //cv::imshow("Images", img);
    
    if (counter % 60 == 0)
    {
        cv::imwrite("D:\\TestData\\TestFiles\\test" + std::to_string(counter) + ".png", dilated);
    }

}



float GetCameraPitch()
{
    float pitch = 0;
    if (testingMode == 0)
    {

    }
    else if (testingMode == 1)
    {
        
    }
    else if (testingMode == 2)
    {
        pitch = currentCameraData.pitch;
    }
    else if (testingMode == 3)
    {
        pitch = 0.0f;
    }
    return pitch;
}

float GetCameraYaw()
{
    float yaw = 0;
    if (testingMode == 0)
    {

    }
    else if (testingMode == 1)
    {

    }
    else if (testingMode == 2)
    {
        yaw = currentCameraData.yaw;
    }
    else if (testingMode == 3)
    {
        yaw = 90.0f;
    }
    return yaw;
}

float GetCameraRoll()
{
    float roll = 0;
    if (testingMode == 0)
    {

    }
    else if (testingMode == 1)
    {

    }
    else if (testingMode == 2)
    {
        roll = currentCameraData.roll;
    }
    else if (testingMode == 3)
    {
        roll = 0.00;
    }
    return roll;
}

float GetCameraXPosition() 
{
    float x = 0;
    if (testingMode == 0)
    {

    }
    else if (testingMode == 1)
    {

    }
    else if (testingMode == 2)
    {
        x = currentCameraData.xPos;
    }
    else if (testingMode == 3)
    {
        x = 150.0f;
    }
    return x;
}

float GetCameraYPosition()
{
    float y = 0;
    if (testingMode == 0)
    {

    }
    else if (testingMode == 1)
    {

    }
    else if (testingMode == 2)
    {
        y = currentCameraData.yPos;
    }
    else if (testingMode == 3)
    {
        y = 0.0f;
    }

    return y;
}

float GetCameraZPosition()
{
    float z = 0;

    if (testingMode == 0)
    {

    }
    else if (testingMode == 1)
    {

    }
    else if (testingMode == 2)
    {
        z = currentCameraData.zPos;
    }
    else if (testingMode == 3)
    {
        z = 10.0f;
    }

    return z;
}

