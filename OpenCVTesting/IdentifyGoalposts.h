#pragma once

//Using C++ version 98

///
///Model importing
///
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
///

///
///OpenGL libraries
///
#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#include <KHR/khrplatform.h> 

#include <GLM/glm.hpp>
#include <GLM/gtc/matrix_transform.hpp>
#include <GLM/gtc/type_ptr.hpp>
///

///
///More opengl
/// 
#include "opengl/mesh.h"
///


//#include "linmath.h"
#include <stdlib.h>
#include <stdio.h>


#define _USE_MATH_DEFINES

///
///OpenCV library requirements
///
#include <cstring>
#include <iostream>
#include <string>
#include <filesystem>
#include <ctime>
#include <vector>
#include <math.h>
#include <list>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dirent.h"
#include "csv.h"
///

///
///Blackmagic
///
#include "LiveVideoWithOpenCV.h"
#include "com_utils.h"

#ifdef _WIN32
#include "DeckLinkAPI_i.c"
#endif
///

//using std::filesystem::directory_iterator;

using namespace cv;
using namespace std;

struct RecordingData {
    double epochTime = 0;
    std::string timeStamp = "";
    double roll = 0;
    double pitch = 0;
    double yaw = 0;
    double xPos = 0;
    double yPos = 0;
    double zPos = 0;
    double focalLength = 0;
    double k1 = 0;
    double k2 = 0;
    double k3 = 0;
    double p1 = 0;
    double p2 = 0;
    double fx = 0;
    double fy = 0;
    double cx = 0;
    double cy = 0;
};

struct FieldData {
    std::string intIdentifier = "";
    double xPos = 0;
    double yPos = 0;
    double zPos = 0;
    std::string pointIdentifier = "";
};

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

//std::vector<GLfloat> GenerateVertices(vector<vector<float>> points){}

void ComparePoints(vector<vector<int>> points1, vector<vector<int>> points2);

void ReadDataField(string filePath);

void ReadImageNames(const char* imagesPath);

void ReadData(string filePath);

void AddGoalpostPoints();

void GenerateField();

FieldData GetPostDataEntry(int i);

vector<double> MakePoint(double x, double y, double z);

void AdjustPositionsForOpenGL(vector<vector<double>> postPoints);

//void GenerateSinglePostPoints(FieldData point, double height, double radius, double positionOffsetX, double positionOffsetY, double positionOffsetZ);

//void GenerateAllPostPoints(double radius, double shortPostHeight, double tallPostHeight, double positionOffsetX, double positionOffsetY, double positionOffsetZ);

int FindClosestPointMatch(int j);

Mat ProbabilisticHoughLines(Mat imgToUse);

Mat Mask(Mat imageToMask, Mat mask);

Mat Skeletonize(Mat imgToUse);

Mat ProcessGoalposts(Mat img, Scalar lower_color, Scalar upper_color);

Mat CropImage(Mat img);

void GetContours(Mat imgDil, Mat img);

void MyDistortPoints(const std::vector<cv::Point2d>& src, std::vector<cv::Point2d>& dst, cv::InputArray camMat, cv::InputArray distMat);

Matx33d CameraMatrixSet(double fx, double fy, double ux, double uy);

cv::Matx<double, 1, 5> DistortionMatrixSet(double k1, double k2, double k3, double p1, double p2);

Mat UnDistort(std::string imageName, double fx, double fy, double ux, double uy, double k1, double k2, double k3, double p1, double p2);

float CalculateFocalPixel(float focalLength, float sensorDimension, float pixelDimension);

void ProcessData(int i, std::string savePath);

void RunImageProcessing();

void IterateNextFrame();

void ProcessGLFrame(Mat m, int counter);

//Getters
float GetCameraPitch();
float GetCameraYaw();
float GetCameraRoll();
float GetCameraXPosition();
float GetCameraYPosition();
float GetCameraZPosition();

//HRESULT Capture::VideoInputFrameArrived(IDeckLinkVideoInputFrame* videoFrame, IDeckLinkAudioInputPacket* /* unused */);