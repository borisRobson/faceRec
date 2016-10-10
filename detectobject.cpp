#include "detectobject.h"

using namespace cv;
using namespace std;

CascadeClassifier faceCascade, eyeCascade, eyeGlassesCascade;

detectobject::detectobject()
{
    //init cascades
    const char* faceCascadeFilename = "./cascades/lbpcascade_frontalface.xml";
    const char* eyeCascadeFilename = "./cascades/haarcascade_eye_tree_eyeglasses.xml";
    const char* eyeGlassesCascadeFilename = "./cascades/haarcascade_eye.xml";

    faceCascade.load(faceCascadeFilename);
    eyeCascade.load(eyeCascadeFilename);
    eyeGlassesCascade.load(eyeGlassesCascadeFilename);

    if(faceCascade.empty() || eyeCascade.empty() || eyeGlassesCascade.empty()){
        qDebug() << "error loading cascade files";
        qDebug() << "check filenames and paths";
        return;
    }else{
        qDebug() << "cascades loaded";
    }
}

Mat detectobject::findFace(Mat &image)
{

    //convert image to greyscale
    Mat grey(Size(640,480),CV_8UC1);
    //detect channels and apply appropriate conversion
    if(image.channels() == 3){
        cvtColor(image,grey,CV_BGR2GRAY);
    }else if(image.channels() == 4){
        cvtColor(image,grey, CV_BGRA2GRAY);
    }else{
        grey = image;
    }

    //equalise hist - smooths differences in lighting
    equalizeHist(grey,grey);

    //find larget object in image : face
    vector<Rect> objects;
    Rect faceRect;

    detectlargestobject(grey, faceCascade, objects);

    if(objects.size() > 0){
        //set largest object
        faceRect = objects.at(0);
    }else{
        //set false rect
        faceRect = Rect(-1,-1,-1,-1);
        return Mat();
    }

    if (faceRect.width > 0){
        //return face
        Mat face = grey(faceRect);
        int detectWidth = 200;
        resize(face,face,Size(detectWidth,detectWidth));
        Mat warped = warpImage(face);
        return warped;
    }else{
        //return empty Mat
        qDebug() << " no face found";
        return Mat();
    }
}


//Mat detectobject::warpImage(Mat &face, vector<Point> eyes)
Mat detectobject::warpImage(Mat &face)
{
    const int faceWidth = 200;
    const int faceHeight = faceWidth;

    //perform the transform
    Mat warped = Mat(faceHeight, faceWidth, CV_8UC1);
    face.copyTo(warped);

    //find centre of face
    Mat mask = Mat(warped.size(), CV_8UC1, Scalar(0));
    Point faceCentre = Point(faceWidth/2, cvRound(faceHeight*0.40));
    Size size = Size(cvRound(faceWidth * 0.4), cvRound(faceHeight * 0.7));

    //apply mask around the edges
    ellipse(mask, faceCentre, size, 0, 0, 360, Scalar(255), CV_FILLED);
    Mat dstImg = Mat(warped.size(), CV_8U, Scalar(0));
    warped.copyTo(dstImg,mask);

    //crop unnecessary black edges
    Rect ROI;
    ROI.x = 20;
    ROI.width = dstImg.cols - 40;
    ROI.y = 0;
    ROI.height = dstImg.rows;

    //make square
    Mat final = dstImg(ROI);
    resize(final,final,Size(160,160));
    return final;
}

void detectobject::detectlargestobject(Mat &image, CascadeClassifier &cascade, vector<Rect> &objects)
{
    bool scaled = false;
    //set flags for single object detection
    int flags = CASCADE_FIND_BIGGEST_OBJECT;

    /*
        set detection parameters:
        -min size
        -search detail
        -false detection threshold
    */
    Size minFeatureSize = Size(20,20);
    float searchScaleFactor = 1.1f;
    int minNeighbours = 2;

    Mat dst;
    image.copyTo(dst);

    //resize large image for faster processing time
    if(dst.cols > 500){
        resize(dst, dst, Size(), 0.5, 0.5);
        scaled = true;
    }

    //opencv obj detect function
    cascade.detectMultiScale(dst,objects,searchScaleFactor,minNeighbours, flags, minFeatureSize);

    //resize if scaled
    if(scaled){
        for(uint i = 0; i < objects.size(); i++){
            objects[i].x = cvRound(objects[i].x * 2);
            objects[i].y = cvRound(objects[i].y * 2);
            objects[i].width = cvRound(objects[i].width * 2);
            objects[i].height = cvRound(objects[i].height * 2);
        }
    }
    return;
}







