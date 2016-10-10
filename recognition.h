#ifndef RECOGNITION_H
#define RECOGNITION_H

#include"opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv/cv.h"
#include "opencv/cxcore.h"


using namespace cv;
using namespace std;

class recognition
{
public:
    recognition();
    ~recognition();
    Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels,
                                            const string facerecAlgorithm = "FaceRecognizer.Eigenfaces");

    // Generate an approximately reconstructed face by back-projecting the eigenvectors & eigenvalues of the given (preprocessed) face.
    Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace);

    // Compare two images by getting the L2 error (square-root of sum of squared error).
    double getSimilarity(const Mat A, const Mat B);
};

#endif // RECOGNITION_H
