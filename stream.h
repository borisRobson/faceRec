#ifndef STREAM_H
#define STREAM_H


#include <QObject>
#include <QDebug>
#include <QTime>
#include <QTimer>
#include <QFile>
#include <vector>
#include <string>
#include <QCoreApplication>

#include "gst/gst.h"
#include "gst/app/gstappsink.h"
#include "glib-2.0/glib.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class stream
{
public:
    stream();
    bool buildpipeline();
    bool trainrecogniser(int userId, double threshold);
    void startstream();
};


class Task: public QObject{
    Q_OBJECT
public:
    Task(QObject* parent=0) : QObject(parent){}
private:

public slots:
    void run();
signals:
    void finished();
};


#endif // STREAM_H
