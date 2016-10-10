#include <QCoreApplication>
#include "stream.h"

using namespace std;

stream *strm;
char* thresh;
char* userId;

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    if(argc < 2){
        qDebug() << "Usage is ./faceRec <thresh> <userId>";
        return -1;
    }

    thresh = argv[1];
    userId = argv[2];

    Task *task = new Task(&app);

    QObject::connect(task, SIGNAL(finished()), &app, SLOT(quit()));

    //init gstreamer
    if(!gst_is_initialized())
        gst_init(&argc,&argv);

    //run program
    QTimer::singleShot(10,task,SLOT(run()));

    return app.exec();
}

void Task::run()
{
    bool pipeBuilt, recogniserBuilt;    

    //convert char* value to double
    int temp = atoi(thresh);
    double threshold = (double)temp/100;

    int user = atoi(userId);

    strm = new stream();
    /*
     *Use userId to load images from nvdata
     *and confirm faceRecogniser model built
     *successfully
     */
    recogniserBuilt = strm->trainrecogniser(user, threshold);

    /*
     *Create and link all gst-elements
     *Confirm Success
     */
    pipeBuilt = strm->buildpipeline();

    if(recogniserBuilt && pipeBuilt){
        strm->startstream();
    }else{
        qDebug() << "failed to intialise properly";
        QCoreApplication::instance()->exit(1);
    }

    return;
}
