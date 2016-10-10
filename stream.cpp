#include "stream.h"
#include "detectobject.h"
#include "recognition.h"

using namespace cv;
using namespace std;

gboolean bus_cb(GstBus *bus, GstMessage *msg, gpointer user_data);
GstFlowReturn new_preroll(GstAppSink* asink, gpointer data);
GstFlowReturn new_buffer(GstAppSink* asink, gpointer data);
gboolean timeout(gpointer data);
void writeimage(Mat image);

const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";

detectobject *obj;
recognition *rec;

double SIMILARITY_THRESHOLD = 0.4f;
int MATCH_THRESHOLD = 2;

int match_count;
bool userFound;

Ptr<FaceRecognizer> model;

GMainLoop *loop;
GstElement *pipeline;
GstElement *appsink;

stream::stream()
{
    obj = new detectobject();
    rec = new recognition();
}

bool stream::buildpipeline()
{
    /*gst-launch-0.10 v4l2src ! videoscale ! capsfilter caps="video/x-raw-yuv,width=640,height=480" ! ffmpegcolorspace \
       !capsfilter caps="video/x-raw-gray, width=640, height=480, bpp=8, depth=8" ! ffmpegcolorspace ! ximagesink */
    //create components
    GstElement *src;
    GstElement *scale;
    GstElement *yuvfilter;
    GstElement *conv1;
    GstElement *rgbfilter;
    GstElement *conv2;

    loop = g_main_loop_new(NULL,false);

    pipeline = gst_pipeline_new(NULL);
#ifdef IMX6
    src = gst_element_factory_make("mfw_v4lsrc", NULL);
#else
    src = gst_element_factory_make("v4l2src", NULL);
#endif
    scale = gst_element_factory_make("videoscale", NULL);
    yuvfilter = gst_element_factory_make("capsfilter", NULL);
    conv1 = gst_element_factory_make("ffmpegcolorspace", NULL);
    rgbfilter = gst_element_factory_make("capsfilter", NULL);
    conv2 = gst_element_factory_make("ffmpegcolorspace", NULL);

    gst_util_set_object_arg(G_OBJECT(yuvfilter), "caps",
                            "video/x-raw-yuv, width=640, height=480");
    gst_util_set_object_arg(G_OBJECT(rgbfilter), "caps",
                            "video/x-raw-gray, width=640, height=480, bpp=8, depth=8");

    appsink = gst_element_factory_make("appsink", NULL);

    //set app sink properties
    gst_app_sink_set_emit_signals((GstAppSink*)appsink, true);
    gst_app_sink_set_drop((GstAppSink*)appsink, true);
    gst_app_sink_set_max_buffers((GstAppSink*)appsink, 1);
    GstAppSinkCallbacks callbacks = {NULL, new_preroll, new_buffer, NULL};
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink), &callbacks, NULL, NULL);

    //add components to pipelines
    gst_bin_add_many(GST_BIN(pipeline), src,scale, yuvfilter, conv1, rgbfilter, conv2, appsink, NULL);

    //link
    if(!gst_element_link_many(src, scale, yuvfilter, conv1, rgbfilter, conv2,appsink,NULL))
        return false;

    //assign bus callback
    gst_bus_add_watch(GST_ELEMENT_BUS(pipeline), bus_cb, loop);
    gst_object_unref(GST_ELEMENT_BUS(pipeline));

    return true;
}

bool stream::trainrecogniser(int userId, double threshold)
{
    vector<Mat>faces;
    vector<int>labels;

    /*
     * Load each database image and detect faces
     * store faces to array and train the recogniser
     */
    for(int i = 0; i <=2; i++){

        char* filename = new char[64];
#ifdef IMX6
        sprintf(filename,"/nvdata/tftpboot/user%d_%d.jpg",userId,i);
#else
        sprintf(filename,"./faces/user%d_%d.jpg",userId,i);
#endif
        qDebug() << filename;
        Mat dbImage = imread(filename, 0);
        if(dbImage.empty()){
            return false;
        }

        Mat face = obj->findFace(dbImage);
        if(face.empty())
            return false;

        faces.push_back(face);
        labels.push_back(0);

        writeimage(face);
        delete []filename;
    }

    model = rec->learnCollectedFaces(faces, labels, facerecAlgorithm);

    SIMILARITY_THRESHOLD = threshold;

    return true;
}

void stream::startstream()
{
    qDebug() << __FUNCTION__;

    //set to playing
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    //run loop
    g_main_loop_run(loop);

    //once exited, dispose of elements
    gst_element_set_state(pipeline, GST_STATE_NULL);

    gst_object_unref(pipeline);

    if(userFound){
        qDebug() << "**Access Granted**";
        QCoreApplication::instance()->exit(0);
        return;
    }

    qDebug() << "**Access Denied**";
    QCoreApplication::instance()->exit(1);    
}


gboolean bus_cb(GstBus *bus, GstMessage *msg, gpointer user_data)
{
    Q_UNUSED(bus);
    Q_UNUSED(user_data);
    //parse bus messages
    switch(GST_MESSAGE_TYPE(msg)){
        case GST_MESSAGE_ERROR:{
            //quit on error
            GError *err;
            gchar *dbg;
            gst_message_parse_error(msg, &err, &dbg);
            gst_object_default_error(msg->src, err, dbg);
            g_clear_error(&err);
            g_free(dbg);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_STATE_CHANGED:{
            GstState old_state, new_state, pending_state;
            gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
            //just show pipeline messages
            if(GST_OBJECT_NAME(msg->src) == GST_OBJECT_NAME(pipeline)){
                g_print("'%s' state changed from %s to %s \n", GST_OBJECT_NAME(msg->src), gst_element_state_get_name(old_state), gst_element_state_get_name(new_state));
            }
            break;
        }
        default:
            break;
    }
    return TRUE;
}

GstFlowReturn new_preroll(GstAppSink* asink, gpointer data)
{
    Q_UNUSED(asink);
    Q_UNUSED(data);
    qDebug() << "got preroll";
    return GST_FLOW_OK;
}

int fcount;
GstFlowReturn new_buffer(GstAppSink* asink, gpointer data)
{
    fcount++;
    if (fcount == 1){
        g_timeout_add(5000, timeout,NULL);
    }
    qDebug() << "new buffer";
    Q_UNUSED(data);
    gst_app_sink_set_emit_signals((GstAppSink*)asink, false);

    //grab available buffer from appsink
    GstBuffer *buf = gst_app_sink_pull_buffer(asink);

    //allocate buffer data to cvMat format
    Mat frame(Size(640,480), CV_8UC1, GST_BUFFER_DATA(buf), Mat::AUTO_STEP);

    Mat face = obj->findFace(frame);

    if(!face.empty()){
        //run captured image through trained recogniser
        Mat reconstructed = rec->reconstructFace(model, face);

        //compare captured image with average image from database
        double similarity = rec->getSimilarity(reconstructed,face);        
        cout << "similarity: " << similarity << endl;

        if(similarity <= SIMILARITY_THRESHOLD){
            qDebug() << "found user";
            match_count++;
            if(match_count >= MATCH_THRESHOLD){
                userFound = true;
                g_main_loop_quit(loop);
            }
        }
    }

    gst_app_sink_set_emit_signals((GstAppSink*)asink, true);

    return GST_FLOW_OK;
}

gboolean timeout(gpointer data)
{
    Q_UNUSED(data);
    userFound = false;
    g_main_loop_quit(loop);
    return false;
}


int cap;
void writeimage(Mat image)
{
    //write captured image to file
    char filename[16];
    cap++;
    vector<int>compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    sprintf(filename, "face%d.png",cap);
    string file = string(filename);
    imwrite(file, image, compression_params);
}
