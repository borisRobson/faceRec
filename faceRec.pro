#-------------------------------------------------
#
# Project created by QtCreator 2016-09-28T11:47:16
#
#-------------------------------------------------

QT       += core

TARGET = faceRec
TEMPLATE = app



PROJECT_BASE_DIRECTORY = /home/standby/doorentry

QMAKE_CXXFLAGS-=-Wno-missing-field-intializers
#
# Setup paths according to target spec.
#
linux-mxc-g++ {

    # extract the boardtype from the toolchain.make file replaced the need for the project to have this.
    BOARDTYPE=$$system("grep '^BUILD_PROFILE' $$PROJECT_BASE_DIRECTORY/toolchain.make | awk -F'=' '{print $2}'")
    DEFINES += $$BOARDTYPE
    message("Arm Build: $$BOARDTYPE")

    # Allow application source code to conditionally compile for target or development host PC
    DEFINES += $$BOARDTYPE

    contains ( DEFINES, IMX6 ) {
        # add cflag
        QMAKE_CXXFLAGS+=-Wno-psabi
    }

    MYPREFIX = $$PROJECT_BASE_DIRECTORY/ltib/rootfs

    INCLUDEPATH += $$MYPREFIX/usr/include/gstreamer-0.10 $$MYPREFIX/usr/include/glib-2.0 $$MYPREFIX/usr/lib/glib-2.0/include $$MYPREFIX/usr/include/libxml2
    LIBS +=  -L$$MYPREFIX/usr/lib -lgstreamer-0.10 -lgstapp-0.10 -lgstbase-0.10 -lglib-2.0 -lgobject-2.0 -lgmodule-2.0 -lgthread-2.0 -lrt -lz -lxml2

    INCLUDEPATH += $$MYPREFIX/usr/include

    LIBS += -L$$MYPREFIX/usr/lib -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_objdetect -lopencv_contrib -lopencv_flann
    LIBS += -L$$MYPREFIX/usr/lib -lopencv_features2d -lopencv_calib3d -lopencv_ml -lopencv_legacy -lopencv_photo -lopencv_gpu -lopencv_ocl -lopencv_nonfree
    LIBS += -L$$MYPREFIX/usr/lib -lts -lz
}

linux-g++-|linux-g++-64 {
    DEFINES += PCBUILD
    message("x86 Build")



    INCLUDEPATH += /usr/include/gstreamer-0.10 /usr/include/glib-2.0 /usr/lib/glib-2.0/include /usr/include/libxml2 /usr/lib/x86_64-linux-gnu/glib-2.0/include
    LIBS +=  -L/usr/lib -lgstreamer-0.10 -lgstapp-0.10 -lgstbase-0.10 -lglib-2.0 -lgobject-2.0 -lgmodule-2.0 -lgthread-2.0 -lrt -lz -lxml2

    INCLUDEPATH += /usr/local/include

    LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_objdetect -lopencv_contrib -lopencv_flann
    LIBS += -lopencv_features2d -lopencv_calib3d -lopencv_ml -lopencv_legacy -lopencv_photo -lopencv_gpu -lopencv_ocl -lopencv_nonfree


}

SOURCES += main.cpp \
    stream.cpp \
    detectobject.cpp \
    recognition.cpp

HEADERS += \
    stream.h \
    detectobject.h \
    recognition.h
