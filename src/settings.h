

#ifndef CONFIG_H_
#define CONFIG_H_

#include <QApplication>
#include <QSettings>
#include <QString>
#include <string.h>

#include <opencv2/core/core.hpp>

typedef cv::Matx33d IntrinsicsMatrix;
typedef cv::Mat_<double> CoeffMatrix;
//typedef cv::Matx41d CoeffMatrix4;
//typedef cv::Matx51d CoeffMatrix5;

class MfgSettings : QObject {
   Q_OBJECT
public:
   MfgSettings(QString _cameraID="", QObject* _parent=0);
   ~MfgSettings();

   float getSiftThreshold() const {return siftThreshold;}
   int   getImageWidth() const {return imageWidth;}

   QString     getCameraID() const {return cameraID;}
   //std::string getCameraID() const {return cameraID.toStdString();}

   QString     getInitialImage() const {return initialImage;}
   //std::string getInitialImage() const {return initialImage.toStdString();}

   IntrinsicsMatrix  getIntrinsics() const {return cameraIntrinsics;}
   CoeffMatrix       getDistCoeffs() const {return distCoeffs;}

private:
   QString     configDir;     // path to config directory
   QString     settingsFile;  // path to settings file
   QSettings*  mfgSettings;   // settings object

   // General settings
   double   siftThreshold;
   QString  cameraID;

   // Input image settings
   int      imageWidth;
   QString  initialImage;

   // Camera settings
   IntrinsicsMatrix  cameraIntrinsics;
   CoeffMatrix       distCoeffs;

   void loadSettings();
};

#endif

