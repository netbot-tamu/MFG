
#include "settings.h"

#include <QDir>
#include <QFile>
#include <QDebug>

MfgSettings::MfgSettings(QString _cameraID, QObject* parent)
: QObject(parent),
cameraID(_cameraID)
{
   // The application should be located in [mfg location]/bin/
   QString mfgDirStr = QApplication::applicationDirPath();//.left(0);
   QDir mfgDir(mfgDirStr);
   // ...so move the path up to the root [mfg location]
   mfgDir.cdUp();
   configDir = mfgDir.absolutePath();

   // ...and enter the config directory for the settings
   settingsFile = configDir + "/config/mfgSettings.ini";
   if (!QFile::exists(settingsFile))
   {
      qWarning() << "Warning: File" << settingsFile << "does not exist, exiting";
      exit(1);
   }
   qDebug() << "Loaded settings file:\n" << settingsFile;

   mfgSettings = new QSettings(settingsFile, QSettings::IniFormat);

   loadSettings();
}

MfgSettings::~MfgSettings()
{
   delete mfgSettings;
}

#define LOAD_STR(var, key) var = mfgSettings->value(key).toString();
#define LOAD_INT(var, key) var = mfgSettings->value(key).toInt();
#define LOAD_DOUBLE(var, key) var = mfgSettings->value(key).toDouble();

void MfgSettings::loadSettings()
{
   LOAD_DOUBLE(siftThreshold, "sift/threshold")
   qDebug() << "SIFT Threshold:" << siftThreshold;
   
   // If no camera ID was specified, use the default ID in the settings
   if (cameraID == "")
   {
      LOAD_STR(cameraID, "default/camera")
   }
   qDebug() << "Using camera settings:" << cameraID;
   //*
   qDebug() << "stuff..."
            << mfgSettings->childKeys()
            << mfgSettings->childGroups();
   // */

   // Read the settings for this cameraID
   mfgSettings->beginGroup(cameraID);
   LOAD_DOUBLE(imageWidth, "width")
   LOAD_STR(initialImage, "image")
   //*
   qDebug() << "Image path:" << initialImage 
            << mfgSettings->childKeys()
            << mfgSettings->childGroups()
            << mfgSettings->contains("image")
            << mfgSettings->contains("width")
            << mfgSettings->contains("intrinsics");
   // */

   // if the initial image starts with a slash or windows drive, assume it
   // is an absolute path, otherwise assume it is relative, from the user's
   // home directory
   if (!initialImage.startsWith("/"))
   {
      initialImage = QDir::home().absolutePath() + "/" + initialImage;
   }
   qDebug() << "Image path:" << initialImage;

   // Read the intrinsics values and generate the matrix
   mfgSettings->beginGroup("intrinsics");
   LOAD_DOUBLE(cameraIntrinsics(0,0), "alpha_x")
   LOAD_DOUBLE(cameraIntrinsics(0,1), "gamma")
   LOAD_DOUBLE(cameraIntrinsics(0,2), "u_0")
   cameraIntrinsics(1,0) = 0.0;
   LOAD_DOUBLE(cameraIntrinsics(1,1), "alpha_y")
   LOAD_DOUBLE(cameraIntrinsics(1,2), "v_0")
   cameraIntrinsics(2,0) = 0.0;
   cameraIntrinsics(2,1) = 0.0;
   cameraIntrinsics(2,2) = 1.0;
   mfgSettings->endGroup(); // end [cameraID]/intrinsics

   // Read the distance coefficients
   mfgSettings->beginGroup("distCoeffs");
   if (mfgSettings->contains("c4"))
   {
      distCoeffs = CoeffMatrix(5,1);
      LOAD_DOUBLE(distCoeffs(4,0), "c4")
   }
   else 
   {
      distCoeffs = CoeffMatrix(4,1);
   }
   LOAD_DOUBLE(distCoeffs(0,0), "c0")
   LOAD_DOUBLE(distCoeffs(1,0), "c1")
   LOAD_DOUBLE(distCoeffs(2,0), "c2")
   LOAD_DOUBLE(distCoeffs(3,0), "c3")
   mfgSettings->endGroup(); // end [cameraID]/distCoeffs

   mfgSettings->endGroup(); // end [cameraID]
}
