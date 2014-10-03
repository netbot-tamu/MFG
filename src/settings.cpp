
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

   // ...and enter the config directory for the settings
   settingsFile = mfgDir.absolutePath() + "/config/mfgSettings.ini";
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
   //siftThreshold = mfgSettings.value("sift/threshold").toString();
   LOAD_DOUBLE(siftThreshold, "sift/threshold")
   
   // If no camera ID was specified, use the default ID in the settings
   if (cameraID == "")
   {
      //cameraID = mfgSettings.value("default/camera").toString();
      LOAD_STR(cameraID, "default/camera")
   }

   // Read the settings for this cameraID
   mfgSettings->beginGroup(cameraID);
   //imageWidth = mfgSettings.value("width").toInt();
   //initialImage = mfgSettings.value("image").toString();
   LOAD_DOUBLE(imageWidth, "width")
   LOAD_STR(initialImage, "image")

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
