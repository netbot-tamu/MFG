
#include "settings.h"

#include <QDir>
#include <QFile>
#include <QDebug>

// TODO: make these settings better
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
#define LOAD_BOOL(var, key) var = mfgSettings->value(key).toBool();
#define LOAD_DOUBLE(var, key) var = mfgSettings->value(key).toDouble();

void MfgSettings::loadSettings()
{
   // If no camera ID was specified, use the default ID in the settings
   if (cameraID == "")
   {
      LOAD_STR(cameraID, "default/camera")
   }
   qDebug() << "Using camera settings:" << cameraID;

   // Load the keypoint detection settings
   mfgSettings->beginGroup("features");
   LOAD_INT(featureAlg, "algorithm");
   LOAD_INT(featureDescriptorRadius, "descriptor_radius");
   switch(featureAlg)
   {
      case KPT_SIFT:
         loadSIFTSettings();
         break;
      case KPT_SURF:
         loadSURFSettings();
         break;
      case KPT_GFTT:
         loadGoodFeatSettings();
         break;
      default:
         qDebug() << "Error: invalid keypoint detection algorithm";
         system.exit(1);
   }
   mfgSettings->endGroup(); // end group "features"

   // Load Optic Flow settings
   LOAD_DOUBLE(oflkMinEigenval, "optic-flow/min_eigenval");
   LOAD_INT(oflkWindowSize, "optic-flow/window_size");

   // Load MFG and Bundle Adjustment settings
   loadMFGSettings();
   loadBASettings();

   //*
   // TODO: this should be removed, but it seems to make the program work
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

   loadIntrinsicsSettings();
   loadDistCoeffSettings();

   mfgSettings->endGroup(); // end [cameraID]
}

void loadSIFTSettings()
{
   qDebug() << "Detecting features with SIFT";
   LOAD_DOUBLE(siftThreshold, "sift/threshold")
   qDebug() << "   Threshold:" << siftThreshold;
}

void loadSURFSettings()
{
   qDebug() << "Detecting features with SURF";
}

void loadGFTTSettings()
{
   qDebug() << "Detecting features with GoodFeat";
   mfgSettings->beginGroup("gftt");
   LOAD_INT(gfttMaxPoints, "max_points");
   LOAD_DOUBLE(gfttQuality, "quality");
   LOAD_DOUBLE(gfttMinPointDistance, "min_point_distance");
   mfgSettings->endGroup();
}

void loadMFGSettings()
{
   mfgSettings->beginGroup("mfg");
   LOAD_DOUBLE(mfgPointToPlaneDist, "point_to_plane_dist");
   LOAD_INT(mfgNumRecentPoints, "num_recent_points");
   LOAD_INT(mfgNumRecentLines, "num_recent_lines");
   LOAD_INT(mfgPointsPerPlane, "points_per_plane");
   LOAD_INT(frameStep, "frame_step");
   LOAD_INT(frameStepInitial, "frame_step_initial");
   LOAD_DOUBLE(vpointAngleThresh, "vpoint_angle_thresh");
   mfgSettings->endGroup();
}

void loadBASettings()
{
   mfgSettings->beginGroup("ba");
   LOAD_DOUBLE(baWeightVPoint, "weight_vpoint");
   LOAD_DOUBLE(baWeightLine, "weight_line");
   LOAD_DOUBLE(baWeightPlane, "weight_plane");
   LOAD_BOOL(baUseKernel, "use_kernel");
   LOAD_INT(baNumFramesVPoint, "num_frames");

   if (baUseKernel)
   {
      mfgSettings->beginGroup("kernel");
      LOAD_DOUBLE(baKernelDeltaVPoint, "delta_vpoint");
      LOAD_DOUBLE(baKernelDeltaPoint, "delta_point");
      LOAD_DOUBLE(baKernelDeltaLine, "delta_line");
      LOAD_DOUBLE(baKernelDeltaPlane, "delta_plane");
      mfgSettings->endGroup(); // "ba/kernel"
   }
   mfgSettings->endGroup(); // "ba"
}

// Read the intrinsics values and generate the matrix
void loadIntrinsicsSettings()
{
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
}

// Read the distance coefficients
void loadDistCoeffSettings()
{
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
}

