

#ifndef CONFIG_H_
#define CONFIG_H_

#include <QApplication>
#include <QSettings>
#include <QString>
#include <string.h>

#include <opencv2/core/core.hpp>

#include "consts.h"

typedef cv::Matx33d IntrinsicsMatrix;
typedef cv::Mat_<double> CoeffMatrix;
//typedef cv::Matx41d CoeffMatrix4;
//typedef cv::Matx51d CoeffMatrix5;

class MfgSettings : QObject {
   Q_OBJECT
public:
   //---------------------------------------------------------------------------
   // TODO: move this out of here
   //---------------------------------------------------------------------------
   //double   angle_thresh_vp2d3d; // degree, match local 2d vp to existing 3d vp


   MfgSettings(QString _cameraID="", QObject* _parent=0);
   ~MfgSettings();
   
   void printAllSettings() const;
   void setKeypointAlgorithm(FeatureDetectionAlgorithm _alg);

   //---------------------------------------------------------------------------
   // Access functions
   //---------------------------------------------------------------------------
   // General settings
   int   getImageWidth() const {return imageWidth;}

   QString getCameraID() const {return cameraID;}
   QString getInitialImage() const {return initialImage;}

   IntrinsicsMatrix  getIntrinsics() const {return cameraIntrinsics;}
   CoeffMatrix       getDistCoeffs() const {return distCoeffs;}

   // Feature Point Detection algorithm
   FeatureDetectionAlgorithm getKeypointAlgorithm() const {return featureAlg;}
   double   getFeatureDescriptorRadius() const {return featureDescriptorRadius;}

   // SIFT
   float    getSiftThreshold() const {return siftThreshold;}

   // GFTT
   int      getGfttMaxPoints() const {return gfttMaxPoints;}
   double   getGfttQualityLevel() const {return gfttQuality;}
   double   getGfttMinimumPointDistance() const {return gfttMinPointDistance;}

   // Optic Flow
   double   getOflkMinEigenval() const {return oflkMinEigenval;}
   int      getOflkWindowSize() const {return oflkWindowSize;}

   // MFG
   double   getMfgPointToPlaneDistance() const {return mfgPointToPlaneDist;}
   int      getMfgNumRecentPoints() const {return mfgNumRecentPoints;}
   int      getMfgNumRecentLines() const {return mfgNumRecentLines;}
   int      getMfgPointsPerPlane() const {return mfgPointsPerPlane;}

   int      getFrameStep() const {return frameStep;}
   int      getFrameStepInitial() const {return frameStepInitial;}
   double   getVPointAngleThresh() const {return vpointAngleThresh;}

   // BA
   double   getBaWeightVPoint() const {return baWeightVPoint;}
   double   getBaWeightLine() const {return baWeightLine;}
   double   getBaWeightPlane() const {return baWeightPlane;}
   bool     getBaUseKernel() const {return baUseKernel;}
   double   getBaKernelDeltaVPoint() const {return baKernelDeltaVPoint;}
   double   getBaKernelDeltaPoint() const {return baKernelDeltaPoint;}
   double   getBaKernelDeltaLine() const {return baKernelDeltaLine;}
   double   getBaKernelDeltaPlane() const {return baKernelDeltaPlane;}
   int      getBaNumFramesVPoint() const {return baNumFramesVPoint;}

private:
   //---------------------------------------------------------------------------
   // General settings
   //---------------------------------------------------------------------------
   QString     configDir;     // path to config directory
   QString     settingsFile;  // path to settings file
   QSettings*  mfgSettings;   // settings object

   // Camera settings
   QString           cameraID;
   IntrinsicsMatrix  cameraIntrinsics;
   CoeffMatrix       distCoeffs;

   // Initial image settings
   int      imageWidth;
   QString  initialImage;
   //int    use_img_width; // user designated image width
   

   //---------------------------------------------------------------------------
   // Feature Detection algorithm settings
   //---------------------------------------------------------------------------
   // Feature Point Detection algorithm
   FeatureDetectionAlgorithm featureAlg;

   // Keypoint region radius for descriptor
   double   featureDescriptorRadius;
   
   // SIFT
   double   siftThreshold;

   // GFTT settings
   int      gfttMaxPoints;        // maximum number of points
   double   gfttQuality;          // quality level
   double   gfttMinPointDistance; // minimum distance between two features


   //---------------------------------------------------------------------------
   // Optic Flow (oflk) settings
   //---------------------------------------------------------------------------
   // default 0.0001, minEignVal threshold for the 2x2 spatial motion matrix,
   // to eleminate bad points
   double   oflkMinEigenval; 
   int      oflkWindowSize;


   //---------------------------------------------------------------------------
   // MFG settings
   //---------------------------------------------------------------------------
   // MFG discover 3d planes
   double   mfgPointToPlaneDist;
   int      mfgNumRecentPoints;  // use recent points to discover planes
   int      mfgNumRecentLines;   // use recent lines to discover planes
   int      mfgPointsPerPlane;   // min num of points to claim a new plane

   int      frameStep;           // step size
   int      frameStepInitial;    // first step size
   double   vpointAngleThresh;   // vpoint angle threshold (2d -> 3d mapping)

   //---------------------------------------------------------------------------
   // Bundle Adjustment settings
   //---------------------------------------------------------------------------
   double   baWeightVPoint;
   double   baWeightLine;
   double   baWeightPlane;
   int      baNumFramesVPoint; // number of frames used for vp in BA

   bool     baUseKernel;
   double   baKernelDeltaVPoint;
   double   baKernelDeltaPoint;
   double   baKernelDeltaLine;
   double   baKernelDeltaPlane;


   //---------------------------------------------------------------------------
   // Private functions
   //---------------------------------------------------------------------------
   void loadSettings();
   void loadSIFTSettings();
   void loadSURFSettings();
   void loadGFTTSettings();
   
   void loadMFGSettings();
   void loadBASettings();
   
   void loadIntrinsicsSettings();
   void loadDistCoeffSettings();
};

#endif

