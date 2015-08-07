/////////////////////////////////////////////////////////////////////////////////
//
//  Multilayer Feature Graph (MFG), version 1.0
//  Copyright (C) 2011-2015 Yan Lu, Madison Treat, Dezhen Song
//  Netbot Laboratory, Texas A&M University, USA
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//
/////////////////////////////////////////////////////////////////////////////////

/********************************************************************************
 * MFG settings
 ********************************************************************************/



#ifndef CONFIG_H_
#define CONFIG_H_

#include <QCoreApplication>
#include <QSettings>
#include <QString>
#include <string.h>

#include <opencv2/core/core.hpp>

#include "consts.h"

typedef cv::Matx33d IntrinsicsMatrix;
typedef cv::Mat_<double> CoeffMatrix;
//typedef cv::Matx41d CoeffMatrix4;
//typedef cv::Matx51d CoeffMatrix5;

class MfgSettings : QObject
{
    Q_OBJECT
public:
    
    MfgSettings(QString _configFile = "mfgSettings.ini", QString _cameraID = "", QObject *_parent = 0);
    ~MfgSettings();

    void printAllSettings() const;
    void setKeypointAlgorithm(FeatureDetectionAlgorithm _alg);

    //---------------------------------------------------------------------------
    // Access functions
    //---------------------------------------------------------------------------
    // General settings
    int   getImageWidth() const
    {
        return imageWidth;
    }

    QString getCameraID() const
    {
        return cameraID;
    }
    QString getInitialImage() const
    {
        return initialImage;
    }
    uint64_t getPRNGSeed() const
    {
        return prngSeed;
    }

    IntrinsicsMatrix  getIntrinsics() const
    {
        return cameraIntrinsics;
    }
    CoeffMatrix       getDistCoeffs() const
    {
        return distCoeffs;
    }

    // Feature Point Detection algorithm
    FeatureDetectionAlgorithm getKeypointAlgorithm() const
    {
        return featureAlg;
    }
    double   getFeatureDescriptorRadius() const
    {
        return featureDescriptorRadius;
    }

    // SIFT
    float    getSiftThreshold() const
    {
        return siftThreshold;
    }

    // GFTT
    int      getGfttMaxPoints() const
    {
        return gfttMaxPoints;
    }
    double   getGfttQualityLevel() const
    {
        return gfttQuality;
    }
    double   getGfttMinimumPointDistance() const
    {
        return gfttMinPointDistance;
    }

    // Optic Flow
    double   getOflkMinEigenval() const
    {
        return oflkMinEigenval;
    }
    int      getOflkWindowSize() const
    {
        return oflkWindowSize;
    }

    // MFG
    double   getMfgPointToPlaneDistance() const
    {
        return mfgPointToPlaneDist;
    }
    int      getMfgNumRecentPoints() const
    {
        return mfgNumRecentPoints;
    }
    int      getMfgNumRecentLines() const
    {
        return mfgNumRecentLines;
    }
    int      getMfgPointsPerPlane() const
    {
        return mfgPointsPerPlane;
    }

    int      getFrameStep() const
    {
        return frameStep;
    }
    int      getFrameStepInitial() const
    {
        return frameStepInitial;
    }
    double   getVPointAngleThresh() const
    {
        return vpointAngleThresh;
    }
    double   getDepthLimit() const
    {
        return depthLimit;
    }
    int      getDetectGround() const
    {
        return detectGround;
    }

    // BA
    double   getBaWeightVPoint() const
    {
        return baWeightVPoint;
    }
    double   getBaWeightLine() const
    {
        return baWeightLine;
    }
    double   getBaWeightPlane() const
    {
        return baWeightPlane;
    }
    bool     getBaUseKernel() const
    {
        return baUseKernel;
    }
    double   getBaKernelDeltaVPoint() const
    {
        return baKernelDeltaVPoint;
    }
    double   getBaKernelDeltaPoint() const
    {
        return baKernelDeltaPoint;
    }
    double   getBaKernelDeltaLine() const
    {
        return baKernelDeltaLine;
    }
    double   getBaKernelDeltaPlane() const
    {
        return baKernelDeltaPlane;
    }
    int      getBaNumFramesVPoint() const
    {
        return baNumFramesVPoint;
    }

private:
    //---------------------------------------------------------------------------
    // General settings
    //---------------------------------------------------------------------------
    QString     configDir;     // path to config directory
    QString     settingsFile;  // path to settings file
    QSettings  *mfgSettings;   // settings object

    // Camera settings
    QString           cameraID;
    IntrinsicsMatrix  cameraIntrinsics;
    CoeffMatrix       distCoeffs;

    // Initial image settings
    int      imageWidth;
    QString  initialImage;

    // PRNG seed for XOR Shift* 64-bit algorithm, which in turn seeds the
    // 1024-bit algorithm, providing a 64-bit seed for a 1024-bit PRNG.
    uint64_t prngSeed;


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
    double   depthLimit;          // if triangulated point is too far, ignore it

    int      detectGround;        // detect ground plane for scale estimation:0,1

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

