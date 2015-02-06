/* ---------------------------------------------------------------------------
   MFG - Multilayer Feature Graph for Robot Navigation

   This is the main file for C++ Implementation of MFG

   Author: Yan Lu (sinoluyan@gmail.com)
   @ NetBot Lab, TAMU (http://telerobot.cs.tamu.edu/)
   Created on Oct. 22, 2011
-----------------------------------------------------------------------------*/

//------------------------------ Header files ---------------------------------

#include <QCoreApplication>
#include <QDebug>

#include <iostream>
#include <math.h>

#include "settings.h"
#include "view.h"
#include "mfg.h"
#include "utils.h"

using namespace std;

//------------------------------ Global variables -----------------------------
int IDEAL_IMAGE_WIDTH;
double THRESH_POINT_MATCH_RATIO = 0.50;

double SIFT_THRESH_HIGH = 0.05;
double SIFT_THRESH_LOW  = 0.03; // for turning

double SIFT_THRESH = SIFT_THRESH_HIGH;

MfgSettings* mfgSettings;

//------------------------------- Main function -------------------------------
int main(int argc, char *argv[])
{
   // -----------  Qt GUI ------------
   QCoreApplication app(argc, argv);

   //MfgSettings mfgSettings; // TODO: arg-parse a cameraID (optional arg)
   mfgSettings = new MfgSettings();


   seed_xrand(mfgSettings->getPRNGSeed());  	
   srand(time(NULL));

   string imgName;			// first image name
   cv::Mat K, distCoeffs;	// distortion coeff: k1, k2, p1, p2
   int imIdLen = 5;			// n is the image number length,
   int ini_incrt = mfgSettings->getFrameStepInitial();
   int increment = mfgSettings->getFrameStep();
   int totalImg = 12500;

   // *** TODO: UPDATING ***
   imgName     = mfgSettings->getInitialImage().toStdString();
   K           = cv::Mat(mfgSettings->getIntrinsics()); // TODO: remove conversion
   distCoeffs  = cv::Mat(mfgSettings->getDistCoeffs()); // TODO: remove conversion
   std::cout << "K =\n" << K << std::endl;
   std::cout << "distCoeffs =\n" << distCoeffs << std::endl;
   std::cout << "getImageWidth() = " << mfgSettings->getImageWidth() << std::endl;

   // *** END UPDATING ***

   // Fix problem with image name ending in '\r'
   if (imgName[imgName.size()-1] == '\r')
      imgName.erase(imgName.size()-1);

   //------- initialize -------
   View view0(imgName, K, distCoeffs, 0, mfgSettings);
   qDebug() << "View initialized";
   view0.frameId = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());
   imgName = nextImgName(imgName, imIdLen, ini_incrt);
//   View view1(imgName, K, distCoeffs, 1, mfgSettings);
//   view1.frameId = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());

//   Mfg map(view0, view1, 10);

   Mfg map(view0, ini_incrt, distCoeffs);
   map.fps = 10;

   mfgSettings->setKeypointAlgorithm(KPT_GFTT); // use gftt tracking

   MfgThread mthread(mfgSettings);
   mthread.pMap = &map;
   mthread.imgName = imgName;
   mthread.increment = increment;
   mthread.totalImg = totalImg;
   mthread.imIdLen = imIdLen;
   mthread.K = K;
   mthread.distCoeffs = distCoeffs;

   mthread.start();

   QObject::connect(&mthread,SIGNAL(closeAll()),&app,SLOT(quit()));
   return app.exec();

}

