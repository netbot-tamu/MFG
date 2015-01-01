
#include "mfg.h"

//#include <QtGui>
//#include <QtOpenGL/QtOpenGL>

//#include <gl/GLU.h>
// replaced with:
#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#elif __linux__
#include <GL/glut.h>
#include <GL/gl.h>
#else
#include <gl/glut.h>
#include <gl/gl.h>
#endif

#include <math.h>
#include <fstream>

//#include "glwidget.h"
#include "mfgutils.h"
#include "export.h"
#include "utils.h"
#include "settings.h"
#include <opencv2/nonfree/nonfree.hpp> // For opencv2.4+,

#define DETECT_BLUR
void MfgThread::run()
{
	pMap->adjustBundle();
	int trialNo = 3;
	int threshPtPairNum = 50;
	int thresh3dPtNum = 7;
	double interval_ratio = 1;

	MyTimer timer;
	timer.start();
	for(int i=0; true; ++i) {
		if(pMap->rotateMode()||pMap->angleSinceLastKfrm > 10*PI/180) {
			increment = max(1, mfgSettings->getFrameStep()/2);
			interval_ratio = 0.5;
		} else {
			increment = mfgSettings->getFrameStep();
			interval_ratio = 1;
		}
//		cout<<"<"<<i<<"> ";
		bool toExpand = false;
		imgName = nextImgName(imgName, imIdLen, increment);
		if (!isFileExist(imgName)) break;
		View v(imgName, K, distCoeffs, mfgSettings);
		int fid = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());

		if(fid-pMap->views[0].frameId > totalImg)			break;

		// --- determine keyframe or not ---
		if(fid-pMap->views.back().frameId >  // interval limit
			(pMap->views[1].frameId-pMap->views[0].frameId)*interval_ratio) {
				toExpand = true;
				imgName = prevImgName(imgName, imIdLen, increment);
		} else if (isKeyframe(*pMap, v, threshPtPairNum, thresh3dPtNum)) { // previous frame is selected
			string selName;
			bool found = false; // found a view with more overlap
			string oldName = imgName;
			for(int j = 0; j < trialNo; ++j) {
				int tryinc = max(int(increment/4),1);
				imgName = nextImgName(imgName, imIdLen, tryinc);
				if (!isFileExist(imgName)) break;
				View v(imgName, K, distCoeffs, mfgSettings);
				if(!isKeyframe(*pMap, View(imgName, K, distCoeffs, mfgSettings), 50, thresh3dPtNum)) {
					selName = imgName;
					found = true;
					break;
				}
			}
			if (found && pMap->angleSinceLastKfrm < 15*PI/180) {
				imgName = selName;
				continue;
			} else {
				int fid = atoi (oldName.substr(oldName.size()-imIdLen-4, imIdLen).c_str());
				if (fid == pMap->views.back().frameId+increment) {
					imgName = oldName;
				} else {
					imgName = prevImgName(oldName, imIdLen, increment);
					int fid = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());
				}
				toExpand = true;
			}
		} else
			continue;

		if(toExpand) { // to avoid Motion blurred image, search around
#ifdef DETECT_BLUR // avoiding blurred image is VERY helpful in practice!

			string prev, curt, next;
			vector<cv::KeyPoint> prevfeat, curtfeat, nextfeat;
			cv::SurfFeatureDetector		featDetector;

			prev = prevImgName(imgName, imIdLen, 1);
			next = nextImgName(imgName, imIdLen, 1);

			cv::Mat previmg = cv::imread(prev,1);
			cv::Mat curtimg = cv::imread(imgName,1);
			cv::Mat nextimg = cv::imread(next,1);

			featDetector.detect(previmg,  prevfeat);
			featDetector.detect(curtimg,  curtfeat);
			featDetector.detect(nextimg,  nextfeat);

			if (prevfeat.size() > 1.5* curtfeat.size() ) {
				imgName = prev;
	//			cout<<"blured image, select previous ";
			}	else if (1.5* curtfeat.size() < nextfeat.size()) {
				imgName = next;
	//			cout<<"blured image, select next ";
			}

#endif
			int fid = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());
			cout<<"\nframe:"<<fid<<endl;
         View imgView(imgName,K,distCoeffs,-1, mfgSettings);
			pMap->expand(imgView,fid);
		}
	}
	timer.end();
	cout<<"total time = "<<timer.time_s<<"s"<<endl;
	exportCamPose (*pMap, "camPose.txt") ;
}


