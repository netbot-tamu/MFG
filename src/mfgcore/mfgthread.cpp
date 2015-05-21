
#include "mfgthread.h"

#include <math.h>

#include "mfg.h"
#include "mfgutils.h"
#include "export.h"
#include "utils.h"
#include "settings.h"
#include <opencv2/nonfree/nonfree.hpp> // For opencv2.4+,

void MfgThread::run()
{
	pMap->adjustBundle();
	int trialNo = 3;
	int threshPtPairNum;
	int thresh3dPtNum;
	double interval_ratio = 1;

	if(mfgSettings->getDetectGround()) {
		threshPtPairNum = 200;
		thresh3dPtNum = 30;
	} else {
		threshPtPairNum = 50;
		thresh3dPtNum = 7;
	}

	MyTimer timer;
	timer.start();
	bool finished = false;
	for(int i=0; !finished; ++i) {
		if(pMap->rotateMode()||pMap->angleSinceLastKfrm > 10*PI/180) {
			increment = max(1, mfgSettings->getFrameStep()/2);
			if(mfgSettings->getDetectGround()) {			
				interval_ratio = 0.1;
			} else {
				interval_ratio = 0.5;	
			}
		} else {
			increment = mfgSettings->getFrameStep();
			if(mfgSettings->getDetectGround()) {			
				interval_ratio = 0.55;
			} else {
				interval_ratio = 1;	
			}
		}
	
		bool toExpand = false;
		imgName = nextImgName(imgName, imIdLen, increment);
			
		if (!isFileExist(imgName)) {
			do {			
				imgName = prevImgName(imgName, imIdLen, 1);				
			} while(!isFileExist(imgName));
			if(imgName != pMap->views.back().filename)
				toExpand = true;
			finished = true;
		}
		View v(imgName, K, distCoeffs, mfgSettings);
		int fid = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());

		if(fid-pMap->views[0].frameId > totalImg)			break;

		// --- determine keyframe or not ---
		if(fid-pMap->views.back().frameId >  // interval limit
			(pMap->views[1].frameId-pMap->views[0].frameId)*interval_ratio
		   && fid-pMap->views.back().frameId >1
		   && !finished) {
				toExpand = true;
				imgName = prevImgName(imgName, imIdLen, increment);
		} else if (isKeyframe(*pMap, v, threshPtPairNum, thresh3dPtNum) && !finished) { // previous frame is selected
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
		} else if(!toExpand && !finished)
			continue;

		if(toExpand) { // to avoid Motion blurred image, search around
			int fid = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());
			cout<<"\nframe:"<<fid<<endl;
			MyTimer tm; tm.start();
         	View imgView(imgName,K,distCoeffs,-1, mfgSettings);
        // 	tm.end(); cout<<"view setup time "<<tm.time_ms<<" ms"<<endl;
			pMap->expand(imgView,fid);
		}
	}
	timer.end();
	cout<<"total time = "<<timer.time_s<<"s"<<endl;
   QString exportDir = mfgSettings->getOutputDir();
	exportCamPose (*pMap, exportDir + "/camPose.txt");
//	pMap->exportAll(exportDir);
	emit closeAll();
}


