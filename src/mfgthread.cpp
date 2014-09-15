#include <QtGui>
#include <QtOpenGL/QtOpenGL>

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

#include "glwidget.h"
#include "mfg.h"
#include "mfg_utils.h"
#include "opencv2/nonfree/nonfree.hpp" // For opencv2.4+,

#define DETECT_BLUR
//#define FIX_KEYFRAME
extern SysPara syspara;

vector<int> keyFrameIdx ();
void MfgThread::run()
{
#ifdef FIX_KEYFRAME
	pMap->adjustBundle();
	vector<int> kfIdx =  keyFrameIdx ();
	string head = imgName.substr(0,imgName.size()-imIdLen-4);
	string tail = imgName.substr(imgName.size()-4, 4);	
	
	MyTimer timer;
	timer.start();
	for(int i =0; i < kfIdx.size(); ++i) {
		int idx = kfIdx[i];
		string idxStr = num2str(idx);
		for (int j = 0; j < imIdLen- int(log10(double(idx))) -1; ++j ) {
			idxStr = '0'+ idxStr;
		}
		string imgName = head + idxStr + tail;
		cout<<"\n<"<<i<<","<<idx<<"> \n";
		pMap->expand(View(imgName,K,distCoeffs, -1), idx);
	}

	timer.end();
	cout<<"total time = "<<timer.time_s<<"s"<<endl;

#else
	pMap->adjustBundle();
	int trialNo = 3;
	int threshPtPairNum = 50;
	int thresh3dPtNum = 7;
	double interval_ratio = 1;
	
	MyTimer timer;
	timer.start();
	for(int i=0; true; ++i) {		
		if(pMap->rotateMode()||pMap->angleSinceLastKfrm > 10*PI/180) {
			increment = max(1, syspara.frm_increment/2);
			interval_ratio = 0.5;
		} else {
			increment = syspara.frm_increment;
			interval_ratio = 1;
		}
		cout<<"<"<<i<<"> ";
		bool toExpand = false;
		imgName = nextImgName(imgName, imIdLen, increment);
		if (!isFileExist(imgName)) break;
		View v(imgName, K, distCoeffs);
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
				View v(imgName, K, distCoeffs);
				if(!isKeyframe(*pMap, View(imgName, K, distCoeffs), 50, thresh3dPtNum)) {
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
				cout<<"blured image, select previous ";
			}	else if (1.5* curtfeat.size() < nextfeat.size()) {
				imgName = next;
				cout<<"blured image, select next ";
			}
			
#endif
			int fid = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());
			cout<<"frame:"<<fid<<endl;
			pMap->expand(View(imgName,K,distCoeffs, -1),fid);
		}
	}
	timer.end();
	cout<<"total time = "<<timer.time_s<<"s"<<endl;
	exportCamPose (*pMap, "campose.txt") ;
#endif
}

vector<int> keyFrameIdx ()
{

	 int frmIdx[] = { // 50 frames
        1730,	1780,	1831,	1880,	1930,	1980,	2032,	2080,	2131,	2180,
		2231,	2280,	2331,	2380,	2432,	2480,	2530,	2580,	2630,	2680, 
		2730,	2780,	2830,	2879,	2930,	2980,	3030,	3081,	3130,	3180, 
		3230,	3281,	3330,	3380,	3430,	3480,	3531,	3581,	3630,	3680,
		3730,	3780, 	3830,   3870,   3909,               
		
		3940,        3979,        4020,       4060,       4100,        4140,		  4180,  
	//	4220,        4258,		  4289,		   4317,		4342,		4367,		  4396,  4426,
	    4220,        4258,		  4289,		   4317,		4342,		4378,		  4396,  4426,
		4465,		 4505,		  4545 ,	   4585,		4619,	4644, 4668,	  4700,  4728,
		
		4760,	4800, 4840,	4880, 	4920, 4962,	5000,	5040,  5080,	5120,	5160, 5200,	5242, 	5280, 5320,
		5360,	5400, 5440,	5480,	5520,	5560, 5600,  5640, 5680,	5720,	5760,	5800,	5840,  5879, 5921,
		5960,	5999, 6040,	6079,	6120,  6160,	6199,	6240,  6281,	6319,	6360, 6400,	6440,	6481, 6520,
		6557,	6600, 6640,	6680,	6720,	6760, 
		
		6800,        6840,		  6880,        6920,        6960,        7000,        7040,        7080,        7120,
        7160,        7199,        7240,        7280,        7320,       7360,        7400, 
        7441,        7479,        7520,		   7560, 7597, 7634,	7658,7680,	7706,	7746,	7786,	7824, 7867, 
		7907,        7947,        7988,        8027,        8070,        8107,		8147,

		8190,	8239,	8289,	8340,	8390,	8439,	8490,	8540,	8590,	8640,
		8690,	8740,	8789,	8840,	8890,	8940,	8990,	9040,	9090,	9139,
		9190,	9239,	9290,	9340,	9390,	9440,	9490,	9540,	9590,	9640,
		9690,	9739,	9790,	9841,	9890,	9940,	9990,	10037,	10089,	10140,
		10190,	10240,  10290,	10340,	10388,	
		
	   10440,		10480,		10520,		
       10560,       10600,       10639,       10680,       10720,       10760,       10789, 10819,
       10849,       10879,       10920,       10961,       11002,       11040,       11080, 

	   11120,       11158,       11200,       11240,       11280,       11320,       11360, 11380,
       11400,       11440,       11480,       11520,       11560,       11600,       11640,
       11680,       11720,       11760,       11800,       11840,       11880,       11920,
       11960,       12000,       12040,       12080,       12120,       12160,       12200,
       12240,       12280,       12320,       12360,       12400,       12440,       12480,
       12520,       12560,       12600,       12640,       12680,       12720,       12760,
       12800,       12840,       12880,       12920,       12960,       13000,       13040,
       13080,       13120,       13160,       13200,       13240,       13280,       13320,
       13360,       13400,       13440,       13480,       13520,       13560,       13600,
       13640,       13680,       13720,       13760,       13800,       13840,       13880
  //     13920,       13960,       14000,       14040,       14080,       14120,       14160,
  //     14200,       14240,       14280,       14320,       14360,       14400
	};

	 int frmIdx2[] = { // 60 frames
/*		920, 960, 1000, 1030, 1060, 1090, 1120, 1150, 1180, 1210, 1240, 1270,  
		920, 960, 1000, 1040, 1080, 1121, 1145, 1185, 1225, 1262, 
		1302,        1342,        1382,        1422,        1462,        1502,        1542,
        1582,        1622,        1662,        
		1740,		 1800,		  1860,		   1920,     1980, 2040, 2100, 2160, 2220,
		2280, 2340,  2400,  2460, 2520,  2580,  2640,  2700,  2760, 2840,	2900,	2960,
		3020,	3080,	3140,	3200,	3260,	3320,	3380,	3440,	3500,	3560,
		3620,	3680,	3740,	3800,	3860,	3920,

		3950,        3980,        4020,        4060,        4100,        4140,		  4180,
		4218,        4258,		  4298,		   4317,		4357,		 4387,		  4425,
		4465,		 4505,		  4545 ,	   4585,*/		4620,		 4660,		  4700, 
		
		4760,	4820,	4880,	4940,	5000,	5060,	5120,	5180,	5240,	5300, 
		5360,	5420,	5480,	5520,	5560, 5600,  5640, 5680,	5720,	5780,	5840,	5900, 
		5960,	6020,	6080,	6140,	6200,	6260,	6320,	6380,	6440,	6500,
		6560,	6620,	6680,	6740,
		
		6800,        6840,		  6880,        6920,        6960,        7000,        7040,        7080,        7120,
        7160,        7200,        7240,        7280,        7320,       7360,        7400,
        7440,        7480,        7520,		   7560, 7597, 7634,	7658,7680,	7707,	7747,	7787,	7827, 7867, 
		7907,        7947,        7987,        8027,        8067,        8107,        8167,
        8227,        8287,        8347,        8407,        8467,        8527,        8567,        
		8607,  8657,      8707,        8747,   8787,     8827,        8867,   8907,     8947,        9007,
        9067,        9127,        9187,        9247,        9307,        9367,        9427,        
		9487,        9547,        9607,        9667,        9727,        9787,        9827,
        9867,        9907,        9947,        9987,        10010,	
		10050,       10090,      10130,       10170,       10210,       10250,       10290,
		10330,       10370,      
		
	   10400,	   10440,		10480,		10520,
       10560,       10600,       10640,       10680,       10720,       10760,       10790, 10820,
       10850,       10880,       10920,       10960,       11000,       11040,       11080,
       11117,       11157,       11200,       11240,       11280,       11320,       11360,
       11400,       11440,       11480,       11520,       11560,       11600,       11640,
       11680,       11720,       11760,       11800,       11840,       11880,       11920,
       11960,       12000,       12040,       12080,       12120,       12160,       12200,
       12240,       12280,       12320,       12360,       12400,       12440,       12480,
       12520,       12560,       12600,       12640,       12680,       12720,       12760,
       12800,       12840,       12880,       12920,       12960,       13000,       13040,
       13080,       13120,       13160,       13200,       13240,       13280,       13320,
       13360,       13400,       13440,       13480,       13520,       13560,       13600,
       13640,       13680,       13720,       13760,       13800,       13840,       13880,
       13920,       13960,       14000,       14040,       14080,       14120,       14160,
       14200,       14240,       14280,       14320,       14360,       14400
		};

	 vector<int> frameIdx (frmIdx, frmIdx + sizeof(frmIdx) / sizeof(int) );

	 return frameIdx;
}
