/* ---------------------------------------------------------------------------
MFG - Multilayer Feature Graph for Robot Navigation

This is the main file for C++ Implementation of MFG
Author: Yan Lu (sinoluyan@gmail.com) 
@ NetBot Lab, TAMU (http://telerobot.cs.tamu.edu/) 
Created on Oct. 22, 2011
Current version: 1.0
History:
-----------------------------------------------------------------------------*/

//------------------------------ Header files ---------------------------------

#include <QApplication>
#include <QDesktopWidget>
#include <math.h>
#include "window.h"
#include "mfg.h"
#include "mfg_utils.h"

using namespace std; 

//------------------------------ Global variables -----------------------------
int IDEAL_IMAGE_WIDTH;
double THRESH_POINT_MATCH_RATIO = 0.50;

double SIFT_THRESH_HIGH = 0.05;
double SIFT_THRESH_LOW  = 0.03; // for turning

double SIFT_THRESH = SIFT_THRESH_HIGH;

SysPara syspara;
vector< vector<double> > planeColors;

void SysPara::init()
{
	use_img_width = -1;
	kpt_detect_alg = 1; // intial to be 1

	gftt_max_ptnum = 300; 
	gftt_qual_levl = 0.01;
	gftt_min_ptdist = 3; // 3 fo bicocca, 5 for hrbb; 

	oflk_min_eigval = 0.001;
	oflk_win_size = 15; // hrbb 19, biccoca 15

	ini_increment = 26;// 60 for bicocca, 80 for hrbb
	frm_increment = 2; // 2 for bicocca, 4 for hrbb;
	kpt_desc_radius = 11; // 11 for bicocca, 21 for hrbb;

	nFrm4VptBA = 10;

	mfg_pt2plane_dist = 0.16;
	mfg_num_recent_pts = 400; // use recent points to discover planes
	mfg_num_recent_lns = 50; 
	mfg_min_npt_plane = 100; // min num of points to claim a new plane

	ba_weight_vp = 15;
	ba_weight_ln = 1;
	ba_weight_pl = 100;

	ba_use_kernel = true;
	ba_kernel_delta_pt = 1;
	ba_kernel_delta_vp = 1;
	ba_kernel_delta_ln = 3;
	ba_kernel_delta_pl = 1;

	angle_thresh_vp2d3d = 5; // degree
}

//------------------------------- Main function -------------------------------
int main(int argc, char *argv[])
{		
	// -----------  Qt GUI ------------ 
	QApplication app(argc, argv);
	Window win;
	win.resize(win.sizeHint());
	
	syspara.init();	
	for (int i=0; i<100; ++i) {
		vector<double> color(3);
		color[0] = 1;//(rand()%100)/100.0;
		color[1] = 1;//(rand()%100)/100.0;
		color[2] = 1;//(rand()%100)/100.0;
		planeColors.push_back(color);
	}

	srand(1);
//	srand((unsigned)time(NULL));
	string imgName;			// first image name
	cv::Mat K, distCoeffs;	// distortion coeff: k1, k2, p1, p2
	int imIdLen = 5;			// n is the image number length,
	int ini_incrt = syspara.ini_increment;
	int increment = syspara.frm_increment;
	int totalImg = 12500;
	getConfigration (&imgName, K, distCoeffs, syspara.use_img_width) ;
	
	//------- initialize -------
	View view0(imgName, K, distCoeffs, 0);
	view0.frameId = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());
	imgName = nextImgName(imgName, imIdLen, ini_incrt);
	View view1(imgName, K, distCoeffs, 1);
	view1.frameId = atoi (imgName.substr(imgName.size()-imIdLen-4, imIdLen).c_str());

	Mfg map(view0, view1, 30);	
	
	syspara.kpt_detect_alg = 3; // use gftt tracking

	MfgThread mthread;
	mthread.pMap = &map;
	mthread.imgName = imgName;
	mthread.increment = increment;
	mthread.totalImg = totalImg;
	mthread.imIdLen = imIdLen;
	mthread.K = K;
	mthread.distCoeffs = distCoeffs;

	mthread.start();
	// ------- plot ---------
	win.setMfgScene(&map);
	int desktopArea = QApplication::desktop()->width() *
		QApplication::desktop()->height();
	int widgetArea = win.width() * win.height();
	if (((float)widgetArea / (float)desktopArea) < 0.75f)
		win.show();
	else
		win.showMaximized();
	return app.exec();

}

