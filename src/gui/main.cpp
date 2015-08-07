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

// Qt
#include <QApplication>
#include <QDesktopWidget>
#include <QDebug>

// Standard library
#include <iostream>
#include <math.h>

// MFG
#include "settings.h"
#include "view.h"
#include "window.h"
#include "mfg.h"
#include "utils.h"
#include "random.h"

using namespace std;
using namespace cv;

int IDEAL_IMAGE_WIDTH; // image width in use
double THRESH_POINT_MATCH_RATIO = 0.50; // SIFT distance ratio threshold

// SIFT
double SIFT_THRESH_HIGH = 0.05;
double SIFT_THRESH_LOW  = 0.03; // for turning
double SIFT_THRESH = SIFT_THRESH_HIGH;

// Write MFG output
bool mfg_writing = false;

// MFG settings
MfgSettings *mfgSettings;

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

	// Create main window
    Window win;
    win.resize(win.sizeHint());

	// Read MFG settings file
    if (argc > 1)        
        mfgSettings = new MfgSettings(argv[1]);
    else
        mfgSettings = new MfgSettings();

	// Seed random number generators
    seed_xrand(mfgSettings->getPRNGSeed());
    srand(1);
    theRNG().state = 1;

	// Set frame information
	int imIdLen = 5;									// num of digits in frame file id
    int ini_incrt = mfgSettings->getFrameStepInitial(); // 1st frame step
    int increment = mfgSettings->getFrameStep();		// frame step
    int totalImg = 12500;								// total num of frames
    string imgName;										// first image name
	imgName = mfgSettings->getInitialImage().toStdString();
	
	// Initialize camera parameters
    Mat K;			// camera intrinsic matrix
	Mat distCoeffs;	// camera distortion coefficients (k1, k2, p1, p2)
    K = Mat(mfgSettings->getIntrinsics());
    distCoeffs = Mat(mfgSettings->getDistCoeffs());
    
	// Display camera parameters
	cout << "K =\n" << K << endl;
    cout << "distCoeffs =\n" << distCoeffs << endl;
    cout << "getImageWidth() = " << mfgSettings->getImageWidth() << endl;

    // Fix problem with image name ending in '\r'
    if (imgName[imgName.size() - 1] == '\r')
        imgName.erase(imgName.size() - 1);

    // Create 1st view
    View view0(imgName, K, distCoeffs, 0, mfgSettings);
    qDebug() << "View initialized";
    view0.frameId = atoi(imgName.substr(imgName.size() - imIdLen - 4, imIdLen).c_str());

	// Set next image
    imgName = nextImgName(imgName, imIdLen, ini_incrt);

	// If we are assuming fixed height from ground ...
    if (mfgSettings->getDetectGround())
    {
		// Create MFG (with first view)
        Mfg map(view0, ini_incrt, distCoeffs, 10);
        mfgSettings->setKeypointAlgorithm(KPT_GFTT); // use gftt tracking

		// Create, set, and start MFG thread
        MfgThread mthread(mfgSettings);
        mthread.pMap = &map;
        mthread.imgName = imgName;
        mthread.increment = increment;
        mthread.totalImg = totalImg;
        mthread.imIdLen = imIdLen;
        mthread.K = K;
        mthread.distCoeffs = distCoeffs;
        mthread.start();
        
	   	// Set MFG scene
		win.setMfgScene(&map);

        QObject::connect(&mthread, SIGNAL(closeAll()), &app, SLOT(quit()));
       
	   	// Show main window
		int desktopArea = QApplication::desktop()->width() * QApplication::desktop()->height();
        int widgetArea = win.width() * win.height();
        if (((float)widgetArea / (float)desktopArea) < 0.75f)
            win.show();
        else
            win.showMaximized();

        return app.exec();
    }
    else
    {
		// Create 2nd view
        View view1(imgName, K, distCoeffs, 1, mfgSettings);
        view1.frameId = atoi(imgName.substr(imgName.size() - imIdLen - 4, imIdLen).c_str());

		// Create MFG (with first two views)
		Mfg map(view0, view1, 10);
        mfgSettings->setKeypointAlgorithm(KPT_GFTT); // use gftt tracking

		// Create, set, and start MFG thread
        MfgThread mthread(mfgSettings);
        mthread.pMap = &map;
        mthread.imgName = imgName;
        mthread.increment = increment;
        mthread.totalImg = totalImg;
        mthread.imIdLen = imIdLen;
        mthread.K = K;
        mthread.distCoeffs = distCoeffs;
        mthread.start();
       
	   	// Set MFG scene
		win.setMfgScene(&map);
        
		QObject::connect(&mthread, SIGNAL(closeAll()), &app, SLOT(quit()));
        
	   	// Show main window
		int desktopArea = QApplication::desktop()->width() * QApplication::desktop()->height();
        int widgetArea = win.width() * win.height();
        if (((float)widgetArea / (float)desktopArea) < 0.75f)
            win.show();
        else
            win.showMaximized();

        return app.exec();
    }
}
