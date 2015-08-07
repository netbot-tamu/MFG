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

#include "mfg.h"

// OpenGL
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

// Standard library
#include <math.h>
#include <fstream>

// MFG
#include "mfgutils.h"
#include "export.h"
#include "utils.h"
#include "settings.h"

// OpenCV
#include <opencv2/nonfree/nonfree.hpp> // for OpenCV 2.4+

void MfgThread::run()
{
	
    pMap->adjustBundle();

    int trialNo = 3;
    int threshPtPairNum; //number of point matches between two frames
    int thresh3dPtNum;   // number of observed 3D points in a frame
    double interval_ratio = 1;

	
    if (mfgSettings->getDetectGround()) // If we are assuming fixed height from ground
    {
        threshPtPairNum = 200; 	
        thresh3dPtNum = 30;		
    }
    else
    {
        threshPtPairNum = 50; 	
        thresh3dPtNum = 7;		
    }

    MyTimer timer;
    timer.start();

   	bool finished = false;
    
	// Expand MFG until no more frames to process
	for (int i = 0; !finished; ++i)
    {
		// If the camera is rotating or moved enough...
        if (pMap->rotateMode() || pMap->angleSinceLastKfrm > 10 * PI / 180)
        {
			// Reduce frame step
            increment = max(1, mfgSettings->getFrameStep() / 2);

            if (mfgSettings->getDetectGround())
                interval_ratio = 0.1;
            else
                interval_ratio = 0.5;
        }
        else
        {
			// Use fixed frame step
            increment = mfgSettings->getFrameStep();

            if (mfgSettings->getDetectGround())
                interval_ratio = 0.55;
            else
                interval_ratio = 1;
        }

        bool toExpand = false;

		// Jump to next frame
        imgName = nextImgName(imgName, imIdLen, increment);

		// If we jumped too far and frame does not exist...
        if (!isFileExist(imgName))
        {
			// Step back until frame is found
            do { imgName = prevImgName(imgName, imIdLen, 1); } while (!isFileExist(imgName));

			// If this frame is not processed yet...	
            if (imgName != pMap->views.back().filename)
                toExpand = true;

            finished = true; // this will be the last loop
        }

		// Create view
        View v(imgName, K, distCoeffs, mfgSettings);
        int fid = atoi(imgName.substr(imgName.size() - imIdLen - 4, imIdLen).c_str());

		// If we reached maximum num of images to process...
        if (fid - pMap->views[0].frameId > totalImg)			
			break;

        // If camera did not move enough (based on num of frames)...
        if (fid - pMap->views.back().frameId > (pMap->views[1].frameId - pMap->views[0].frameId) * interval_ratio // interval limit
			&& fid - pMap->views.back().frameId > 1 
			&& !finished)
        {
            toExpand = true;
            imgName = prevImgName(imgName, imIdLen, increment);
        }
        else if (isKeyframe(*pMap, v, threshPtPairNum, thresh3dPtNum) && !finished) // seletct previous frame as key frame
        {
            string selName;
            string oldName = imgName;

			// Check subsequent frames to see if the reduction of feature correspondence
			// is not caused by a sudden image blur. In other words, we want to skip frames 
			// with motion blur.
			
			bool found = false; // true when frame has enough feature correspondence
          
		 	// Check next frames
		    for (int j = 0; j < trialNo; ++j)
            {
				// Jump to next frame
                int tryinc = max(int(increment / 4), 1);
                imgName = nextImgName(imgName, imIdLen, tryinc);

                if (!isFileExist(imgName)) // no more frames
					break;

				// Check if this frame should be considered as a key frame
                View v(imgName, K, distCoeffs, mfgSettings);
                if (!isKeyframe(*pMap, View(imgName, K, distCoeffs, mfgSettings), 50, thresh3dPtNum))
                {
                    selName = imgName;
                    found = true;
                    break;
                }
            }

			// If camera did not move enough...
            if (found && pMap->angleSinceLastKfrm < 15 * PI / 180)
            {
				// Skip this frame
                imgName = selName;
                continue;
            }
            else // camera moved enough
            {
				// Set current frame as key frame
                int fid = atoi(oldName.substr(oldName.size() - imIdLen - 4, imIdLen).c_str());
                if (fid == pMap->views.back().frameId + increment)
                    imgName = oldName;
                else
                {
                    imgName = prevImgName(oldName, imIdLen, increment);
                    int fid = atoi(imgName.substr(imgName.size() - imIdLen - 4, imIdLen).c_str());
                }

				// Add this frame to MFG
                toExpand = true;
            }
        }
        else if (!toExpand && !finished)
		{
            continue;
		}

		// Expand MFG?
        if (toExpand) // to avoid motion blurred image, search around
        {
            int fid = atoi(imgName.substr(imgName.size() - imIdLen - 4, imIdLen).c_str());
            cout << "\nframe:" << fid << endl;
           
		   	// Create view
			MyTimer tm;
            tm.start();
			View imgView(imgName, K, distCoeffs, -1, mfgSettings);
            // tm.end(); 
			// cout << "view setup time " << tm.time_ms << " ms" << endl;
           
		   	// Expand MFG with current view
			pMap->expand(imgView, fid);
        }
    }

    timer.end();
    cout << "total time = " << timer.time_s << "s" << endl;

	// Save camera trajectory to file
    exportCamPose(*pMap, "camPose.txt") ;
	//pMap->exportAll("MFG");

	// Signal that MFG thread finished
	emit closeAll();
}
