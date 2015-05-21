/*
 * File:   mfgthread.h
 * Author: madtreat
 *
 * Created on January 23, 2015, 2:39 PM
 */

#ifndef MFGTHREAD_H
#define MFGTHREAD_H

#include <QThread>
#include <string>
#include <opencv2/core/core.hpp>

class Mfg;
class MfgSettings;

class MfgThread : public QThread
{
   Q_OBJECT

signals:
   void closeAll();

protected:
   void run();

public:
   Mfg* pMap;
   std::string imgName;			// first image name
   cv::Mat K, distCoeffs;	// distortion coeff: k1, k2, p1, p2
   int imIdLen;			// n is the image number length,
   int ini_incrt;
   int increment;
   int totalImg;

   MfgThread(MfgSettings* _settings) : mfgSettings(_settings) {}

private:
   MfgSettings* mfgSettings;
};

#endif	/* MFGTHREAD_H */

