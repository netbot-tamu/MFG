#include "utils.h"
#include "consts.h"
#include "random.h"

#include <vector>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

extern int IDEAL_IMAGE_WIDTH;

using namespace std;

int computeSubPSR (cv::Mat* xGradient, cv::Mat* yGradient,
        cv::Point2d p, double s, cv::Point2d g, vector<double>& vs) {
   /* input: p - 2D point position
    s - side length of square region
    g - unit vector of gradient of line
    output: vs = (v1, v2, v3, v4)
    */
   double tl_x = floor(p.x - s/2), tl_y = floor(p.y - s/2);
   if (tl_x < 0 || tl_y < 0 ||
           tl_x+s+1 > xGradient->cols || tl_y+s+1 > xGradient->rows)
      return 0; // out of image
   double v1=0, v2=0, v3=0, v4=0;
   for (int x  = tl_x; x < tl_x+s; ++x) {
      for (int y = tl_y; y < tl_y+s; ++y) {
         //			cout<< xGradient->at<double>(y,x) <<","<<yGradient->at<double>(y,x) <<endl;
         //			cout<<"("<<y<<","<<x<<")"<<endl;
         double tmp1 =
         xGradient->at<double>(y,x)*g.x + yGradient->at<double>(y,x)*g.y;
         double tmp2 =
         xGradient->at<double>(y,x)*(-g.y) + yGradient->at<double>(y,x)*g.x;
         if ( tmp1 >= 0 )
            v1 = v1 + tmp1;
         else
            v2 = v2 - tmp1;
         if (tmp2 >= 0)
            v3 = v3 + tmp2;
         else
            v4 = v4 - tmp2;
      }
   }
   vs.resize(4);
   vs[0] = v1; vs[1] = v2;
   vs[2] = v3; vs[3] = v4;
   return 1;
}

int computeMSLD (LineSegmt2d& l, cv::Mat* xGradient, cv::Mat* yGradient)
// compute msld and gradient
{
	cv::Point2d gradient = l.getGradient(xGradient, yGradient);
	l.gradient = gradient;
	int s = 5 * xGradient->cols/800.0;
	double len = l.length();

	vector<vector<double>> GDM;
	double step = 1; // the step length between sample points on line segment
	for (int i=0; i*step < len; ++i) {
		vector<double> col, psr(4);
		col.clear();
		cv::Point2d pt =    // compute point position on the line
              l.endpt1 + (l.endpt2 - l.endpt1) * (i*step/len);
		bool fail = false;
		for (int j=-4; j <= 4; ++j ) { // 9 PSR for each point on line
			psr.clear();
			if (computeSubPSR (xGradient, yGradient, pt+j*s*gradient, s, gradient, psr)) {
				col.push_back(psr[0]);
				col.push_back(psr[1]);
				col.push_back(psr[2]);
				col.push_back(psr[3]);
			} else
				fail = true;
		}
		if (fail)
			continue;
		GDM.push_back(col);
	}

	cv::Mat MS(72, 1, CV_64F);
	if (GDM.size() ==0 ) {
		for (int i=0; i<MS.rows; ++i)
			MS.at<double>(i,0) = xrand(); // if not computable, assign random num
		l.msldDesc = MS;
		return 0;
	}

	double gauss[9] = { 0.24142,0.30046,0.35127,0.38579,0.39804,
   0.38579,0.35127,0.30046,0.24142};
	for (int i=0; i < 36; ++i) {
		double sum=0, sum2=0, mean, std;
		for (int j=0; j < GDM.size(); ++j) {
			GDM[j][i] = GDM[j][i] * gauss[i/4];
			sum += GDM[j][i];
			sum2 += GDM[j][i]*GDM[j][i];
		}
		mean = sum/GDM.size();
		std = sqrt(abs(sum2/GDM.size() - mean*mean));
		MS.at<double>(i,0)		= mean;
		MS.at<double>(i+36, 0)	= std;
	}
	// normalize mean and std vector, respectively
	MS.rowRange(0,36) = MS.rowRange(0,36) / cv::norm(MS.rowRange(0,36));
	MS.rowRange(36,72) = MS.rowRange(36,72) / cv::norm(MS.rowRange(36,72));


	for (int i=0; i < MS.rows; ++i) {
		if (MS.at<double>(i,0) > 0.4)
			MS.at<double>(i,0) = 0.4;
	}
	MS = MS/cv::norm(MS);
	l.msldDesc.create(72, 1, CV_64F);
	l.msldDesc = MS;
   if(MS.at<double>(0)!=MS.at<double>(0))
	{
      //   std::cout << "enter 1 to continue...";
      //   int tmp; cin>>tmp;
	}
	return 1;
}

int isPtInLineNeighbor (IdealLine2d line, cv::Point2d pt, double imageWidth);

void matchLinesByPointPairs (double imWidth,
        vector<IdealLine2d>& lines1,vector<IdealLine2d>& lines2,
        FeaturePointPairs& pointPairs,
        vector<vector<int>>& linePairIdx)
// algorithm: Line Matching leveraged by Point Correspondences (CVPR2010)
// img1 and img2 need be gray images
{
	double lsLenThresh = 0;//imWidth/100.0;

	// ======= Find neighborhood points for lines in img1 =========
	vector<vector<int>> lhsPairIdx1, rhsPairIdx1;
	for (int i=0; i<lines1.size(); ++i) {
		//		lines1[i].getGradient(&xGradImg1,&yGradImg1);
		cv::Point2d nDirect = cv::Point2d(
              lines1[i].lineEq().at<double>(0),
              lines1[i].lineEq().at<double>(1));
		cv::Point2d lineDirect =
              cv::Point2d(nDirect.y, -nDirect.x) * (1/cv::norm(nDirect));
		cv::Point2d midPt = (lines1[i].extremity1 + lines1[i].extremity2)*0.5;
		double len = lines1[i].length(); //line length
		vector<int> lhsIdx, rhsIdx;
		if (len >= lsLenThresh) {
			for (int j=0; j<pointPairs.size(); ++j) {
				// check if in neighborhood, left or right
				int neigh = isPtInLineNeighbor
            (lines1[i], pointPairs[j][0], imWidth);
				if (neigh == 1)
					lhsIdx.push_back(j);
				if (neigh == 2)
					rhsIdx.push_back(j);
			}
		}
		lhsPairIdx1.push_back(lhsIdx);
		rhsPairIdx1.push_back(rhsIdx);
	}
	// =============== compute similarity matrix ===================
	cv::Mat simMat = cv::Mat::zeros(lines1.size(), lines2.size(), CV_64F);
	cv::Mat neighborhood = cv::Mat::zeros
           (pointPairs.size(),lines2.size(), CV_64F)-1; // initialize to -1

	double distThresh = 0.05;// threshold for pt to line distance when computing
	// similarity, pt too close to line is discarded

	for (int i=0; i<lines1.size(); ++i) {
		vector<int> lIdx = lhsPairIdx1[i], rIdx = rhsPairIdx1[i];
		if (lIdx.size()>=2 || rIdx.size()>=2) {
			for (int j=0; j<lines2.size(); ++j) {
				// if two gradients are opposite, then set similarity to 0.
				if (lines1[i].gradient.dot(lines2[j].gradient)>0
                    && lines2[j].length() >= lsLenThresh) {
               vector<int> bothLhsIdx;
               for (int k=0; k<lIdx.size(); ++k) {
                  if (neighborhood.at<double>(lIdx[k],j)<0)
                     neighborhood.at<double>(lIdx[k],j) =
                     isPtInLineNeighbor(lines2[j],pointPairs[lIdx[k]][1],imWidth);
                  if (neighborhood.at<double>(lIdx[k],j)==1) {
                     bothLhsIdx.push_back(lIdx[k]);
                  }
               }
               double maxLeftMedian = 0;
               if (bothLhsIdx.size()>=2) // compute similarity - left
               {
                  for (int b=0; b<bothLhsIdx.size(); ++b) {
                     vector<double> sims;
                     // choose a reference point
                     cv::Point2d basisPt1 = pointPairs[bothLhsIdx[b]][0],
                             basisPt2 = pointPairs[bothLhsIdx[b]][1];
                     double basisDist1 = point2LineDist(lines1[i].lineEq(), basisPt1),
                             basisDist2 = point2LineDist(lines2[j].lineEq(), basisPt2);
                     if (basisDist1 < distThresh || basisDist2 < distThresh)
                        continue; // if dist between basispt to line is too
                     // small, it cannot be used.
                     for (int ii=0; ii<bothLhsIdx.size(); ++ii) {
                        if (ii==b)
                           continue;
                        cv::Point2d Pt1 = pointPairs[bothLhsIdx[ii]][0],
                                Pt2 = pointPairs[bothLhsIdx[ii]][1];
                        if (abs(Pt1.x-basisPt1.x)+abs(Pt1.y-basisPt1.y)<1e-6)
                           sims.push_back(0);
                        else {
                           double d1 = point2LineDist(lines1[i].lineEq(), Pt1)/basisDist1,
                                   d2 = point2LineDist(lines2[j].lineEq(), Pt2)/basisDist2;
                           sims.push_back(exp(-abs(d1-d2)));
                        }
                     }
                     // find median of sims
                     double median;
                     if(sims.size()<35) //for short vector, sort faster
                        sort(sims.begin(),sims.end());
                     else      // for long vector, nth_ is faster
                        nth_element(sims.begin(),
                                sims.begin()+sims.size()/2, sims.end());
                     if (sims.size()%2 == 0)
                        median = sims[sims.size()/2-1];
                     else
                        median = sims[sims.size()/2];
                     maxLeftMedian = max(maxLeftMedian,median);
                  }
               }

               vector<int> bothRhsIdx;
               for (int k=0; k<rIdx.size(); ++k) {
                  if (neighborhood.at<double>(rIdx[k],j)<0)
                     neighborhood.at<double>(rIdx[k],j) =
                     isPtInLineNeighbor(lines2[j],pointPairs[rIdx[k]][1],imWidth);
                  if(neighborhood.at<double>(rIdx[k],j)==2) {
                     bothRhsIdx.push_back(rIdx[k]);
                  }
               }
               double maxRightMedian = 0;
               if (bothRhsIdx.size()>=2) // compute similarity - right
               {
                  for (int b=0; b<bothRhsIdx.size(); ++b){
                     vector<double> sims;
                     cv::Point2d basisPt1 = pointPairs[bothRhsIdx[b]][0],
                             basisPt2 = pointPairs[bothRhsIdx[b]][1];
                     double basisDist1 = point2LineDist(lines1[i].lineEq(), basisPt1),
                             basisDist2 = point2LineDist(lines2[j].lineEq(), basisPt2);
                     if (basisDist1 < distThresh || basisDist2 < distThresh)
                        continue; // if dist between basispt to line is too
                     // small, it cannot be used.
                     for (int ii=0; ii<bothRhsIdx.size(); ++ii) {
                        if (ii==b)
                           continue;
                        cv::Point2d Pt1 = pointPairs[bothRhsIdx[ii]][0],
                                Pt2 = pointPairs[bothRhsIdx[ii]][1];
                        if (abs(Pt1.x-basisPt1.x)+abs(Pt1.y-basisPt1.y)<1e-6)
                           sims.push_back(0);
                        else {
                           double  d1 = point2LineDist(lines1[i].lineEq(), Pt1)/basisDist1,
                                   d2 = point2LineDist(lines2[j].lineEq(), Pt2)/basisDist2;
                           sims.push_back(exp(-abs(d1-d2)));
                        }
                     }
                     // find median of sims
                     if(sims.size()<35) //for short vector, sort faster
                        sort(sims.begin(),sims.end());
                     else      // for long vector, nth_ is faster
                        nth_element(sims.begin(),
                                sims.begin()+sims.size()/2, sims.end());
                     double median = 0;
                     if (sims.size()%2 == 0)
                        median = (sims[sims.size()/2-1]);
                     else
                        median = (sims[sims.size()/2]);
                     maxRightMedian = max(maxRightMedian,median);
                  }
               }
               simMat.at<double>(i,j)=max(maxRightMedian,maxLeftMedian);
				}
			}
		}
	}
	// ============  Identify matches based on simlarity matrix ==========
	for (int i=0; i<simMat.rows; ++i) {
		vector<int> onePairIdx;
		double maxVal;
		cv::Point maxPos;
		cv::minMaxLoc(simMat.row(i),NULL,&maxVal,NULL,&maxPos);
		if (maxVal > 0.95) {
			double maxV;
			cv::Point maxP;
			cv::minMaxLoc(simMat.col(maxPos.x),NULL,&maxV,NULL,&maxP);
			if (i==maxP.y) {    // commnent this for more potential matches
				onePairIdx.push_back(lines1[i].lid);
				onePairIdx.push_back(lines2[maxPos.x].lid);
            linePairIdx.push_back(onePairIdx);
			}
		}
	}
}


int isPtInLineNeighbor (IdealLine2d line, cv::Point2d pt, double imageWidth)
// check if point is in the neighborhood of line,
// input:  a line, a point
// output: 0, if pt not in neiborhood of line
//		   1, if pt in leftneighbor  -
//		   2, if pt in rightneighbor -
// Note, before calling, line.gradient must have been computed !!
{
	double threshDistToLine
   = min(imageWidth/12.0,100.0),//THRESH_LINE_NEIGHBOR_WIDTH,
           threshDistAlongLine;

	cv::Point2d nDirect = cv::Point2d(line.lineEq().at<double>(0),
           line.lineEq().at<double>(1));
	cv::Point2d lineDirect =
           cv::Point2d(nDirect.y, -nDirect.x) * (1/cv::norm(nDirect));
	cv::Point2d midPt = (line.extremity1 + line.extremity2)*0.5;

	double len = line.length(); //line length
	cv::Point2d diffVect = pt - midPt;
	double distAlongLine = abs(diffVect.dot(lineDirect))
   /sqrt(double(lineDirect.x*lineDirect.x+lineDirect.y*lineDirect.y)),
           distToLine = point2LineDist(line.lineEq(),pt);
	threshDistAlongLine		= len/2;
	if (distAlongLine<=threshDistAlongLine && distToLine<=threshDistToLine) {
		cv::Point2d gradPt = line.gradient*100+cv::Point2d(midPt);
		double gradSign = cvpt2mat(gradPt).dot(line.lineEq()),
              ptSign = cvpt2mat(pt).dot(line.lineEq());
		if ((gradSign>0)^(ptSign>0)) //XOR
			return 1;
		else
			return 2;
	} else
		return 0;
}


vector<cv::Point2d> sampleFromLine (IdealLine2d l, double stepSize)
{
	int stepNum = ceil(l.length()/stepSize);
	vector<cv::Point2d> samples;

	cv::Point2d step = stepSize*
           (l.extremity2 - l.extremity1)*(1/cv::norm(l.extremity2 - l.extremity1));
	for (int i=0; i<stepNum; ++i) {
		samples.push_back(l.extremity1 + i*step);
		cout<<samples.back()<<endl;
	}
	samples.push_back(l.extremity2);
	return samples;
}


vector<vector<int>> F_guidedLinematch (cv::Mat F, vector<IdealLine2d> lines1,
        vector<IdealLine2d> lines2, cv::Mat img1, cv::Mat img2)
{
	double patchDiameter = 20; // point area diameter for featextractor
	cv::SurfDescriptorExtractor featExtractor;
	int	sampleNum = 10; // smaple points per line
	cv::Mat scores =
           cv::Mat::zeros(lines1.size(), lines2.size(),CV_64F) + 1e6;
	vector<vector<int>> linePairIdx;


	cv::Mat gImg1, gImg2;
	if (img1.dims >2) {
		cv::cvtColor(img1, gImg1, CV_RGB2GRAY);
		cv::cvtColor(img2, gImg2, CV_RGB2GRAY);
	} else	{
		gImg1 = img1;
		gImg2 = img2;
	}

	for (int i=0; i < lines1.size(); ++i) {

		double angle_i = -1;
		double minScore = 1e6;
		int minj;
		cv::Point2d direct1 = cv::Point2d(lines1[i].lineEq().at<double>(0),
              lines1[i].lineEq().at<double>(1));
		direct1 = direct1 * (1/cv::norm(direct1));
		if(lines1[i].gradient.dot(direct1) <0)
			direct1 = direct1*(-1);
		if( direct1.y >0 )
			angle_i = acos(direct1.x)*180/PI;
		else
			angle_i = (2*PI-acos(direct1.x))*180/PI;

		sampleNum = max(10, (int)ceil(2*lines1[i].length()/patchDiameter));
		vector<cv::Point2d> pts1 = sampleFromLine (lines1[i], sampleNum);

		for (int j=0; j<lines2.size(); ++j) {
			// compute matching score for each pair
			double angle_j = -1;
			vector<cv::KeyPoint> kpts1(0), kpts2(0);
			cv::Mat desc1, desc2;

			cv::Point2d direct2 = cv::Point2d(
                 lines2[j].lineEq().at<double>(0), lines2[j].lineEq().at<double>(1));
			direct2 = direct2 * (1/cv::norm(direct2));
			if(lines2[j].gradient.dot(direct2) <0)
				direct2 = direct2*(-1);
			if( direct2.y >0 )
				angle_j = acos(direct2.x)*180/PI;
			else
				angle_j = (2*PI-acos(direct2.x))*180/PI;

			// == gradient orientation check
			if (direct1.dot(direct2) < 0)
				continue;

			// == MSLD similarity check
			if (compMsldDiff(lines1[i],lines2[j]) > 0.8)
         {continue;}

			// == line (parallel) distance check
			if (aveLine2LineDist(lines1[i],lines2[j]) > img1.cols/5.0)
			{	continue;}

			for (int k=0; k<pts1.size(); ++k) {
				// ensure sift-descriptor can be computed
				if (pts1[k].x + patchDiameter/2 > img1.cols ||
                    pts1[k].x - patchDiameter/2 < 1 ||
                    pts1[k].y + patchDiameter/2 > img1.rows ||
                    pts1[k].y - patchDiameter/2 < 1)
					continue;

				cv::KeyPoint kp1(pts1[k], patchDiameter, angle_i);

				cv::Point2d p2 = mat2cvpt(
                    lines2[j].lineEq().cross(F*cvpt2mat(pts1[k],1)));

				// ensure sift-descriptor can be computed
				if (p2.x + patchDiameter/2 > img1.cols ||
                    p2.x - patchDiameter/2 < 1 ||
                    p2.y + patchDiameter/2 > img1.rows ||
                    p2.y - patchDiameter/2 < 1)
					continue;

				if (!isPtOnLineSegment(p2, lines2[j]))
					continue;

				cv::KeyPoint kp2(p2, patchDiameter, angle_j);
				kpts1.push_back(kp1);
				kpts2.push_back(kp2);
			}

			if (kpts1.size() == 0 )	{
				scores.at<double>(i,j) = 1e6;
				continue;
			}

			featExtractor.compute(gImg1, kpts1, desc1);
			featExtractor.compute(gImg2, kpts2, desc2);
			double sum = 0;

			for (int k=0; k<kpts1.size(); ++k){
				sum = sum + cv::norm(desc1.row(k)-desc2.row(k));
			}

			scores.at<double>(i,j) = sum/kpts1.size();
		}

	}
   // ============  Identify matches based on simlarity matrix ==========
	for (int i=0; i<scores.rows; ++i) {
		vector<int> onePairIdx;
		double minVal;
		cv::Point minPos;
		cv::minMaxLoc(scores.row(i),&minVal,NULL,&minPos,NULL);
		if (minVal < 0.5) {
			double minV;
			cv::Point minP;
			cv::minMaxLoc(scores.col(minPos.x),&minV,NULL,&minP,NULL);
			if (i==minP.y) {    // commnent this for more potential matches
				onePairIdx.push_back(lines1[i].lid);
				onePairIdx.push_back(lines2[minPos.x].lid);
            linePairIdx.push_back(onePairIdx);
			}
		}
	}
	return linePairIdx;
}


vector<cv::Point2d> sampleFromLine (IdealLine2d l, int ptNum)
{
	double stepSize = l.length()/(ptNum-1);
	vector<cv::Point2d> samples;

	cv::Point2d step = stepSize*
           (l.extremity2 - l.extremity1)*(1/cv::norm(l.extremity2 - l.extremity1));
	for (int i=0; i<ptNum; ++i) {
		samples.push_back(l.extremity1 + i*step);
	}

	return samples;
}
