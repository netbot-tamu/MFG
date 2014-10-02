#include "mfg.h"
#include "mfg_utils.h"

extern int IDEAL_IMAGE_WIDTH;
extern double SIFT_THRESH;
extern SysPara syspara;

//#define VPDETECT_USE_JLINK

View::View (string imgName, cv::Mat _K, cv::Mat dc)
	// only deal with points
{	
	filename = imgName;
	id = -2;
	cv::Mat oriImg = cv::imread(imgName,1); 
	cv::Mat tmpImg;
	tmpImg = oriImg.clone();
	if ( cv::norm(dc) > 1e-6) {
		cv::undistort(oriImg, tmpImg, _K, dc);
	}
	// resize image and ajust camera matrix
	if (syspara.use_img_width > 1) {
		double scl = syspara.use_img_width/double(tmpImg.cols);
		cv::resize(tmpImg,img,cv::Size(),scl,scl,cv::INTER_AREA);
		K = _K * scl;
		K.at<double>(2,2) = 1;	
	} else {
		img = tmpImg;
		K = _K;
	}

	IDEAL_IMAGE_WIDTH = img.cols;

//	cv::Mat grayImg;
	if (img.channels()==3)
		cv::cvtColor(img, grayImg, CV_RGB2GRAY);
	else
		grayImg = img;	

	detectFeatPoints ();	
}

View::View (string imgName, cv::Mat _K, cv::Mat dc, int _id)
{	
	filename = imgName;
	id = _id;
	cv::Mat oriImg = cv::imread(imgName,1); 
	cv::Mat tmpImg;
	tmpImg = oriImg.clone();

	if ( cv::norm(dc) > 1e-6) {
		cv::undistort(oriImg, tmpImg, _K, dc);
	}
	// resize image and ajust camera matrix
	if (syspara.use_img_width > 1) {
		double scl = syspara.use_img_width/double(tmpImg.cols);
		cv::resize(tmpImg,img,cv::Size(),scl,scl,cv::INTER_AREA);
		K = _K * scl;
		K.at<double>(2,2) = 1;	
	} else {
		img = tmpImg;
		K = _K;
	}

	IDEAL_IMAGE_WIDTH = img.cols;

//	cv::Mat grayImg;
	if (img.channels()==3)
		cv::cvtColor(img, grayImg, CV_RGB2GRAY);
	else
		grayImg = img;	
	lsLenThresh = img.cols/100.0;   // for raw line segments

	MyTimer timer;
	timer.start();
	detectFeatPoints ();	
	timer.end();
	cout<<"Keypoint detection time:" << timer.time_ms << " ms"<<endl;
	IplImage pImg = img;
	timer.start();
	detectLineSegments(pImg);	
	timer.end();
//	cout<<"Line segment detection time:" << timer.time_ms << " ms"<<endl;
	//	drawAllLineSegments();
	compMsld4AllSegments (grayImg);

	timer.start();
	detectVanishPoints();
#ifndef HIGH_SPEED_NO_GRAPHICS
	
#endif
	drawAllLineSegments(true);
	timer.end();
//	cout<<"Vanishing point detection time:" << timer.time_ms << " ms"<<endl;


	timer.start();
	extractIdealLines();
	timer.end();
//	cout<<"Ideal line extraction time:" << timer.time_ms << " ms"<<endl;
	//	drawPointandLine();

	//	drawIdealLines();

	errPt=0; errLn=0; errAll=0;
	errPtMean=0; errLnMean=0;
}

cv::Point2d LineSegmt2d::getGradient(cv::Mat* xGradient, cv::Mat* yGradient)
{	
	cv::LineIterator iter(*xGradient, endpt1, endpt2, 8);
	double xSum=0, ySum=0;
	for (int i=0; i<iter.count; ++i, ++iter) {
		xSum += xGradient->at<double>(iter.pos());
		ySum += yGradient->at<double>(iter.pos());
	}
	double len = sqrt(xSum*xSum+ySum*ySum);
	return cv::Point2d(xSum/len, ySum/len);
}

cv::Mat LineSegmt2d::lineEq ()
{
	cv::Mat pt1 = (cv::Mat_<double>(3,1)<<endpt1.x, endpt1.y, 1);
	cv::Mat pt2 = (cv::Mat_<double>(3,1)<<endpt2.x, endpt2.y, 1);
	cv::Mat lnEq = pt1.cross(pt2); // lnEq = pt1 x pt2		
	lnEq = lnEq/sqrt(lnEq.at<double>(0)*lnEq.at<double>(0)
		+lnEq.at<double>(1)*lnEq.at<double>(1)); // normalize, optional
	return lnEq;

}

void View::detectFeatPoints()
{
	vector<cv::KeyPoint>				poses;
	cv::Mat 							descs;
	vector<cv::Point2f>					ptpos;
	// Only work for opencv 2.3.1
	switch (syspara.kpt_detect_alg) {
	case 1: {
		// opencv 2.3
		//cv::SiftFeatureDetector		siftFeatDetector(SIFT_THRESH, 10);	// 0.05 surpress point num
		//cv::SiftDescriptorExtractor siftDeatExtractor; 
		//siftFeatDetector.detect(img,  poses);
		//siftDeatExtractor.compute(img,  poses,  descs);
		// for opencv2.4.. -- 
		cv::FeatureDetector * pfeatDetector = new cv::SIFT(0,3,0.01,10);
		pfeatDetector->detect(img, poses);
		cv::DescriptorExtractor * pfeatExtractor = new cv::SIFT();
		pfeatExtractor->compute(img, poses, descs);	
			}
			break;
	case 2: {
		cv::SurfFeatureDetector		surfFeatDetector(200);
		cv::SurfDescriptorExtractor  surfFeatExtractor;
		surfFeatDetector.detect(img,  poses);
		surfFeatExtractor.compute(img,  poses,  descs);
			}
			break;
	case 3: { 
		if (id==-2) break; // don't detect features when tracking between key frames
		detect_featpoints_buckets(grayImg, // the image 
			3,
			ptpos,   // the output detected features
			syspara.gftt_max_ptnum,  // the maximum number of features 
			syspara.gftt_qual_levl,     // quality level
			syspara.gftt_min_ptdist     // min distance between two features		
			);
			}
		break;
	default:	;
	}	

	switch (syspara.kpt_detect_alg) {
	case 1:
	case 2:
		for (int i=0; i< poses.size(); ++i) {
			featurePoints.push_back(
			FeatPoint2d(poses[i].pt.x, poses[i].pt.y, descs.row(i).t(), i, -1));
		}
		break;
	case 3: {
/*		vector<cv::KeyPoint> kpts;
		for(int i=0; i<ptpos.size(); ++i) {
			kpts.push_back(cv::KeyPoint(ptpos[i], 21));// no angle info provided
		}
		cv::ORB orb;
		orb(grayImg, cv::Mat(), kpts, descs, true);
*/		for(int i=0; i<ptpos.size(); ++i) {
			featurePoints.push_back(
				FeatPoint2d(ptpos[i].x,ptpos[i].y, i));
		//	featurePoints.back().lid = i;
		//	featurePoints.back().siftDesc = descs.row(i).clone().t();
		}
			}
			break;
	}
	

}

void View::detectLineSegments(IplImage pImg)
{
	ntuple_list  lsdOut;
	lsdOut = callLsd(&pImg, 0);// use LSD method
	int dim = lsdOut->dim;
	double a,b,c,d;
	for(int i=0; i<lsdOut->size; i++) {// store LSD output to lineSegments 
		a = lsdOut->values[i*dim];
		b = lsdOut->values[i*dim+1];
		c = lsdOut->values[i*dim+2];
		d = lsdOut->values[i*dim+3];
		if ( sqrt((a-c)*(a-c)+(b-d)*(b-d)) > lsLenThresh) {
			lineSegments.push_back(LineSegmt2d(cv::Point2d(a,b), 
				cv::Point2d(c,d)));
		}
	}
	for (int i=0; i<lineSegments.size(); ++i)
		lineSegments[i].lid = i;
}

void View::compMsld4AllSegments (cv::Mat grayImg) {
	cv::Mat xGradImg, yGradImg;
	int ddepth = CV_64F;	
	cv::Sobel(grayImg, xGradImg, ddepth, 1, 0, 5); // Gradient X
	cv::Sobel(grayImg, yGradImg, ddepth, 0, 1, 5); // Gradient Y
	for (int i=0; i<lineSegments.size(); ++i)  {
		computeMSLD(lineSegments[i], &xGradImg, &yGradImg);
	}
}

void View::detectVanishPoints ()
	// input: line segments
	// output: vanishing points including children segments, vp labels of segments 
	// can use vanishing points orthogonality based on camera matrix.
{
	// ----- parameters setting -----
	double vertAngThresh = 15.0 * PI/180; //angle threshold for vertical lines	
	vector<LineSegmt2d>&ls = lineSegments;

	double orthThresh =			sin(3*PI/180);  // threshold for VP orthogonality 
	double vp2LineDistThresh =	tan(1.5*PI/180);  // normalized dist 
	double line2VpDistThresh =	1;    // mledist threshold

	vector<int> oriIdx;  // original index of ls
	for (int i=0; i<ls.size(); ++i) oriIdx.push_back(i);

	// ======== 1. group vertical line segments using angle threshold =======
	double tanThresh = tan(vertAngThresh);
	vector<int> vertGroupIdx;
	for (int i = 0; i < ls.size(); ++i)	{
		if (ls[i].endpt1.y-ls[i].endpt2.y != 0 && 
			abs((ls[i].endpt1.x-ls[i].endpt2.x)/
			(ls[i].endpt1.y-ls[i].endpt2.y)) <tanThresh) {
				vertGroupIdx.push_back(i);	
		}		
	}

	int sz = vertGroupIdx.size()+1;
	cv::Mat vp = cv::Mat::zeros(3, 1, CV_64F);
	cv::Mat cov, covhomo;
	while (vertGroupIdx.size()<sz) {				
		sz = vertGroupIdx.size();
		refineVanishPt (lineSegments, vertGroupIdx, vp, cov, covhomo);	
	}	
	//	drawLineSegmentGroup(vertGroupIdx);
	vanishPoints.push_back(
		VanishPnt2d(vp.at<double>(0),vp.at<double>(1),vp.at<double>(2),0, -1));
	vanishPoints.back().cov = cov.clone();
	vanishPoints.back().cov_homo = covhomo.clone();
	for (int i=0; i<vertGroupIdx.size(); ++i) // assign vp lid to line segments
		lineSegments[vertGroupIdx[i]].vpLid = 0;

	
	cv::Mat v0 = K.inv()*(cv::Mat_<double>(3,1)<<vanishPoints[0].x,
		vanishPoints[0].y, vanishPoints[0].w);
	// ==== 2.  horizontal VPs search =====
	int	maxHvpNo  = 2;	// number of horizontal VPs
	int minLsNoTh = 30; // minimum number of supporting ls for a hrz vp
#ifdef VPDETECT_USE_JLINK
	vector<unsigned int> labNo;
	double lenThresh = img.cols/50.0;
	vector<cv::Mat> hvpCov;
	vector<cv::Mat> hvps =  detectVP_Jlink(ls, labNo, lenThresh, hvpCov);
	for(int i=0; i < ls.size(); ++i) {
		if(ls[i].vpLid >= maxHvpNo+1) 
			ls[i].vpLid = -1;
	}
	for(int i=0; i < hvps.size(), i < maxHvpNo; ++i) {
		if(labNo[i] < minLsNoTh ||      // if supporting lines number too small
			labNo[i] < labNo[0] * 0.7 || // or, smaller than the largest * ratio 
			abs(v0.dot(K.inv()*hvps[i]))/cv::norm(v0)/cv::norm(K.inv()*hvps[i]) > orthThresh) {
				for(int j=0; j < ls.size(); ++j) {
					if(ls[j].vpLid == (i+1))
						ls[j].vpLid = -1;
				}
		} else {				
			for(int j=0; j < ls.size(); ++j) {					
				if(ls[j].vpLid == int(i+1))
					ls[j].vpLid = vanishPoints.size();
			}
			int vplid = vanishPoints.size();
			vanishPoints.push_back(VanishPnt2d(hvps[i].at<double>(0),
				hvps[i].at<double>(1),hvps[i].at<double>(2), vplid, -1));
			vanishPoints.back().cov = hvpCov[i].clone();
		}
	}

#else

	double vpAngleLB =	80;  //degree
	double confidence = 0.99;  // for recomputing maxIterNo.

	for (int vpNo = 1; vpNo < maxHvpNo+1; ++vpNo) {
		int	inlierNoThresh	= ls.size()/maxHvpNo; 
		vector<int> maxInlierIdx;    
		vector<int>	inlierIdx;
		int	maxIterNo	= 500;   // initial max RANSAC iteration number
		for (int i=0; i < maxIterNo; ++i) {
			int j = rand() % ls.size();
			int k = rand() % ls.size();
			if (j==k || ls[j].vpLid!=-1 || ls[k].vpLid!=-1) { //redo it
				--i;
				continue;
			}					
			cv::Mat vpSeed = ls[j].lineEq().cross(ls[k].lineEq());
			cv::Mat tmpVp = K.inv()*vpSeed;
			double dotPrdt = v0.dot(tmpVp)/(cv::norm(v0)*cv::norm(tmpVp));
			if (abs(dotPrdt) > orthThresh) { // check orthogonality of VPs
				// only keep potential VPs being horizontal 
				//	--i;
				continue;
			}
			// if too similar to existing vps, skip
			bool overlap = false;			
			for (int vi=0; vi < vanishPoints.size(); ++vi) {
				if(abs(tmpVp.dot(K.inv()*vanishPoints[vi].mat())/(cv::norm(K.inv()*vanishPoints[vi].mat())
					*cv::norm(tmpVp))) > cos(vpAngleLB*PI/180) )
					overlap = true;	
			}
			if (overlap) {
				//	--i;
				continue;
			}
			for (int ii=0; ii<ls.size(); ++ii) {	
				if (ls[ii].vpLid != -1) continue;
				double dis = mleVp2LineDist(vpSeed, ls[ii]);
				//				if (dis < line2VpDistThresh )
				if (dis/ls[ii].length() < vp2LineDistThresh)
					inlierIdx.push_back(ii);
			}	
			if (inlierIdx.size() > maxInlierIdx.size()){
				maxInlierIdx.clear();
				maxInlierIdx = inlierIdx;				
			}
			inlierIdx.clear();
			//			maxIterNo = abs(log(1-confidence) // re-compute maxIterNo.
			//				/log(1-pow(double(maxInlierIdx.size())/ls.size(),2.0)));
		}

		if (maxInlierIdx.size() < 2) continue;
		//------ Next is guided matching, to find more inliers--------
		cv::Mat newCrsPt;
		vector<LineSegmt2d> lines;
		while (true) {
			lines.clear();
			cv::Mat A(3,maxInlierIdx.size(), CV_64F); // A'*x = 0;
			inlierIdx.clear();			
			if (maxInlierIdx.size()==2)
				newCrsPt = ls[maxInlierIdx[0]].lineEq().cross
				(ls[maxInlierIdx[1]].lineEq());
			else {
				for (int i = 0; i < maxInlierIdx.size(); ++i) {					
					ls[maxInlierIdx[i]].lineEq().copyTo(A.col(i));
					lines.push_back(ls[maxInlierIdx[i]]);
				}
				cv::SVD::solveZ(A.t(),newCrsPt); // linear solution			
				optimizeVainisingPoint(lines, newCrsPt);			
			}

			for (int i=0; i<ls.size(); ++i) {
				if (ls[i].vpLid != -1) continue;
				double dist = mleVp2LineDist(newCrsPt, ls[i]);
				//				if (dist < line2VpDistThresh )
				if(dist/ls[i].length() < vp2LineDistThresh)
					inlierIdx.push_back(i);
			}
			if (inlierIdx.size() > maxInlierIdx.size())
				maxInlierIdx = inlierIdx;
			else
				break;
		}
		if (maxInlierIdx.size() < 5) continue;
		int sz = maxInlierIdx.size()+1;
		cv::Mat vp = cv::Mat::zeros(3, 1, CV_64F);
		cv::Mat hvpCov, hvpCovHomo;
		while (maxInlierIdx.size() < sz) {				
			sz = maxInlierIdx.size();
			refineVanishPt (lineSegments, maxInlierIdx, vp, hvpCov, hvpCovHomo);	
			if (maxInlierIdx.size() < 5) break;
		}

		//	drawLineSegmentGroup(maxInlierIdx);
		if (maxInlierIdx.size() < 15) continue; // inliers too few
		
		bool overlap = false;
		for (int vi=0; vi < vanishPoints.size(); ++vi) {
			if(abs((K.inv()*vp).dot(K.inv()*vanishPoints[vi].mat())/(cv::norm(K.inv()*vanishPoints[vi].mat())
				*cv::norm((K.inv()*vp)))) > cos(vpAngleLB*PI/180) )
				overlap = true;	
		}
		if (overlap) {
			//	--i;
			continue;
		}
		
		int vplid = vanishPoints.size();
		for (int i=0; i<maxInlierIdx.size(); ++i)
			ls[maxInlierIdx[i]].vpLid = vplid;		
		vanishPoints.push_back(VanishPnt2d(vp.at<double>(0), vp.at<double>(1),vp.at<double>(2), vplid , -1));
		vanishPoints.back().cov = hvpCov.clone();
		vanishPoints.back().cov_homo = hvpCovHomo.clone();
		
	}
#endif
	vpGrpIdLnIdx.resize(vanishPoints.size());

	for (int i=0; i<vanishPoints.size(); ++i) {
		//		cout<<vanishPoints[i].x<<","<<vanishPoints[i].y<<","<<vanishPoints[i].w<<endl;
		//		cout<<vanishPoints[i].cov<<endl;
		//		cout<<vanishPoints[i].cov_homo<<endl;

		//		vanishPoints[i].cov_ab = vanishpoint_cov_xy2ab(vanishPoints[i].mat(0),K,vanishPoints[i].cov);
		//		cout<<"cov_ab="<<vanishPoints[i].cov_ab<<endl;
		vanishPoints[i].cov_ab = vanishpoint_cov_xyw2ab(vanishPoints[i].mat(),K,vanishPoints[i].cov_homo);
		//		cout<<"cov_ab="<<vanishPoints[i].cov_ab<<endl;
	}

}

IdealLine2d combineIdeallines (IdealLine2d l1, IdealLine2d l2)
	// l1 l2 should have the same vplid
{
	IdealLine2d l = l1;
	// choose two furthest endpoints to be new endpoints
	double tmp = cv::norm(l1.extremity1-l1.extremity2);
	if (cv::norm(l1.extremity1-l2.extremity1) > tmp)	{
		l.extremity1 = l1.extremity1;
		l.extremity2 = l2.extremity1;
		tmp = cv::norm(l1.extremity1-l2.extremity1);
	}
	if (cv::norm(l1.extremity1-l2.extremity2) > tmp)	{
		l.extremity1 = l1.extremity1;
		l.extremity2 = l2.extremity2;
		tmp = cv::norm(l1.extremity1-l2.extremity2);
	} 
	if (cv::norm(l1.extremity2-l2.extremity1) > tmp)	{
		l.extremity1 = l1.extremity2;
		l.extremity2 = l2.extremity1;
		tmp = cv::norm(l1.extremity2-l2.extremity1);
	}
	if (cv::norm(l1.extremity2-l2.extremity2) > tmp)	{
		l.extremity1 = l1.extremity2;
		l.extremity2 = l2.extremity2;
		tmp = cv::norm(l1.extremity2-l2.extremity2);
	}
	if (cv::norm(l2.extremity1-l2.extremity2) > tmp)	{
		l.extremity1 = l2.extremity1;
		l.extremity2 = l2.extremity2;
	}

	l.lsLids.clear();
	l.lsLids.reserve(l1.lsLids.size()+l2.lsLids.size());
	l.lsLids.insert(l.lsLids.end(),l1.lsLids.begin(),l1.lsLids.end());
	l.lsLids.insert(l.lsLids.end(),l2.lsLids.begin(),l2.lsLids.end());

	l.msldDescs.clear();
	l.msldDescs.reserve(l1.msldDescs.size()+l2.msldDescs.size());
	l.msldDescs.insert(l.msldDescs.end(),l1.msldDescs.begin(),l1.msldDescs.end());
	l.msldDescs.insert(l.msldDescs.end(),l2.msldDescs.begin(),l2.msldDescs.end());

	l.gradient = (l1.gradient + l2.gradient)* 0.5;
	l.gradient = l.gradient * (1/cv::norm(l.gradient));

	return l;
}

double collinearCheckUseVP(cv::Mat vp, vector<LineSegmt2d>& segments, 
	IdealLine2d a, IdealLine2d b)
	// given vanishing point, check if a and b are collinear or not
	// when vp is zeros, it imposes no constraint
	// return pt to line distance
{
	// 1. compute a line passing vp and all endpoints of a
	cv::Mat A(3,a.lsLids.size()*2+1, CV_64F); // A'*l = 0;
	for (int i = 0; i < a.lsLids.size(); ++i) {
		cv::Mat pt = (cv::Mat_<double>(3,1)<< segments[a.lsLids[i]].endpt1.x,
			segments[a.lsLids[i]].endpt1.y,1);
		pt.copyTo(A.col(i));
		pt = (cv::Mat_<double>(3,1)<< segments[a.lsLids[i]].endpt2.x,
			segments[a.lsLids[i]].endpt2.y,1);
		pt.copyTo(A.col(a.lsLids.size()+i));
	}
	cv::Mat tmp = vp*100;
	tmp.copyTo(A.col(A.cols-1));
	cv::Mat l;
	cv::SVD::solveZ(A.t(),l); // line equation for a and vp

	// 2.  compute avarage pt to l distance from b
	double sum = 0;
	for (int i=0; i<b.lsLids.size(); ++i) {
		sum = sum + point2LineDist(l, segments[b.lsLids[i]].endpt1)
			+ point2LineDist(l, segments[b.lsLids[i]].endpt2);
	}
	return sum/(2*b.lsLids.size());
}

void View::extractIdealLines()
	// input: raw line segment, with vp label
	// output: ideal lines with line segment as children
{
	double lsIntvThresh = img.cols/8.0; // interval between line segment endpoints
	double threshLineLen = img.cols/20.0;


	vector <vector<IdealLine2d>> ilines(vanishPoints.size()+1);
	for (int i=0; i<lineSegments.size(); ++i) { // generate group of ideal lines
		ilines[lineSegments[i].vpLid+1].push_back(IdealLine2d(lineSegments[i]));
	}
	for (int i=0; i<ilines.size(); ++i)	{
		cv::Mat vp = cv::Mat::zeros(3,1,CV_64F);
		if (i>0)
			vp = vanishPoints[i-1].mat();
		for (int j=0; j < ilines[i].size(); ++j) {
			cv::Point2d vj = (ilines[i][j].extremity1-ilines[i][j].extremity2)
				*(1/cv::norm(ilines[i][j].extremity1-ilines[i][j].extremity2));
			for (int k=0; k < ilines[i].size(); ++k) {
				// find potential neighbor
				if (j==k) continue;			
				cv::Point2d vk = (ilines[i][k].extremity1-ilines[i][k].extremity2)
					*(1/cv::norm(ilines[i][k].extremity1-ilines[i][k].extremity2));
				if (abs(vj.dot(vk)) < cos(PI/10)) // check line angles
					continue;
				if (ilines[i][j].gradient.dot(ilines[i][k].gradient) < 0)
					continue; // gradient not consistent, skip
				if (point2LineDist(ilines[i][j].lineEq(), ilines[i][k].extremity1)>20)
					continue;

				if (collinearCheckUseVP(vp, lineSegments,ilines[i][j], ilines[i][k]) > 1 &&
					collinearCheckUseVP(vp, lineSegments,ilines[i][k], ilines[i][j]) > 1 )
					continue;
				if (getLineEndPtInterval(ilines[i][j], ilines[i][k]) > lsIntvThresh )
					continue;

				//		if (compMsldDiff(ilines[i][j],ilines[i][k]) > 0.8 )
				//			continue; // msld not similar

				// combine two lines
				ilines[i][j] = (combineIdeallines(ilines[i][j], ilines[i][k]));
				ilines[i].erase(ilines[i].begin()+k);
				k=0;
			}	
			// --- negelect short ideal lines ---(optional)
			if(ilines[i][j].length() < threshLineLen) {
				//	cout<<ilines[i][j].length<<endl;
				ilines[i].erase(ilines[i].begin()+j);
				j = -1;
				continue;
			}

			idealLines.push_back(ilines[i][j]);
			idealLines.back().lid = idealLines.size()-1;
			// add line lid to corresponding vanishing point group
			if ( idealLines.back().vpLid >=0 )
				vanishPoints[idealLines.back().vpLid].idlnLids.push_back(idealLines.back().lid);
			// put line lid into corresponding groups according to vp
			if (idealLines.back().vpLid >=0)
				vpGrpIdLnIdx[idealLines.back().vpLid].push_back(idealLines.back().lid);
			ilines[i].erase(ilines[i].begin()+j);
			j = -1;
		}
	}

	// ----- collect line segments' endpoints -----
	for(int i=0; i < idealLines.size(); ++i) {
		for(int j=0; j < idealLines[i].lsLids.size(); ++j) {
			idealLines[i].lsEndpoints.push_back(lineSegments[idealLines[i].lsLids[j]].endpt1);
			idealLines[i].lsEndpoints.push_back(lineSegments[idealLines[i].lsLids[j]].endpt2);
		}
	}

	// ----- recompute ideal line equation (extremities) -----
	// enforce ideal line to pass vanishing point
	// NOTE: need to improve by an optimzation method !!!!!!!
	for(int i=0; i < idealLines.size(); ++i) {
		if (idealLines[i].vpLid < 0) continue;
		cv::Mat vp = vanishPoints[idealLines[i].vpLid].mat();
		cv::Mat mid = cvpt2mat((idealLines[i].extremity1 + idealLines[i].extremity2)*0.5);
		cv::Mat lneq = vp.cross(mid);
		//	idealLines[i].extremity1 = mat2cvpt(findNearestPointOnLine (lneq, cvpt2mat(idealLines[i].extremity1)));
		//	idealLines[i].extremity2 = mat2cvpt(findNearestPointOnLine (lneq, cvpt2mat(idealLines[i].extremity2)));
	}
}

void View::drawLineSegmentGroup(vector<int> idx) // draw grouped line segments
{
	cv::Mat canvas = img.clone();
	cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
	for (int i=0; i<idx.size(); ++i){		
		cv::line(canvas,lineSegments[idx[i]].endpt1,
			lineSegments[idx[i]].endpt2, color,2);		
	}	
	showImage("Grouped line segments"+rand(),&canvas);
	cv::waitKey();
}

void View::drawAllLineSegments(bool write2file)
{
	vector<cv::Scalar> colors;
	colors.push_back(cv::Scalar(rand()%255,rand()%255,rand()%255,0));
	colors.push_back(cv::Scalar(0,0,255,0));
	colors.push_back(cv::Scalar(0,255,0,0));
	colors.push_back(cv::Scalar(255,0,0,0));
	for (int i=0; i < vanishPoints.size()+1; ++i)	{
		colors.push_back(cv::Scalar(rand()%255,rand()%255,rand()%255,0));
	}

#ifndef HIGH_SPEED_NO_GRAPHICS
	cv::Mat canvas = img.clone();
	for (int i=0; i<lineSegments.size(); ++i) {
		if(lineSegments[i].vpLid==-1)
			cv::line(canvas, lineSegments[i].endpt1,
			lineSegments[i].endpt2, colors[lineSegments[i].vpLid+1],1);
		else
			cv::line(canvas, lineSegments[i].endpt1,
			lineSegments[i].endpt2, colors[lineSegments[i].vpLid+1],2);
	}
	if(!write2file) {
		showImage("view"+num2str(id)+":line segments "+num2str(lineSegments.size()),&canvas);
		cv::waitKey(0);
	}else{
		// ---- write to images for debugging ----
		cv::imwrite("./tmpimg/vp_"+num2str(id)+".jpg", canvas);
	}
#endif
}

void View::drawIdealLineGroup(vector<IdealLine2d> ilines)
{
	cv::Mat canvas = img.clone();
	cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
	for (int i=0; i<ilines.size(); ++i) {		
		//		for (int j=0; j<ilines[i].lsLids.size(); ++j) 		
		//			cv::line(canvas, lineSegments[ilines[i].lsLids[j]].endpt1,
		//			lineSegments[ilines[i].lsLids[j]].endpt2, 
		//			cv::Scalar(255,255,255,0)-color, 2);
		cv::line(canvas,ilines[i].extremity1,ilines[i].extremity2, color,1);
		cv::circle(canvas,ilines[i].extremity1,1,color,1);
		cv::circle(canvas,ilines[i].extremity2,1,color,1);
	}
	showImage("Ideal line & segments",&canvas);
	cv::waitKey(0);
}

void View::drawIdealLines()
{
	cv::Mat canvas = img.clone();
	vector<cv::Scalar> colors;

	//	colors.push_back(cv::Scalar(rand()%255,rand()%255,rand()%255,0));
	colors.push_back(cv::Scalar(0,255,255,0));
	colors.push_back(cv::Scalar(0,0,255,0));
	colors.push_back(cv::Scalar(0,255,0,0));
	colors.push_back(cv::Scalar(255,0,0,0));
	for (int i=0; i < vanishPoints.size()+1; ++i)	{
		colors.push_back(cv::Scalar(rand()%255,rand()%255,rand()%255,0));
	}
	for (int i=0; i<idealLines.size(); ++i) {
		cv::line(canvas, idealLines[i].extremity1,
			idealLines[i].extremity2, colors[idealLines[i].vpLid+1],1);
	}
	showImage("view"+num2str(id)+":ideal lines",&canvas);
	cv::waitKey(0);
}

void View::drawPointandLine()
{
	cv::Mat canvas = img.clone();

	for (int i=0; i<featurePoints.size(); ++i)
	{
		cv::circle(canvas, cv::Point2d(featurePoints[i].x, featurePoints[i].y), 2,
			cv::Scalar(rand()%255,rand()%255,rand()%255,0), 1);
	}

	vector<cv::Scalar> colors;

	colors.push_back(cv::Scalar(rand()%255,rand()%255,rand()%255,0));
	//colors.push_back(cv::Scalar(0,0,255,0));
	colors.push_back(cv::Scalar(0,255,0,0));
	colors.push_back(cv::Scalar(0,255,0,0));
	colors.push_back(cv::Scalar(0,255,0,0));
	//colors.push_back(cv::Scalar(255,0,0,0));
	for (int i=0; i < vanishPoints.size()+1; ++i)	{
		colors.push_back(cv::Scalar(rand()%255,rand()%255,rand()%255,0));
	}
	for (int i=0; i<lineSegments.size(); ++i) {
		cv::line(canvas, lineSegments[i].endpt1,
			lineSegments[i].endpt2, colors[lineSegments[i].vpLid+1],1);
	}
	showImage("view"+num2str(id)+":line segments",&canvas);
	cv::waitKey(0);
}

