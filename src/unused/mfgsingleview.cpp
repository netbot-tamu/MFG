#include "mfg.h"
#include "mfg_utils.h"
//#include "opencv2/nonfree/nonfree.hpp" // For opencv2.4+, need link to opencv_nonfree242.lib

extern int IDEAL_IMAGE_WIDTH;

/////////////////////////////// MfgSingleView /////////////////////////////////

MfgSingleView::MfgSingleView (string imgName, cv::Mat K, cv::Mat dc)
{	
	cv::Mat oriImg = cv::imread(imgName,1); 
	cv::Mat tmpImg;
	tmpImg = oriImg;
//	cv::undistort(oriImg, tmpImg, K, dc);
	// resize image and ajust camera matrix
	double scl = IDEAL_IMAGE_WIDTH/double(tmpImg.cols);
	cv::resize(tmpImg,mImg,cv::Size(),scl,scl); 
	camMat = K*scl;
	camMat.at<double>(2,2) = 1;	
	pImg = mImg;
	
	if (mImg.channels()==3)
		cv::cvtColor(mImg, mImgGray, CV_RGB2GRAY);
	else
		mImgGray = mImg;	
	lsLenThresh = mImg.cols/200.0;   // for raw line segments
	MyTimer timer;
	timer.start();	detectLineSegments();		timer.end();
	cout<<"detectLineSegments: "<<timer.time_ms<<" ms."<<endl;
//	drawAllLineSegments();
	cout<< "Total line segments number = " <<lineSegments.size()<<endl;
	computeLineSegMSLD ();

	timer.start();	oneStepLineGrouping();  timer.end();
	cout<<"oneStepLineGrouping: "<<timer.time_ms<<" ms."<<endl;
	getIdealLineMSLDs ();
//	drawIdealLineGroups();
	cv::Mat v0 = camMat.inv()*vanishPoints[0].matHomo(),
		    v1 = camMat.inv()*vanishPoints[1].matHomo(),
			v2 = camMat.inv()*vanishPoints[2].matHomo();

	timer.start();	detectKeyPoints ();		timer.end();
	cout<<"detectKeyPoints: "<<timer.time_ms<<" ms."
		<<"Keypoint Num = "<<keyPointPoses.size()<<endl;
	//cv::destroyAllWindows();
}

void  MfgSingleView::detectKeyPoints()
{
	// Only work for opencv 2.3.0
	cv::SiftFeatureDetector		featDetector(0.01, 10);
	cv::SiftDescriptorExtractor featExtractor;
//	cv::SurfFeatureDetector		featDetector;
//	cv::SurfDescriptorExtractor  featExtractor;

	featDetector.detect   (mImg,  keyPointPoses);
	featExtractor.compute (mImg,  keyPointPoses,  keyPointFeatures);
	// NOTE: featExtractor.compute causes MEMEORY LEAK!!!


	

/*	// for opencv2.4.2 -- 
	cv::FeatureDetector * featDetector = new cv::SIFT(0,3,0.01,10);
	featDetector->detect(mImg, keyPointPoses);
	cv::DescriptorExtractor * featExtractor = new cv::SIFT();
	featExtractor->compute(mImg, keyPointPoses, keyPointFeatures);
*/
		
}

cv::Mat MfgSingleView::vertPersCorrect()
// this function return a homography that transforms the current vertical
// vanishing point to (0, vy, 0).
{
	cv::Mat invK = camMat.inv();
	double theta_x =atan(vanishPoints[0].z_/(invK.at<double>(1,1)
		*vanishPoints[0].y_+invK.at<double>(1,2)*vanishPoints[0].z_));
	cv::Mat Rx = (cv::Mat_<double>(3,3)<<1,0,0,0,cos(-theta_x), sin(theta_x),
		0,sin(-theta_x),cos(-theta_x));
	cv::Mat Hx = camMat*Rx*invK;
	cv::Mat v0p = Hx*vanishPoints[0].matHomo();
	//cout<<v0p<<endl;
	double theta_z = atan(-invK.at<double>(0,0)*v0p.at<double>(0)/
		(invK.at<double>(1,1)*v0p.at<double>(1)));
	cv::Mat Rz = (cv::Mat_<double>(3,3)<<cos(-theta_z),sin(theta_z),0,
		sin(-theta_z),cos(-theta_z),0,0,0,1);
	cv::Mat Hz = camMat*Rz*invK;
	return Hz*Hx;
}

void MfgSingleView::oneStepLineGrouping ()
	// using vanishing points orthogonality based on camera matrix.
{
	idealLineGroups.resize(4);
	lineMemberSegmentIdx.resize (4);
	lineSegmentGroups.resize (4);
	vanishPoints.resize(4);

	double vertAngThresh = 15.0 * PI/180; //angle threshold for vertical lines
	double idealLineLenThresh = IDEAL_IMAGE_WIDTH / 150.0;
	vector<MfgLineSegment>	ls = lineSegments;
	vector<int> oriIdx;  // original index of ls
	for (int i=0; i<ls.size(); ++i) oriIdx.push_back(i);

	// ======== 1. group vertical line segments using angle threshold =======
	double tanThresh = tan(vertAngThresh);
	vector<int> vertGroupIdx;
	for (int i = 0; i < ls.size(); ++i)	{
		if (ls[i].ptA.y_ih()-ls[i].ptB.y_ih() != 0 && 
			abs((ls[i].ptA.x_ih()-ls[i].ptB.x_ih())/
			(ls[i].ptA.y_ih()-ls[i].ptB.y_ih())) <tanThresh) {
				vertGroupIdx.push_back(i);
				vector<int> memIdx;
				memIdx.push_back(i);
				idealLineGroups[0].push_back(ls[i]);
				lineMemberSegmentIdx[0].push_back(memIdx);
		}		
	}
	for(int i = vertGroupIdx.size(); i > 0; --i) { // remove from ls 
		ls.erase(ls.begin()+vertGroupIdx[i-1]);			
		oriIdx.erase(oriIdx.begin()+vertGroupIdx[i-1]);
	}
	lineSegmentGroups[0] = vertGroupIdx; 
	int sz = idealLineGroups[0].size()+1;
	cv::Mat vp;
	while (idealLineGroups[0].size()<sz) {				
		sz = idealLineGroups[0].size();
		combineLineSegments(idealLineGroups[0], 
			lineMemberSegmentIdx[0], mImg.cols);
		filterConcurrentLines (idealLineGroups[0],idealLineGroups.back(),
			lineMemberSegmentIdx[0], lineMemberSegmentIdx.back(),	
			vp, lineSegments, mImg.cols);	
		vanishPoints[0] = Mfg2dPoint(vp);
		recomputeLinesByVp (idealLineGroups[0], vp);		
	}	

	// ==== 2. use VP orthogonality to narrow horizontal VPs search =====
	cv::Mat v0 = camMat.inv()*vanishPoints[0].matHomo();
	double orthThresh = 0.05;  // threshold for VP orthogonality 
	double vp2LineDistThresh = tan(1.5*PI/180);  // normalized dist 
	int	maxIterNo	  = 500;   // initial max RANSAC iteration number
	double confidence = 0.99;  // for recomputing maxIterNo.
	int	maxVpNo  = 2;	// number of horizontal VPs
	for (int vpNo = 1; vpNo < maxVpNo+1; ++vpNo) {
		int	inlierNoThresh	= ls.size()/maxVpNo; 
		vector<int>    maxInlierIdx;    
		vector<int>	inlierIdx;
		for (int i=0; i < maxIterNo; ++i) {
			int j = rand() % ls.size();
			int k = rand() % ls.size();
			if (j==k) { // if j=k, then redo it
				--i;
				continue;
			}		
			if (abs(ls[j].matLnEq(cv::Range(0,2),cv::Range::all()).dot(
				ls[k].matLnEq(cv::Range(0,2),cv::Range::all()))) < 0.5) {
					// when two lines have too large angle (60), ignore them
					--i;
					continue;
			}
			cv::Mat crsPt = ls[j].matLnEq.cross(ls[k].matLnEq);
			double dotPrdt = v0.dot(camMat.inv()*crsPt)
				/(cv::norm(v0)*cv::norm(camMat.inv()*crsPt));
			if (abs(dotPrdt) > orthThresh) { // check orthogonality of VPs
				// only keep potential VPs being horizontal 
				--i;
				continue;
			}				
			crsPt = crsPt/crsPt.at<double>(2); //inhomogeneous ( last ?0!)
			for (int ii=0; ii<ls.size(); ++ii) {			
				double dis = mleVp2LineDist(crsPt, ls[ii]);
				if (dis/ls[ii].length() < vp2LineDistThresh)
					inlierIdx.push_back(ii);
			}	
			if (inlierIdx.size() > maxInlierIdx.size()){
				maxInlierIdx.clear();
				maxInlierIdx = inlierIdx;
				//if (maxInlierIdx.size() > inlierNoThresh) 
				//	break;
			}
			inlierIdx.clear();
			maxIterNo = abs(log(1-confidence) // re-compute maxIterNo.
				/log(1-pow(double(maxInlierIdx.size())/ls.size(),2.0)));
		}
		//------ Next is guided matching, to find more inliers--------
		cv::Mat newCrsPt;
		vector<MfgLineSegment> lines;
		while (true) {
			cv::Mat A(3,maxInlierIdx.size(), CV_64F); // A'*x = 0;
			inlierIdx.clear();			
			if (maxInlierIdx.size()==2)
				newCrsPt = ls[maxInlierIdx[0]].matLnEq.cross
				(ls[maxInlierIdx[1]].matLnEq);
			else {
				for (int i = 0; i < maxInlierIdx.size(); ++i) {					
					ls[maxInlierIdx[i]].matLnEq.copyTo(A.col(i));
					lines.push_back(ls[maxInlierIdx[i]]);
				}
				cv::SVD::solveZ(A.t(),newCrsPt); //	
				Mfg2dPoint mfgPt(newCrsPt); // 
				optimizeVainisingPoint(lines, mfgPt);
				newCrsPt = mfgPt.matHomo();
			}
			newCrsPt = newCrsPt/newCrsPt.at<double>(2); // ERROR-PRONE !!!
			for (int i=0; i<ls.size(); ++i) {
				double dis = mleVp2LineDist(newCrsPt, ls[i]);
				if(dis/ls[i].length() < vp2LineDistThresh)
					inlierIdx.push_back(i);
			}
			if (inlierIdx.size() > maxInlierIdx.size())
				maxInlierIdx = inlierIdx;
			else
				break;
		}
		vector<int> maxInlierOriIdx;
		while(!maxInlierIdx.empty()) { // remove found inliers from ls set 
			maxInlierOriIdx.push_back(oriIdx[maxInlierIdx.back()]);
			vector<int> memIdx;
			memIdx.push_back(oriIdx[maxInlierIdx.back()]);
			idealLineGroups[vpNo].push_back(ls[maxInlierIdx.back()]);
			lineMemberSegmentIdx[vpNo].push_back(memIdx);
			ls.erase(ls.begin()+maxInlierIdx.back()); // ls's size is changing			
			oriIdx.erase(oriIdx.begin()+maxInlierIdx.back());
			maxInlierIdx.pop_back();
		}
		lineSegmentGroups[vpNo] = maxInlierOriIdx;
		vanishPoints[vpNo] = Mfg2dPoint(newCrsPt);

		int sz = idealLineGroups[vpNo].size()+1;
		cv::Mat vp;
		while (idealLineGroups[vpNo].size() < sz) {				
			sz = idealLineGroups[vpNo].size();
			combineLineSegments(idealLineGroups[vpNo], 
				lineMemberSegmentIdx[vpNo], mImg.cols);
			filterConcurrentLines (idealLineGroups[vpNo],
				idealLineGroups.back(),	lineMemberSegmentIdx[vpNo], 
				lineMemberSegmentIdx.back(),vp,lineSegments, mImg.cols);
			vanishPoints[vpNo] = Mfg2dPoint(vp);
			recomputeLinesByVp (idealLineGroups[vpNo], vp);
		}
	}
	//============= 3. the rest segments form a group  =============
	lineSegmentGroups.back() = oriIdx; 
	for (int i=0; i<oriIdx.size(); ++i)	{
		idealLineGroups.back().push_back(ls[i]);
		vector<int> memIdx;
		memIdx.push_back(oriIdx[i]);
		lineMemberSegmentIdx.back().push_back(memIdx);
	}
	combineLineSegments(idealLineGroups.back(),
		lineMemberSegmentIdx.back(), mImg.cols);

	// **** remove too-short ideal lines ****
	for (int i=0; i<idealLineGroups.size(); ++i) {
		for (int j=idealLineGroups[i].size()-1; j>=0; --j) {
			if (idealLineGroups[i][j].length() < idealLineLenThresh) {
				idealLineGroups[i].erase(idealLineGroups[i].begin()+j);
				lineMemberSegmentIdx[i].erase(lineMemberSegmentIdx[i].begin()+j);
			}

		}
	}

}

void MfgSingleView::detectLineSegments()
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
			lineSegments.push_back(MfgLineSegment(a, b, c,d)) ;
			lineSegments.back().getEndPtCov(); // comptute endpoints covaraince
		}
	}
}

void MfgSingleView::computeLineSegMSLD () {
	cv::Mat xGradImg, yGradImg;
	int ddepth = CV_64F;	
	cv::Sobel(mImgGray, xGradImg, ddepth, 1, 0, 5); // Gradient X
	cv::Sobel(mImgGray, yGradImg, ddepth, 0, 1, 5); // Gradient Y
	for (int i=0; i<lineSegments.size(); ++i)  {
		lineDescriptor_MSLD(lineSegments[i], &xGradImg, &yGradImg);
	}
}

void MfgSingleView::getIdealLineMSLDs () {
	for (int i=0; i< lineMemberSegmentIdx.size(); ++i) {
		for (int j=0; j <lineMemberSegmentIdx[i].size(); ++j) {
			for (int k=0; k<lineMemberSegmentIdx[i][j].size(); ++k) {
				idealLineGroups[i][j].vec_MSLD.push_back(
					lineSegments[lineMemberSegmentIdx[i][j][k]].desc_MSLD);
			}
		}
	}
}


void MfgSingleView::drawIdealLineGroups() {
	cv::Mat canvas = mImg.clone();
	string winName = "ideal lines:"+num2str(idealLineGroups[0].size());
	for (int i=0; i<idealLineGroups.size(); ++i){
		cout<<"Ideal Line Group "<<i+1<<" : "
			<<idealLineGroups[i].size()<<endl;
		cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
		for (int j=0; j<idealLineGroups[i].size(); ++j) {			
			cv::line(canvas, idealLineGroups[i][j].ptA.cvPt(),
				idealLineGroups[i][j].ptB.cvPt(), color, 2);
			//cv::circle(canvas, idealLineGroups[i][j].ptA.cvPt(),3,color,2);
			//cv::circle(canvas, idealLineGroups[i][j].ptB.cvPt(),3,color,2);
			/*for (int k=0; k<idealLineGroups[i][j].lineSegmentIndex.size(); ++k)
				{cv::line(canvas,
				lineSegments[idealLineGroups[i][j].lineSegmentIndex[k]].ptA.cvPt(),
				lineSegments[idealLineGroups[i][j].lineSegmentIndex[k]].ptB.cvPt(),
				color,2);}*/
			//showImage("ideal lines",canvas);
			//cv::waitKey(0);
		}
		showImage(winName,&canvas);
		cv::waitKey(0);		
	}
//	cv::destroyWindow(winName);
}

void MfgSingleView::drawLineSegmentGroups() // draw grouped line segments
{
	cv::Mat canvas = mImg.clone();
	for (int i=0; i<lineSegmentGroups.size(); ++i){
		cout<<"Line Segment Group "<<i+1<<" : "
			<<lineSegmentGroups[i].size()<<endl;
		cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
		for (int j=0; j<lineSegmentGroups[i].size(); ++j) {
			cv::line(canvas,lineSegments[lineSegmentGroups[i][j]].ptA.cvPt(),
				lineSegments[lineSegmentGroups[i][j]].ptB.cvPt(),
				color,2);
		}
		showImage("Grouped line segments",&canvas);
		cv::waitKey(0);
	}	
}

void MfgSingleView::drawAllLineSegments()
{
	cv::Mat canvas = mImg.clone();
	cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
	for (int i=0; i<lineSegments.size(); ++i)
		cv::line(canvas,lineSegments[i].ptA.cvPt(),
		lineSegments[i].ptB.cvPt(),
		color,1);

	showImage("All line segments",&canvas);
	cv::waitKey(0);
	//cv::destroyWindow("All line segments");
}

void MfgSingleView::drawPersCorrResult(cv::Mat H)
{
	cv::Mat canvas = mImg.clone(), result;
	for (int i=0; i<3;/*idealLineGroups.size();*/ ++i){
		cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
		for (int j=0; j<idealLineGroups[i].size(); ++j) {			
			cv::line(canvas, idealLineGroups[i][j].ptA.cvPt(),
				idealLineGroups[i][j].ptB.cvPt(), color, 2);
		}
	}
	cv::Mat offset = cv::Mat::eye(3,3,CV_64F);
	double xshift = 10, yshift = 50;
	offset.at<double>(0,2) = xshift;
	offset.at<double>(1,2) = yshift;
	//cout<<offset<<endl;
	cv::warpPerspective(canvas,result,H*offset,cv::Size(canvas.cols+50,
		canvas.rows+100));
	showImage("corrected view",&result);
	cv::waitKey(0);
}

/*
void MfgSingleView::groupLineSegments () 
	// roughly group line segments according to vanishing points
	// and initialize vanishPoints.
	// the first three groups correspond to the 3 dominant vp
	// the fourth group may not have a vanishing point
{
	//double lengthThresh = mImg.cols/100.0;
	double angleThresh	 = 15 * PI/180;       // angle constraint for vertical lines
	vector<MfgLineSegment>	ls = lineSegments;
	vector<int> oriIdx;  // original index of ls
	for (int i=0; i<ls.size(); ++i) oriIdx.push_back(i);
	// ======== 1. group vertical line segments using angle threshold =======
	double tanThresh = tan(angleThresh);
	vector<int> vertGroupIdx;
	for (int i = 0; i < ls.size(); ++i)	{
		if (ls[i].ptA.y_ih()-ls[i].ptB.y_ih() != 0 && 
			abs((ls[i].ptA.x_ih()-ls[i].ptB.x_ih())/
			(ls[i].ptA.y_ih()-ls[i].ptB.y_ih())) <tanThresh)
			vertGroupIdx.push_back(i);
	}
	vector<int> lineGroupOriIdx;  // index in lineSegments
	while(!vertGroupIdx.empty()) { // remove from ls 
		lineGroupOriIdx.push_back(oriIdx[vertGroupIdx.back()]);//order matters!
		ls.erase(ls.begin()+vertGroupIdx.back());			
		oriIdx.erase(oriIdx.begin()+vertGroupIdx.back());
		vertGroupIdx.pop_back();
	}
	lineSegmentGroups.push_back(lineGroupOriIdx); 
	vanishPoints.push_back(Mfg2dPoint(cv::Mat(2,1,CV_64F)));

	// ---- 2. group non-vertial line segments using sequential RANSAC ------
	// this step can also use multi-RANSAC or J-Linkage ~ to be implemented

	int	maxIterNo	  = 500;   // max RANSAC iteration number
	double disThresh  = mImg.cols/20; // dist between line and intersect point
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	int	groupNo  = 2;	
	for (int count = 0; count < groupNo; ++count) {
		int	inlierNoThresh	= ls.size()/groupNo; 
		vector<int>    maxInlierIdx;    
		vector<int>	inlierIdx;
		for (int i=0; i < maxIterNo; ++i) {
			int j = rand() % ls.size();
			int k = rand() % ls.size();
			if (j==k) { // if j=k, then redo it
				--i;
				continue;
			}		
			if (abs(ls[j].matLnEq(cv::Range(0,2),cv::Range::all()).dot(
				ls[k].matLnEq(cv::Range(0,2),cv::Range::all()))) < 0.5) {
					// when two lines have too large angle (60), ignore them
					--i;
					continue;
			}
			cv::Mat crsPt = ls[j].matLnEq.cross(ls[k].matLnEq);
			crsPt = crsPt/crsPt.at<double>(2); // devided by last element (?0!)
			// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
			for (int ii=0; ii<ls.size(); ++ii) {
				double dis = abs(crsPt.dot(ls[ii].matLnEq)); //dist-pt and line
				if (dis < disThresh)
					inlierIdx.push_back(ii);
			}
			if (inlierIdx.size() > maxInlierIdx.size()){
				maxInlierIdx.clear();
				maxInlierIdx = inlierIdx;
				if (maxInlierIdx.size() > inlierNoThresh) 
					break;
			}	
			inlierIdx.clear();
		}
		// Next is guided matching, to find more inliers
		cv::Mat newCrsPt;
		while(1){
			cv::Mat A(3,maxInlierIdx.size(), CV_64F); // A'*x = 0;
			inlierIdx.clear();			
			if (maxInlierIdx.size()==1)
				newCrsPt = ls[maxInlierIdx[0]].matLnEq;
			else {
				for (int i = 0; i < maxInlierIdx.size(); ++i) {					
					ls[maxInlierIdx[i]].matLnEq.copyTo(A.col(i));
				}			
				cv::SVD::solveZ(A.t(),newCrsPt); //		
				newCrsPt = newCrsPt/newCrsPt.at<double>(2); // ERROR-PRONE !!!
				//!!!!!!!!!!!!!!!!!!!
			}
			for (int i=0; i<ls.size(); ++i) {
				double dis = abs(newCrsPt.dot(ls[i].matLnEq)); //pt to ln dist
				if (dis < disThresh)
					inlierIdx.push_back(i);
			}
			if (inlierIdx.size() > maxInlierIdx.size())
				maxInlierIdx = inlierIdx;
			else
				break;
		}
		vector<int> maxInlierOriIdx;
		while(!maxInlierIdx.empty()) { // remove found inliers from ls set 
			maxInlierOriIdx.push_back(oriIdx[maxInlierIdx.back()]);
			ls.erase(ls.begin()+maxInlierIdx.back());			
			oriIdx.erase(oriIdx.begin()+maxInlierIdx.back());
			maxInlierIdx.pop_back();
		}
		lineSegmentGroups.push_back(maxInlierOriIdx);
		vanishPoints.push_back(Mfg2dPoint(newCrsPt));
	}
	lineSegmentGroups.push_back(oriIdx); // the rest segments form a group
}
void MfgSingleView::iterativeLineFilter ()
{
	idealLineGroups.resize(lineSegmentGroups.size());
	lineMemberSegmentIdx.resize (idealLineGroups.size());
	for (int i=0; i<lineSegmentGroups.size(); ++i) 	{
		for (int j=0; j<lineSegmentGroups[i].size(); ++j) {
			idealLineGroups[i].push_back
				(lineSegments[lineSegmentGroups[i][j]]);
			vector<int> idxVec;
			idxVec.push_back(lineSegmentGroups[i][j]);
			lineMemberSegmentIdx[i].push_back(idxVec);
		}		
	}	
	for (int i=0; i<idealLineGroups.size(); ++i) {
		int sz = idealLineGroups[i].size()+1;
		cv::Mat vp;
		while (idealLineGroups[i].size()<sz) {				
			sz = idealLineGroups[i].size();
			combineLineSegments(idealLineGroups[i], 
				lineMemberSegmentIdx[i], mImg.cols);
			if (i == idealLineGroups.size()-1)
				continue; // for the last group, no need to use concurrency
			filterConcurrentLines (idealLineGroups[i],
				idealLineGroups[idealLineGroups.size()-1],
				lineMemberSegmentIdx[i], 
				lineMemberSegmentIdx[idealLineGroups.size()-1],	vp, mImg.cols);
			vanishPoints[i] = Mfg2dPoint(vp);
			if (i == 0) { // after pers-correct, use anlge to filter vert lines
				pcMat = vertPersCorrect();
			//	drawPersCorrResult(pcMat);
				double vAT = 1.5*PI/180.0; // vertical angle threshold 
				for (int j=0; j<idealLineGroups[i].size(); ++j) {
					cv::Mat lEq 
						= pcMat.t().inv()*idealLineGroups[i][j].matLnEq;
					if (abs(lEq.at<double>(1)/lEq.at<double>(0))>tan(vAT)) {
						idealLineGroups[idealLineGroups.size()-1].push_back
							(idealLineGroups[i][j]); // recycle into last group
						lineMemberSegmentIdx[idealLineGroups.size()-1].
							push_back(lineMemberSegmentIdx[i][j]);
						idealLineGroups[i].erase
							(idealLineGroups[i].begin()+j);
						lineMemberSegmentIdx[i].erase
							(lineMemberSegmentIdx[i].begin()+j);
						--j;
					}
				}
			}
		}
	}	
}
void MfgSingleView::fitIdealLines()
	// combine line segments into ideal lines
{
	double distThresh = mImg.cols/250.0; //thresh for dist of endpoints to line 
	double lenThresh = mImg.cols/30.0;
	int    inlierNumThresh	= 1;   // min 
	int    maxIterNum		= 20; // max ransac iteration number
	for (int i = 0; i<4; ++i)	{ // for each vanishing point group
		vector<int> iLsGroup = lineSegmentGroups[i];
		vector<vector<int>> iGpLnMemSegIdx;
		vector<MfgLineSegment> iLineGroup; // ideal line group
		if (i>0)
			distThresh = distThresh*0.7; //smaller thersh for horizontal lines
		// --- sequential ransac ----	
		while (iLsGroup.size()>0) { //
			vector<int> maxInlierSet;
			vector<int> lnMemSegIdx;
			int iter = 0;
			while (iter < maxIterNum) { // 
				++iter;
				vector<int> curInliers;
				int j = rand() % iLsGroup.size(); //a random segment
				cv::Mat jLnEq = lineSegments[iLsGroup[j]].matLnEq;
		       	for (int k=0; k<iLsGroup.size(); ++k) { 
					cv::Mat kPtA = lineSegments[iLsGroup[k]].ptA.matHomo(),
						kPtB = lineSegments[iLsGroup[k]].ptB.matHomo();
				//	double d = abs(point2LineDist(jLnEq,kPtA))+
				//				abs(point2LineDist(jLnEq,kPtB));
					// d is the sum of 2 endpoint distance to jth line
					double d = 
						getSmallerLs2LstDist(lineSegments[iLsGroup[j]],
											 lineSegments[iLsGroup[k]]);					
					if (d<distThresh)
						curInliers.push_back(k);
				}
				if (curInliers.size()>maxInlierSet.size())
					maxInlierSet = curInliers;
				if (maxInlierSet.size()>=inlierNumThresh)
					break;
			}
			// ==== guided matching ====
			cv::Mat newLineEq;
			while (true) {
				vector<int> inliers;
				cv::Mat A(3,maxInlierSet.size()*2, CV_64F); // A'*x = 0;
				for (int ii = 0; ii < maxInlierSet.size(); ++ii) {
					lineSegments[iLsGroup[maxInlierSet[ii]]].ptA.matHomo().
						copyTo(A.col(ii));
					lineSegments[iLsGroup[maxInlierSet[ii]]].ptB.matHomo().
						copyTo(A.col(ii+maxInlierSet.size()));
				}
				if (maxInlierSet.size()==1)
					newLineEq = 
					lineSegments[iLsGroup[maxInlierSet[0]]].matLnEq;
				else {
					cv::SVD::solveZ(A.t(),newLineEq); //
					newLineEq = newLineEq/sqrt(pow(newLineEq.at<double>(0),2)+
									pow(newLineEq.at<double>(1),2)); 
				}
				for (int jj=0; jj<iLsGroup.size(); ++jj) {
					cv::Mat jjPtA = lineSegments[iLsGroup[jj]].ptA.matHomo(),
						jjPtB = lineSegments[iLsGroup[jj]].ptB.matHomo();
					double d = abs(jjPtA.dot(newLineEq)/jjPtA.at<double>(2))+
						abs(jjPtB.dot(newLineEq)/jjPtB.at<double>(2));
					if (d < distThresh)
						inliers.push_back(jj);
				}
				if (inliers.size() > maxInlierSet.size())
					maxInlierSet = inliers;
				else
					break;
			}
			if (maxInlierSet.size() < inlierNumThresh)
				break;
			else {
				MfgLineSegment idealLine(newLineEq.at<double>(0),
					newLineEq.at<double>(1),newLineEq.at<double>(2));				
				// find two endpoints of ideal line
				// and put maxInlierSet into lineSegmentIndex
				if (maxInlierSet.size()==1)	{
					if(lineSegments[iLsGroup[maxInlierSet[0]]].length()
						<lenThresh) { // discard too short line segment
						iLsGroup.erase(iLsGroup.begin()+maxInlierSet[0]);
						continue;
					}
					else {
					idealLine.ptA= lineSegments[iLsGroup[maxInlierSet[0]]].ptA;
					idealLine.ptB= lineSegments[iLsGroup[maxInlierSet[0]]].ptB;
					lnMemSegIdx.push_back(iLsGroup[maxInlierSet[0]]);
					}
				} else {
					double minPrdt = 1e10, maxPrdt = -1e10; 
					cv::Mat mPt1, mPt2; //record the left and rightmost pts
					double lineDirection[2] 
							= {idealLine.y_ih(),-idealLine.x_ih()};
					cv::Mat lineVector(2,1,CV_64F,lineDirection);
					for (int k = 0; k<maxInlierSet.size(); ++k) {
						lnMemSegIdx.push_back(iLsGroup[maxInlierSet[k]]);
						cv::Mat 
							kLsPtA = lineSegments
								   [iLsGroup[maxInlierSet[k]]].ptA.matInhomo(), 
							kLsPtB = lineSegments
								   [iLsGroup[maxInlierSet[k]]].ptB.matInhomo();
						double prdtA = lineVector.dot(kLsPtA),
							prdtB = lineVector.dot(kLsPtB);
						if (minPrdt > prdtA || minPrdt > prdtB) {
							if (prdtA < prdtB)	{
								minPrdt = prdtA;
								mPt1 = kLsPtA;
							} else {
								minPrdt = prdtB;
								mPt1 = kLsPtB;
							}
						}
						if (maxPrdt < prdtA || maxPrdt < prdtB) {
							if (prdtA > prdtB) 	{
								maxPrdt = prdtA;
								mPt2 = kLsPtA;
							} else	{
								maxPrdt = prdtB;
								mPt2 = kLsPtB;
							}
						}					
					}
					cv::Mat nearestPtA= findNearestPointOnLine(newLineEq,mPt1),
						nearestPtB = findNearestPointOnLine(newLineEq,mPt2);
					idealLine.ptA = Mfg2dPoint(nearestPtA);
					idealLine.ptB = Mfg2dPoint(nearestPtB);
				}
				for (int k = maxInlierSet.size(); k>0; --k)//rmv from iLsGroups
				{
					iLsGroup.erase(iLsGroup.begin()+maxInlierSet[k-1]);
				}
				iLineGroup.push_back(idealLine);
				iGpLnMemSegIdx.push_back(lnMemSegIdx);
			}
		}
		idealLineGroups.push_back(iLineGroup);
		lineMemberSegmentIdx.push_back(iGpLnMemSegIdx);
	}
}
*/
