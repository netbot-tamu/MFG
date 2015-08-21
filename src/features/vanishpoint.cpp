/////////////////////////////////////////////////////////////////////////////////
//
//  Multilayer Feature Graph (MFG), version 1.0
//  Copyright (C) 2011-2015 Yan Lu, Dezhen Song
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

/********************************************************************************
 * Compute vanishing points using optimization
 ********************************************************************************/

#include "utils.h"
#include "consts.h"
#include "levmar.h"

#include <vector>

using namespace std;

void refineVanishPt(const vector<LineSegmt2d> &allLs, vector<int> &lsIdx,
                    cv::Mat &vp)
// input: line segments supposed to pass one vp ()
// output: refined vp and line segment classification
{
    double distThresh = 1;  // absolute dist
    double vp2LineDistThresh =	tan(1.5 * PI / 180); // normalized dist
    // 0. collect line segments, and compute initial vp
    vector<LineSegmt2d> ls(lsIdx.size());

    for (int i = 0; i < lsIdx.size(); ++i)
        ls[i] = allLs[lsIdx[i]];

    if (cv::sum(vp).val != 0)    // no initial vp, need to compute by DLT
    {
        cv::Mat A(3, ls.size(), CV_64F); // A'*vp = 0;

        for (int i = 0; i < ls.size(); ++i)
        {
            cv::Mat eq = ls[i].lineEq();
            eq.copyTo(A.col(i));
        }

        if (A.cols == 2)
        {
            cv::SVD svd(A.t());
            vp = svd.vt.row(2).t();
        }
        else
            cv::SVD::solveZ(A.t(), vp);
    }

    // 1. compute optimal vp with given line segments
    if (ls.size() >= 3)
        optimizeVainisingPoint(ls, vp);

    // 2. filter out outliers
    for (int i = 0; i < ls.size(); ++i)
    {
        //	if (mleVp2LineDist (vp, ls[i]) > distThresh) {
        if (mleVp2LineDist(vp, ls[i]) / ls[i].length() > vp2LineDistThresh)
        {
            // if dist too large, discard from group
            ls.erase(ls.begin() + i);
            lsIdx.erase(lsIdx.begin() + i);
            --i;
        }
    }
}

void refineVanishPt(const vector<LineSegmt2d> &allLs, vector<int> &lsIdx,
                    cv::Mat &vp, cv::Mat &cov, cv::Mat &covhomo)
// input: line segments supposed to pass one vp ()
// output: refined vp and line segment classification
{
    double distThresh = 1;  // absolute dist
    double vp2LineDistThresh =	tan(1.5 * PI / 180); // normalized dist
    // 0. collect line segments, and compute initial vp
    vector<LineSegmt2d> ls(lsIdx.size());

    for (int i = 0; i < lsIdx.size(); ++i)
        ls[i] = allLs[lsIdx[i]];

    if (cv::sum(vp).val != 0)    // no initial vp, need to compute by DLT
    {
        cv::Mat A(3, ls.size(), CV_64F); // A'*vp = 0;

        for (int i = 0; i < ls.size(); ++i)
        {
            cv::Mat eq = ls[i].lineEq();
            eq.copyTo(A.col(i));
        }

        if (A.cols == 2)
        {
            cv::SVD svd(A.t());
            vp = svd.vt.row(2).t();
        }
        else
            cv::SVD::solveZ(A.t(), vp);
    }

    // 1. compute optimal vp with given line segments
    if (ls.size() >= 3)
        optimizeVainisingPoint(ls, vp, cov, covhomo);

    // 2. filter out outliers
    for (int i = 0; i < ls.size(); ++i)
    {
        //	if (mleVp2LineDist (vp, ls[i]) > distThresh) {
        if (mleVp2LineDist(vp, ls[i]) / ls[i].length() > vp2LineDistThresh)
        {
            // if dist too large, discard from group
            ls.erase(ls.begin() + i);
            lsIdx.erase(lsIdx.begin() + i);
            --i;
        }
    }
}

struct data_VpMleEst
{
    vector<LineSegmt2d> ls;
};

void costFun_VpMleEst(double *p, double *error, int m, int n, void *adata)
{
    struct data_VpMleEst *dptr;
    dptr = (struct data_VpMleEst *) adata;
    vector<LineSegmt2d> ls = dptr->ls;
    cv::Mat vp = (cv::Mat_<double>(3, 1) << p[0], p[1], p[2]);

    for (int i = 0; i < n; ++i)
        error[i] = mleVp2LineDist(vp, ls[i]);

//	cout<<p[0]<<" "<<p[1]<<"==>"<<cost<<endl;
}

void optimizeVainisingPoint(vector<LineSegmt2d> &lines, cv::Mat &vp)
// Use iterative optimization method (LM algorithm) to find a near optimal
// vanising point for a group of lines.
// vp is 3x1 vector
{
    int n = lines.size();
    double *measurement = new double[n];

    for (int i = 0; i < n; ++i) measurement[i] = 0;

    double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
    opts[0] = LM_INIT_MU * 1;
    opts[1] = 1E-15;
    opts[2] = 1E-50;
    opts[3] = 1E-20;
    opts[4] = LM_DIFF_DELTA;
    int maxIter = 1000;

    double *cov = new double[9];
    double para[3] = {vp.at<double>(0), vp.at<double>(1), vp.at<double>(2)};

    data_VpMleEst dataMle;
    dataMle.ls = lines;

    int ret = dlevmar_dif(costFun_VpMleEst, para, measurement, 3, n,
                          maxIter, opts, info, NULL, cov, (void *)&dataMle);
    vp.at<double>(0) = para[0];
    vp.at<double>(1) = para[1];
    vp.at<double>(2) = para[2];
    cv::Mat covar(3, 3, CV_64F, cov);
    cv::Mat J = (cv::Mat_<double>(2, 3) << 1 / para[2], 0, -para[0] / para[2] / para[2],
                 0, 1 / para[2], -para[1] / para[2] / para[2]);
    cv::Mat COV = J * covar * J.t();

    delete[] measurement;
    delete cov;
}

void optimizeVainisingPoint(vector<LineSegmt2d> &lines, cv::Mat &vp, cv::Mat &covMat, cv::Mat &covHomo)
// Use iterative optimization method (LM algorithm) to find a near optimal
// vanising point for a group of lines.
// vp is 3x1 vector
{
    int n = lines.size();
    double *measurement = new double[n];

    for (int i = 0; i < n; ++i) measurement[i] = 0;

    double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
    opts[0] = LM_INIT_MU * 1;
    opts[1] = 1E-15;
    opts[2] = 1E-50;
    opts[3] = 1E-20;
    opts[4] = LM_DIFF_DELTA;
    int maxIter = 1000;

    double *cov = new double[9];
    double para[3] = {vp.at<double>(0), vp.at<double>(1), vp.at<double>(2)};

    data_VpMleEst dataMle;
    dataMle.ls = lines;

    int ret = dlevmar_dif(costFun_VpMleEst, para, measurement, 3, n,
                          maxIter, opts, info, NULL, cov, (void *)&dataMle);
    vp.at<double>(0) = para[0];
    vp.at<double>(1) = para[1];
    vp.at<double>(2) = para[2];
    // cov in homo img coord
    cv::Mat covar(3, 3, CV_64F, cov);
    covHomo = covar;
    cv::Mat J = (cv::Mat_<double>(2, 3) << 1 / para[2], 0, -para[0] / para[2] / para[2],
                 0, 1 / para[2], -para[1] / para[2] / para[2]);
    // cov in inhomog image coord
    covMat = J * covar * J.t();
    delete[] measurement;
    delete cov;
}

double mleVp2LineDist(cv::Mat vp, LineSegmt2d l)
// MLE of the distance between a vanishing point and a line supposedly passing
// through it.
// method: solving the optimal line passing vp such that the distance from the
// original line endpoints to the new line is minimum, and this minimal
// distance is the returned value.
// vp is 3x1 vector, homogeneous vector
{
    double x1 = l.endpt1.x,
           y1 = l.endpt1.y,
           x2 = l.endpt2.x,
           y2 = l.endpt2.y,
           vx = vp.at<double>(0),
           vy = vp.at<double>(1),
           vw = vp.at<double>(2);

    double sum_square_dist;

    if (vw != 0 && abs(vx / vw) < 1e10 && abs(vy / vw) < 1e10) // finite vp
    {
        vx = vx / vw; // convert to inhomogeneous vector
        vy = vy / vw;
        double	A = pow(x1 - vx, 2) + pow(x2 - vx, 2),
                B = pow(y1 - vy, 2) + pow(y2 - vy, 2),
                C = 2 * ((x1 - vx) * (y1 - vy) + (x2 - vx) * (y2 - vy));
        sum_square_dist = (A + B - sqrt((A - B) * (A - B) + C * C)) / 2;
    }
    else  	// inifinite vp
    {
        double	si = vy / sqrt(vx * vx + vy * vy),
                co = vx / sqrt(vx * vx + vy * vy);
        sum_square_dist = pow(x1 * si - y1 * co, 2) + pow(x2 * si - y2 * co, 2)
                          + pow(x1 * si - y1 * co + x2 * si - y2 * co, 2) / 2;
    }

    return sqrt(abs(sum_square_dist) / 2);
}


