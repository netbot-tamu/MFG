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
 * interface for the EPNP program
 ********************************************************************************/

//#include "mfg.h"
#include "utils.h"
#include "epnp.h"
#include <vector>

using namespace std;

int computePnP(vector<cv::Point3d> X, vector<cv::Point2d> x, cv::Mat K,
               cv::Mat &R, cv::Mat &t)
{
    int n = X.size();
    epnp PnP;
    PnP.set_internal_parameters(K.at<double>(0, 2), K.at<double>(1, 2),
                                K.at<double>(0, 0), K.at<double>(1, 1));
    PnP.set_maximum_number_of_correspondences(n);

    double R_true[3][3], t_true[3];

    PnP.reset_correspondences();

    for (int i = 0; i < n; i++)
        PnP.add_correspondence(X[i].x, X[i].y, X[i].z, x[i].x, x[i].y);

    double R_est[3][3], t_est[3];
    double err2 = PnP.compute_pose(R_est, t_est);

    R = (cv::Mat_<double>(3, 3) << R_est[0][0], R_est[0][1], R_est[0][2],
         R_est[1][0], R_est[1][1], R_est[1][2],
         R_est[2][0], R_est[2][1], R_est[2][2]);
    t = (cv::Mat_<double>(3, 1) << t_est[0], t_est[1], t_est[2]);

    return 0;
}


int computePnP_ransac(vector<cv::Point3d> X, vector<cv::Point2d> x, cv::Mat K,
                      cv::Mat &R, cv::Mat &t, int maxIter)
{
    int N = X.size();
    int n = 7; // min set

    if (N < n)
    {
        computePnP(X, x, K, R, t);
        return -1;
    }

    double imptDistThresh = 3;
    vector<int> maxInlierSet;
    cv::Mat Rmax, tmax;
    vector<int> rnd;

    for (int i = 0; i < N; ++i)		rnd.push_back(i);

    int	iter = 0;

    while (iter < maxIter)
    {
        ++iter;
        vector<int> inlier;
        // --- choose minimal solution set: mss1<->mss2
        random_unique(rnd.begin(), rnd.end(), n);
        vector<cv::Point3d> Xmin(n);
        vector<cv::Point2d> xmin(n);

        for (int i = 0; i < n; ++i)
        {
            Xmin[i] = X[rnd[i]];
            xmin[i] = x[rnd[i]];
        }

        cv::Mat Rm, tm;
        computePnP(X, x, K, Rm, tm);

        for (int i = 0; i < N; ++i)
        {
            double d = cv::norm(mat2cvpt(K * (Rm * cvpt2mat(X[i], 0) + tm)) - x[i]);

            if (d < imptDistThresh)
                inlier.push_back(i);
        }

        if (inlier.size() > maxInlierSet.size())
        {
            maxInlierSet = inlier;
            Rmax = Rm;
            tmax = tm;
        }
    }

    if (maxInlierSet.size() < n)
    {
        computePnP(X, x, K, R, t);
        return n;
    }
    else
    {
        //R = Rmax;
        //t = tmax;
        vector<cv::Point3d> Xin;
        vector<cv::Point2d> xin;

        for (int i = 0; i < maxInlierSet.size(); ++i)
        {
            Xin.push_back(X[maxInlierSet[i]]);
            xin.push_back(x[maxInlierSet[i]]);
        }

        computePnP(Xin, xin, K, R, t);
        return maxInlierSet.size();
    }
}
