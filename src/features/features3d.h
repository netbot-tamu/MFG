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
 * MFG 3D features
 ********************************************************************************/


#ifndef FEATURES3D_H_
#define FEATURES3D_H_

#include <vector>
#include <utility> // for Pair

#include <opencv2/core/core.hpp>

// This class stores the 3d keypoint's information
class KeyPoint3d
{
public:
    double					x, y, z;
    int						gid;
    int						pGid;		  // plane gid
    std::vector< std::vector<int> > viewId_ptLid; // (viewId, lid of featpt)
    bool					is3D;
    int						estViewId;

    KeyPoint3d()
    {
        is3D = false;
        gid = -1;
        pGid = -1;
    }
    KeyPoint3d(double x_, double y_, double z_)
    {
        x = x_;
        y = y_;
        z = z_;
        gid = -1;
        pGid = -1;
        is3D = false;
        estViewId = -1;
    }
    KeyPoint3d(double x_, double y_, double z_, int gid_, bool is3d_)
    {
        x = x_;
        y = y_;
        z = z_;
        gid = gid_;
        pGid = -1;
        is3D = is3d_;
        estViewId = -1;
    }

    cv::Mat mat(bool homo = true) const
    {
        if (homo)
            return (cv::Mat_<double>(4, 1) << x, y, z, 1);
        else
            return (cv::Mat_<double>(3, 1) << x, y, z);
    }
    cv::Point3d cvpt() const
    {
        return cv::Point3d(x, y, z);
    }
};


class PrimPlane3d
{
public:
    cv::Mat			n;
    double			d;
    int				gid;
    std::vector <int>	ilnGids;  // global ids of coplanar ideal lines
    std::vector <int>	kptGids; // global ids of coplanar key points
    std::vector< std::pair<int, int> >	viewID_vpLid;
    int				estViewId;
    int				recentViewId;

    PrimPlane3d()
    {
        estViewId = -1;
        gid = -1;
    }
    PrimPlane3d(cv::Mat nd, int _gid)
    {
        gid = _gid;

        if (nd.cols * nd.rows == 3) // nd = n/d, is a 3-std::vector
        {
            n = nd / cv::norm(nd);
            d = 1 / cv::norm(nd);
        }
        else     // nd=[n d] is 4-std::vector
        {
            n = (cv::Mat_<double>(3, 1) << nd.at<double>(0), nd.at<double>(1), nd.at<double>(2));
            d = nd.at<double>(3);
        }

        estViewId = -1;
    }
    void setPlane(cv::Mat nd)
    {
        if (nd.cols * nd.rows == 3) // nd = n/d, is a 3-std::vector
        {
            n = nd / cv::norm(nd);
            d = 1 / cv::norm(nd);
        }
        else     // nd=[n d] is 4-std::vector
        {
            n = (cv::Mat_<double>(3, 1) << nd.at<double>(0), nd.at<double>(1), nd.at<double>(2));
            d = nd.at<double>(3);
        }
    }
};


class VanishPnt3d
{
public:
    double					x, y, z, w;
    int						gid;		// global id
    std::vector <int>			idlnGids;
    std::vector< std::vector<int> > viewId_vpLid;
    int						estViewId;

    VanishPnt3d() {}
    VanishPnt3d(double x_, double y_, double z_)
    {
        double len = sqrt(x_ * x_ + y_ * y_ + z_ * z_);
        x = x_ / len;
        y = y_ / len;
        z = z_ / len;
        w = 0;
        gid = -1;
        estViewId = -1;
    }

    cv::Mat mat(bool homo = true)
    {
        cv::Mat v;

        if (homo)
            v = (cv::Mat_<double>(4, 1) << x, y, z, 0);
        else
            v = (cv::Mat_<double>(3, 1) << x, y, z);

        return v / cv::norm(v);
    }

};


class IdealLine3d
{
public:
    cv::Point3d				midpt;
    cv::Mat					direct;  // line direction std::vector
    double					length;	 // length between endpoints
    int						gid;	// global id of line
    int						pGid;	// global id of the associated plane
    int						vpGid;  // global id of the associated vanishing point
    std::vector< std::vector<int> > viewId_lnLid;
    bool					is3D;
    int						estViewId;

    IdealLine3d() {}

    IdealLine3d(cv::Point3d mpt, cv::Mat d)
    {
        midpt = mpt;
        direct = d.clone();
        vpGid = -1;
        pGid = -1;
        gid = -1;
        estViewId = -1;
    }

    cv::Point3d extremity1() const
    {
        cv::Mat d = direct / cv::norm(direct);
        return midpt + 0.5 * length * cv::Point3d(d.at<double>(0), d.at<double>(1), d.at<double>(2));
    }

    cv::Point3d extremity2() const
    {
        cv::Mat d = direct / cv::norm(direct);
        return midpt - 0.5 * length * cv::Point3d(d.at<double>(0), d.at<double>(1), d.at<double>(2));
    }
};

#endif

