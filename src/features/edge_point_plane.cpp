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
 * G2O edge modeling the distance from a 3D point to a 3D plane
 ********************************************************************************/

#include "utils.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <iostream>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include "Eigen/src/SVD/JacobiSVD.h"
#include "edge_point_plane.h"

namespace g2o
{
using namespace std;

// point to camera projection, monocular
EdgePointPlane3d::EdgePointPlane3d() : BaseBinaryEdge<1, double, VertexSBAPointXYZ, VertexPlane3d>()
{
    information().setIdentity();
    J.fill(0);
    
    //	resizeParameters(1);

}

bool EdgePointPlane3d::read(std::istream &is)
{
    int pId;
    is >> pId;
    setParameterId(0, pId);
    // measured endpoints
    double meas;
    is >> meas;
    setMeasurement(meas);

    // information matrix is the identity for features, could be changed to allow arbitrary covariances
    if (is.bad())
        return false;

    for (int i = 0; i < information().rows() && is.good(); i++)
        for (int j = i; j < information().cols() && is.good(); j++)
        {
            is >> information()(i, j);

            if (i != j)
                information()(j, i) = information()(i, j);
        }

    if (is.bad())
    {
        //  we overwrite the information matrix
        information().setIdentity();
    }

    return true;
}

bool EdgePointPlane3d::write(std::ostream &os) const
{
    os  << measurement() << " ";

    for (int i = 0; i < information().rows(); i++)
        for (int j = i; j < information().cols(); j++)
            os <<  information()(i, j) << " ";

    return os.good();
}


void EdgePointPlane3d::computeError()
{
    VertexSBAPointXYZ *v_pt = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
    const VertexPlane3d *v_plane = static_cast<const VertexPlane3d *>(_vertices[1]);

    // calculate the error
    const Eigen::Vector3d pt = v_pt->estimate();
    double d = 1 / v_plane->estimate().norm();  // plane depth
    Eigen::Vector3d n = v_plane->estimate() * d; // plane unit normal
    _error(0) = abs(pt.dot(n) + d);
}


} // end namespace
