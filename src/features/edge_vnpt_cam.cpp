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
 * G2O edge modeling the reprojection error of a vanishing point in a camera
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
#include "edge_vnpt_cam.h"

namespace g2o
{
using namespace std;

// point to camera projection, monocular
EdgeVnptCam::EdgeVnptCam() : BaseBinaryEdge<2, Eigen::Vector2d, VertexVanishPoint, VertexCam>()
{
    information().setIdentity();
    J.fill(0);

    //	resizeParameters(1);

}

bool EdgeVnptCam::read(std::istream &is)
{
    int pId;
    is >> pId;
    setParameterId(0, pId);
    // measured endpoints
    Eigen::VectorXd meas;

    for (int i = 0; i < 2; i++) is >> meas[i];

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

bool EdgeVnptCam::write(std::ostream &os) const
{
    for (int i = 0; i < 2; i++) os  << measurement()[i] << " ";

    for (int i = 0; i < information().rows(); i++)
        for (int j = i; j < information().cols(); j++)
            os <<  information()(i, j) << " ";

    return os.good();
}


void EdgeVnptCam::computeError()
{
    // from cam to point (track)
    //VertexSE3 *cam = static_cast<VertexSE3*>(_vertices[0]);

    //	Eigen::Eigen::Vector3d perr = cache->w2n() * point->estimate();
    VertexVanishPoint *v_vp = static_cast<VertexVanishPoint *>(_vertices[0]);
    const VertexCam *cam = static_cast<const VertexCam *>(_vertices[1]);


    // calculate the error: 3D unit vector angles
    const Eigen::Vector3d &vp = v_vp->estimate();
    Eigen::Vector4d hvp(vp(0), vp(1), vp(2), 0); // homogeneous, last element = 0 for vanishing point
    Eigen::Vector3d ihvp_c = cam->estimate().w2n * hvp; // inhomogenous, in camera coordinates
    cv::Mat ihvp_c_mat = (cv::Mat_<double>(3, 1) << ihvp_c(0), ihvp_c(1), ihvp_c(2));
    ihvp_c_mat = ihvp_c_mat / cv::norm(ihvp_c_mat);
    cv::Mat meas_univec = angle2unitVec(_measurement(0), _measurement(1));
    double ang = acos(min(1.0, abs(meas_univec.dot(ihvp_c_mat))));
    _error(0) = ang;
    _error(1) = ang;

}


} // end namespace
