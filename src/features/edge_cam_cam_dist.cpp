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
 * G2O edge modeling euclidean distance between two camera nodes
 ********************************************************************************/

#include "edge_cam_cam_dist.h"

namespace g2o
{
using namespace std;

// point to camera projection, monocular
EdgeCamCamDist::EdgeCamCamDist() : BaseBinaryEdge<1, double, VertexCam, VertexCam>()
{
    information().setIdentity();
    J.fill(0);
    //  resizeParameters(1);
}

bool EdgeCamCamDist::read(std::istream &is)
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

bool EdgeCamCamDist::write(std::ostream &os) const
{
    os  << measurement() << " ";

    for (int i = 0; i < information().rows(); i++)
        for (int j = i; j < information().cols(); j++)
            os <<  information()(i, j) << " ";

    return os.good();
}

void EdgeCamCamDist::computeError()
{
    const VertexCam *cam0 = static_cast<const VertexCam *>(_vertices[0]);
    const VertexCam *cam1 = static_cast<const VertexCam *>(_vertices[1]);

    Eigen::Vector3d pos0 = - cam0->estimate().w2n.block<3, 3>(0, 0).transpose() * cam0->estimate().w2n.block<3, 1>(0, 3),
                    pos1 = - cam1->estimate().w2n.block<3, 3>(0, 0).transpose() * cam1->estimate().w2n.block<3, 1>(0, 3);

    _error(0) = (pos1 - pos0).norm() - _measurement;
}

}

