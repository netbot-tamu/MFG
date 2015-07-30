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
 * G2O edge modeling the distance from a 3D line to a 3D plane where the
 * line direction is determined by a vanishing point
 ********************************************************************************/

#include "utils.h"

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <iostream>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include "Eigen/src/SVD/JacobiSVD.h"
#include "edge_line_vp_plane.h"
#include "vertex_vnpt.h"
#include "vertex_plane.h"

namespace g2o
{

bool EdgeLineVpPlane::read(std::istream &is)
{
    return true;
}

bool EdgeLineVpPlane::write(std::ostream &os) const
{
    return true;
}

void EdgeLineVpPlane::computeError()
{
    VertexSBAPointXYZ *v_lnpt = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
    VertexVanishPoint *v_vp = static_cast<VertexVanishPoint *>(_vertices[1]);
    const VertexPlane3d *v_plane = static_cast<const VertexPlane3d *>(_vertices[2]);

    // represent the line by two points
    cv::Point3d lnpt0(v_lnpt->estimate()(0), v_lnpt->estimate()(1), v_lnpt->estimate()(2));
    cv::Point3d lnpt1 = lnpt0 + cv::Point3d(v_vp->estimate()(0), v_vp->estimate()(1), v_vp->estimate()(2));

    // project real endpoints onto the line
    cv::Point3d A = projectPt3d2Ln3d(lnpt0, lnpt1, endptA);
    cv::Point3d B = projectPt3d2Ln3d(lnpt0, lnpt1, endptB);

    Eigen::Vector3d vA(A.x, A.y, A.z);
    Eigen::Vector3d vB(B.x, B.y, B.z);
    // compute line (endpoints) to plane distance
    double d = 1 / v_plane->estimate().norm();  // plane depth
    Eigen::Vector3d n = v_plane->estimate() * d; // plane unit normal
    _error(0) = sqrt(pow(vA.dot(n) + d, 2) + pow(vB.dot(n) + d, 2));
}
}
