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
 * G2O edge modeling the reprojection error of a 3D line in a camera where the
 * line direction is determined by a vanishing point
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
#include "edge_line_vp_cam.h"
#include "vertex_vnpt.h"

namespace g2o
{

bool EdgeLineVpCam::read(std::istream &is)
{
    return true;
}

bool EdgeLineVpCam::write(std::ostream &os) const
{
    return true;
}

void EdgeLineVpCam::computeError()
{
    VertexSBAPointXYZ *v_lnpt = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
    VertexVanishPoint *v_vp = static_cast<VertexVanishPoint *>(_vertices[1]);
    const VertexCam *cam = static_cast<const VertexCam *>(_vertices[2]);

    const Eigen::Vector3d &vp = v_vp->estimate();
    Eigen::Vector4d hvp(vp(0), vp(1), vp(2), 0); // homogeneous, last element = 0 for vanishing point
    Eigen::Vector3d hvp_im = cam->estimate().w2i * hvp; // homogenous, in image coordinates

    const Eigen::Vector3d &lnpt = v_lnpt->estimate();
    Eigen::Vector4d h_lnpt(lnpt(0), lnpt(1), lnpt(2), 1);
    Eigen::Vector3d h_lnpt_im = cam->estimate().w2i * h_lnpt;

    Eigen::Vector3d leq_im = hvp_im.cross(h_lnpt_im); // line equation in image coords
    double leq[3] = {leq_im(0), leq_im(1), leq_im(2)};
    _error(0) = 0;

    for (int i = 0; i < segpts.size(); ++i)
        _error(0) += pow(point2LineDist(leq, segpts[i]), 2);

    _error(0) = sqrt(_error(0));

}
}
