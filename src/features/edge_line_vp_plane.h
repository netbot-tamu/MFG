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

#ifndef G2O_EDGE_LINE_VP_PLANE_H_
#define G2O_EDGE_LINE_VP_PLANE_H_

#include "g2o/core/base_multi_edge.h"

#include "vertex_vnpt.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"


namespace g2o
{

class EdgeLineVpPlane : public BaseMultiEdge<1, double>
{
public:
    //      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeLineVpPlane()
    {
        information().setIdentity();
        resize(3);
    }
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;
    void computeError();

    void setMeasurement(const double &m)
    {
        _measurement = m;
    }

//	protected:
    cv::Point3d endptA, endptB; // line segment endpoints
};


} // end namespace


#endif
