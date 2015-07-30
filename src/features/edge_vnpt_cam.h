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

#ifndef G2O_EDGE_VNTP_CAM_H_
#define G2O_EDGE_VNTP_CAM_H_

#include "g2o/core/base_binary_edge.h"
#include "vertex_vnpt.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"

namespace g2o
{
// first two args are the measurement type, second two the connection classes
class EdgeVnptCam : public BaseBinaryEdge<2, Eigen::Vector2d, VertexVanishPoint, VertexCam>
{
public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeVnptCam();
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    // return the error estimate as a 2-vector
    void computeError();
    // jacobian
//   virtual void linearizeOplus();

    virtual void setMeasurement(const Eigen::VectorXd &m)
    {
        _measurement = m;
    }

    virtual bool setMeasurementData(const double *d)
    {
        Eigen::Map<const Eigen::Vector2d> v(d, 2);
        _measurement = v;
        return true;
    }

    virtual bool getMeasurementData(double *d) const
    {
        Eigen::Map<Eigen::Vector2d> v(d, 2);
        v = _measurement;
        return true;
    }

    virtual int measurementDimension() const
    {
        return 2;
    }

    virtual double initialEstimatePossible(const OptimizableGraph::VertexSet &from,
                                           OptimizableGraph::Vertex *to)
    {
        (void) to;
        return (from.count(_vertices[0]) == 1 ? 1.0 : -1.0);
    }

    virtual void initialEstimate(const OptimizableGraph::VertexSet &from, OptimizableGraph::Vertex *to) {}

    Eigen::Matrix<double, 2, 2> vptCov;

private:
    Eigen::Matrix < double, 2, 3 + 6 > J; 
};

#ifdef G2O_HAVE_OPENGL
class EdgeVnptCamDrawAction: public DrawAction
{
public:
    EdgeVnptCamDrawAction();
    virtual HyperGraphElementAction *operator()(HyperGraph::HyperGraphElement *element,
            HyperGraphElementAction::Parameters *params_);
};
#endif

}
#endif
