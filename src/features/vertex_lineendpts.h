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
 * G2O vertex representing a 3D line (defined by two points)
 ********************************************************************************/


#ifndef G2O_VERTEX_LINE6D_H_
#define G2O_VERTEX_LINE6D_H_

#include "g2o/types/slam3d/g2o_types_slam3d_api.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"
#include <Eigen/SparseCore>

namespace g2o
{
/**
 * \brief Vertex for a tracked point in space
 */
class VertexLineEndpts : public BaseVertex<6, Eigen::VectorXd>
{
public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexLineEndpts() {}
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    virtual void setToOriginImpl()
    {
        _estimate.fill(0.);
    }

    virtual void oplusImpl(const double *update_)
    {
        Eigen::Map<const Eigen::VectorXd> update(update_, 6);
        _estimate += update;
    }

    virtual bool setEstimateDataImpl(const double *est)
    {
        Eigen::Map<const Eigen::VectorXd> _est(est, 6);
        _estimate = _est;
        return true;
    }

    virtual bool getEstimateData(double *est) const
    {
        Eigen::Map<Eigen::VectorXd> _est(est, 6);
        _est = _estimate;
        return true;
    }

    virtual int estimateDimension() const
    {
        return 6;
    }

    virtual bool setMinimalEstimateDataImpl(const double *est)
    {
        _estimate = Eigen::Map<const Eigen::VectorXd>(est, 6);
        return true;
    }

    virtual bool getMinimalEstimateData(double *est) const
    {
        Eigen::Map<Eigen::VectorXd> v(est, 6);
        v = _estimate;
        return true;
    }

    virtual int minimalEstimateDimension() const
    {
        return 6;
    }

};

class VertexLineEndptsWriteGnuplotAction: public WriteGnuplotAction
{
public:
    VertexLineEndptsWriteGnuplotAction();
    virtual HyperGraphElementAction *operator()(HyperGraph::HyperGraphElement *element, HyperGraphElementAction::Parameters *params_);
};

#ifdef G2O_HAVE_OPENGL
/**
 * \brief visualize a 3D point
 */
class VertexLineEndptsDrawAction: public DrawAction
{
public:
    VertexLineEndptsDrawAction();
    virtual HyperGraphElementAction *operator()(HyperGraph::HyperGraphElement *element,
            HyperGraphElementAction::Parameters *params_);


protected:
    FloatProperty *_pointSize;
    virtual bool refreshPropertyPtrs(HyperGraphElementAction::Parameters *params_);
};
#endif

}
#endif
