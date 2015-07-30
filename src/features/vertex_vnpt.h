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
 * G2O vertex representing a 3D vanishing point
 ********************************************************************************/


#ifndef G2O_VERTEX_VANISHPOINT3D_H_
#define G2O_VERTEX_VANISHPOINT3D_H_

#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"

namespace g2o
{
/**
 * \brief Vertex for a tracked point in space
 */

class VertexVanishPoint : public VertexPointXYZ
{
public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexVanishPoint() {}
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    virtual void setToOriginImpl()
    {
        _estimate.fill(0.);
    }

    virtual void oplusImpl(const double *update_)
    {
        Eigen::Map<const Eigen::Vector3d> update(update_, 3);
        _estimate += update;
    }

    virtual bool setEstimateDataImpl(const double *est)
    {
        Eigen::Map<const Eigen::Vector3d> _est(est, 3);
        _estimate = _est;
        return true;
    }

    virtual bool getEstimateData(double *est) const
    {
        Eigen::Map<Eigen::Vector3d> _est(est, 3);
        _est = _estimate;
        return true;
    }

    virtual int estimateDimension() const
    {
        return 3;
    }

    virtual bool setMinimalEstimateDataImpl(const double *est)
    {
        _estimate = Eigen::Map<const Eigen::Vector3d>(est, 3);
        return true;
    }

    virtual bool getMinimalEstimateData(double *est) const
    {
        Eigen::Map<Eigen::Vector3d> v(est, 3);
        v = _estimate;
        return true;
    }

    virtual int minimalEstimateDimension() const
    {
        return 3;
    }

};

class VertexVanishPointWriteGnuplotAction: public WriteGnuplotAction
{
public:
    VertexVanishPointWriteGnuplotAction();
    virtual HyperGraphElementAction *operator()(HyperGraph::HyperGraphElement *element, HyperGraphElementAction::Parameters *params_);
};

#ifdef G2O_HAVE_OPENGL
class VertexVanishPointDrawAction: public DrawAction
{
public:
    VertexVanishPointDrawAction();
    virtual HyperGraphElementAction *operator()(HyperGraph::HyperGraphElement *element,
            HyperGraphElementAction::Parameters *params_);


protected:
    FloatProperty *_pointSize;
    virtual bool refreshPropertyPtrs(HyperGraphElementAction::Parameters *params_);
};
#endif

}
#endif
