
#ifndef G2O_VERTEX_PLANE3D_H_
#define G2O_VERTEX_PLANE3D_H_

#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"

#include <Eigen/SparseCore>

namespace g2o {
	class VertexPlane3d : public VertexPointXYZ
  {// plane = n/d, 3-vector, representing [n, d] plane
    public:
 //     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
      VertexPlane3d() {}
      virtual bool read(std::istream& is);
      virtual bool write(std::ostream& os) const;

      virtual void setToOriginImpl() { _estimate.fill(0.); }

      virtual void oplusImpl(const double* update_) {
        Eigen::Map<const Eigen::Vector3d> update(update_, 3);
        _estimate += update;
      }

      virtual bool setEstimateDataImpl(const double* est){
        Eigen::Map<const Eigen::Vector3d> _est(est, 3);
        _estimate = _est;
        return true;
      }

      virtual bool getEstimateData(double* est) const{
        Eigen::Map<Eigen::Vector3d> _est(est, 3);
        _est = _estimate;
        return true;
      }

      virtual int estimateDimension() const {
        return 3;
      }

      virtual bool setMinimalEstimateDataImpl(const double* est){
        _estimate = Eigen::Map<const Eigen::Vector3d>(est, 3);
        return true;
      }

      virtual bool getMinimalEstimateData(double* est) const{
        Eigen::Map<Eigen::Vector3d> v(est, 3);
        v = _estimate;
        return true;
      }

      virtual int minimalEstimateDimension() const {
        return 3;
      }

  };

  class VertexPlane3dWriteGnuplotAction: public WriteGnuplotAction
  {
    public:
      VertexPlane3dWriteGnuplotAction();
      virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params_ );
  };

#ifdef G2O_HAVE_OPENGL
  class VertexPlane3dDrawAction: public DrawAction{
    public:
      VertexPlane3dDrawAction();
      virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, 
          HyperGraphElementAction::Parameters* params_);


    protected:
      FloatProperty *_pointSize;
      virtual bool refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_);
  };
#endif

}
#endif
