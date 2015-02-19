
#ifndef G2O_VERTEX_LINE6D_GL_H_
#define G2O_VERTEX_LINE6D_GL_H_

#include "g2o/types/slam3d/g2o_types_slam3d_api.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"
#include <Eigen/SparseCore>

namespace g2o {

#ifdef G2O_HAVE_OPENGL
  /**
   * \brief visualize a 3D point
   */
  class VertexLineEndptsDrawAction: public DrawAction{
    public:
      VertexLineEndptsDrawAction();
      virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element,
          HyperGraphElementAction::Parameters* params_);


    protected:
      FloatProperty *_pointSize;
      virtual bool refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_);
  };
#endif

}
#endif
