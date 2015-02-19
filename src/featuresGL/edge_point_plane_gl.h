#ifndef G2O_EDGE_POINT_PLANE_GL_H_
#define G2O_EDGE_POINT_PLANE_GL_H_

#include "g2o/core/base_binary_edge.h"
#include "vertex_plane.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"

namespace g2o {

#ifdef G2O_HAVE_OPENGL
  class EdgePointPlane3dDrawAction: public DrawAction{
  public:
    EdgePointPlane3dDrawAction();
    virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element,
            HyperGraphElementAction::Parameters* params_);
  };
#endif

}
#endif
