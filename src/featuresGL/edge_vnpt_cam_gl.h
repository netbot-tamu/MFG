#ifndef G2O_EDGE_VNTP_CAM_GL_H_
#define G2O_EDGE_VNTP_CAM_GL_H_

#include "g2o/core/base_binary_edge.h"
#include "vertex_vnpt.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"

namespace g2o {

#ifdef G2O_HAVE_OPENGL
  class EdgeVnptCamDrawAction: public DrawAction{
  public:
    EdgeVnptCamDrawAction();
    virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element,
            HyperGraphElementAction::Parameters* params_);
  };
#endif

}
#endif
