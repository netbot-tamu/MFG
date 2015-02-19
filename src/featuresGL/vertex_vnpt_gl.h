
#ifndef G2O_VERTEX_VANISHPOINT3D_GL_H_
#define G2O_VERTEX_VANISHPOINT3D_GL_H_

#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"

namespace g2o {
   
#ifdef G2O_HAVE_OPENGL
   class VertexVanishPointDrawAction: public DrawAction{
   public:
      VertexVanishPointDrawAction();
      virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element,
      HyperGraphElementAction::Parameters* params_);


   protected:
      FloatProperty *_pointSize;
      virtual bool refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_);
   };
#endif

}
#endif
