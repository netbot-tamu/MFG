#include "vertex_lineendpts_gl.h"
#include "vertex_lineendpts.h"
#include <stdio.h>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <Eigen/SparseCore>
#include <typeinfo>

namespace g2o {

#ifdef G2O_HAVE_OPENGL
   VertexLineEndptsDrawAction::VertexLineEndptsDrawAction(): DrawAction(typeid(VertexLineEndpts).name()){
  }

  bool VertexLineEndptsDrawAction::refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_){
    if (! DrawAction::refreshPropertyPtrs(params_))
      return false;
    if (_previousParams){
      _pointSize = _previousParams->makeProperty<FloatProperty>(_typeName + "::POINT_SIZE", 1.);
    } else {
      _pointSize = 0;
    }
    return true;
  }


  HyperGraphElementAction* VertexLineEndptsDrawAction::operator()(HyperGraph::HyperGraphElement* element,
                     HyperGraphElementAction::Parameters* params ){

    if (typeid(*element).name()!=_typeName)
      return 0;
    refreshPropertyPtrs(params);
    if (! _previousParams)
      return this;

    if (_show && !_show->value())
      return this;
    VertexLineEndpts* that = static_cast<VertexLineEndpts*>(element);


    glPushAttrib(GL_ENABLE_BIT | GL_POINT_BIT);
    glDisable(GL_LIGHTING);
    glColor3f(0.8f,0.5f,0.3f);
    if (_pointSize) {
      glPointSize(_pointSize->value());
    }
//    glBegin(GL_POINTS);
	glBegin(GL_LINES);
    glVertex3f((float)that->estimate()(0),(float)that->estimate()(1),(float)that->estimate()(2));
	glVertex3f((float)that->estimate()(3),(float)that->estimate()(4),(float)that->estimate()(5));
    glEnd();
    glPopAttrib();

    return this;
  }
#endif

}
