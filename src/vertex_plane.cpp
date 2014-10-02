#include "vertex_plane.h"
#include <stdio.h>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <Eigen/SparseCore>
#include <typeinfo>

namespace g2o {

  bool VertexPlane3d::read(std::istream& is) {
    Eigen::Vector3d lv;
    for (int i=0; i<estimateDimension(); i++)
      is >> lv[i];
    setEstimate(lv);
    return true;
  }

  bool VertexPlane3d::write(std::ostream& os) const {
    Eigen::Vector3d lv=estimate();
    for (int i=0; i<estimateDimension(); i++){
      os << lv[i] << " ";
    }
    return os.good();
  }
  

#ifdef G2O_HAVE_OPENGL
   VertexPlane3dDrawAction::VertexPlane3dDrawAction(): DrawAction(typeid(VertexPlane3d).name()){
  }

  bool VertexPlane3dDrawAction::refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_){
    if (! DrawAction::refreshPropertyPtrs(params_))
      return false;
    if (_previousParams){
      _pointSize = _previousParams->makeProperty<FloatProperty>(_typeName + "::POINT_SIZE", 1.);
    } else {
      _pointSize = 0;
    }
    return true;
  }


  HyperGraphElementAction* VertexPlane3dDrawAction::operator()(HyperGraph::HyperGraphElement* element, 
                     HyperGraphElementAction::Parameters* params ){

    if (typeid(*element).name()!=_typeName)
      return 0;
    refreshPropertyPtrs(params);
    if (! _previousParams)
      return this;
    
    if (_show && !_show->value())
      return this;
    VertexPlane3d* that = static_cast<VertexPlane3d*>(element);
    

    glPushAttrib(GL_ENABLE_BIT | GL_POINT_BIT);
    glDisable(GL_LIGHTING);
    glColor3f(0.8f,0.5f,0.3f);
    if (_pointSize) {
      glPointSize(_pointSize->value());
    }
//    glBegin(GL_POINTS);
	glBegin(GL_LINES);
    glVertex3f((float)that->estimate()(0),(float)that->estimate()(1),(float)that->estimate()(2));
    glEnd();
    glPopAttrib();

    return this;
  }
#endif

   VertexPlane3dWriteGnuplotAction:: VertexPlane3dWriteGnuplotAction() :
    WriteGnuplotAction(typeid( VertexPlane3d).name())
  {
  }

  HyperGraphElementAction* VertexPlane3dWriteGnuplotAction::operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params_ )
  {
    if (typeid(*element).name()!=_typeName)
      return 0;
    WriteGnuplotAction::Parameters* params=static_cast<WriteGnuplotAction::Parameters*>(params_);
    if (!params->os){
      std::cerr << __PRETTY_FUNCTION__ << ": warning, no valid os specified" << std::endl;
      return 0;
    }

    VertexPlane3d* v = static_cast<VertexPlane3d*>(element);
    *(params->os) << v->estimate().x() << " " << v->estimate().y() << " " << v->estimate().z() << " " << std::endl;
    return this;
  }
  
}
