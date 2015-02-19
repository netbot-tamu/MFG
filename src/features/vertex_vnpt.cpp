#include "vertex_vnpt.h"
#include <stdio.h>

#include <Eigen/SparseCore>
#include <typeinfo>

namespace g2o {

   bool VertexVanishPoint::read(std::istream& is) {
      Eigen::Vector3d lv;
      for (int i=0; i<estimateDimension(); i++)
         is >> lv[i];
      setEstimate(lv);
      return true;
   }

   bool VertexVanishPoint::write(std::ostream& os) const {
      Eigen::Vector3d lv=estimate();
      for (int i=0; i<estimateDimension(); i++){
         os << lv[i] << " ";
      }
      return os.good();
   }

   VertexVanishPointWriteGnuplotAction:: VertexVanishPointWriteGnuplotAction() :
   WriteGnuplotAction(typeid( VertexVanishPoint).name())
   {
   }

   HyperGraphElementAction* VertexVanishPointWriteGnuplotAction::operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params_ )
   {
      if (typeid(*element).name()!=_typeName)
         return 0;
      WriteGnuplotAction::Parameters* params=static_cast<WriteGnuplotAction::Parameters*>(params_);
      if (!params->os){
         std::cerr << __PRETTY_FUNCTION__ << ": warning, no valid os specified" << std::endl;
         return 0;
      }

      VertexVanishPoint* v = static_cast<VertexVanishPoint*>(element);
      *(params->os) << v->estimate().x() << " " << v->estimate().y() << " " << v->estimate().z() << " " << std::endl;
      return this;
   }

}
