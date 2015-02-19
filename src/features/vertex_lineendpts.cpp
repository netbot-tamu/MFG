#include "vertex_lineendpts.h"
#include <stdio.h>

#include <Eigen/SparseCore>
#include <typeinfo>

namespace g2o {

   bool VertexLineEndpts::read(std::istream& is) {
      Eigen::VectorXd lv;
      for (int i=0; i<estimateDimension(); i++)
         is >> lv[i];
      setEstimate(lv);
      return true;
   }

   bool VertexLineEndpts::write(std::ostream& os) const {
      Eigen::VectorXd lv=estimate();
      for (int i=0; i<estimateDimension(); i++){
         os << lv[i] << " ";
      }
      return os.good();
   }

   VertexLineEndptsWriteGnuplotAction:: VertexLineEndptsWriteGnuplotAction() :
   WriteGnuplotAction(typeid( VertexLineEndpts).name())
   {
   }

   HyperGraphElementAction* VertexLineEndptsWriteGnuplotAction::operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params_ )
   {
      if (typeid(*element).name()!=_typeName)
         return 0;
      WriteGnuplotAction::Parameters* params=static_cast<WriteGnuplotAction::Parameters*>(params_);
      if (!params->os){
         std::cerr << __PRETTY_FUNCTION__ << ": warning, no valid os specified" << std::endl;
         return 0;
      }

      VertexLineEndpts* v = static_cast<VertexLineEndpts*>(element);
      *(params->os) << v->estimate().x() << " " << v->estimate().y() << " " << v->estimate().z() << " " << std::endl;
      return this;
   }

}
