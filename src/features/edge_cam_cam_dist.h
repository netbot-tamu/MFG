#ifndef G2O_EDGE_CAM_CAM_DIST_H_
#define G2O_EDGE_CAM_CAM_DIST_H_

#include "g2o/core/base_binary_edge.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"

namespace g2o {
  // first two args are the measurement type, second two the connection classes
  class EdgeCamCamDist : public BaseBinaryEdge<1, double, VertexCam, VertexCam> {
  public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeCamCamDist();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    // return the error estimate as a 2-vector
    void computeError();
    // jacobian
 //   virtual void linearizeOplus();

    virtual void setMeasurement(const double m){
      _measurement = m;
    }

    virtual bool setMeasurementData(const double* d){      
      _measurement = d[0];
      return true;
    }

    virtual bool getMeasurementData(double* d) const{
      d[0] =_measurement;
      return true;
    }

    virtual int measurementDimension() const {return 1;}

    virtual double initialEstimatePossible(const OptimizableGraph::VertexSet& from,
             OptimizableGraph::Vertex* to) {
      (void) to;
      return (from.count(_vertices[0]) == 1 ? 1.0 : -1.0);
    }

  virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to) {}


  private:
    Eigen::Matrix<double,1,6+6> J; // jacobian before projection ????????
  };


}
#endif
