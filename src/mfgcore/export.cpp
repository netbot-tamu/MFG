
#include "export.h"
#include <fstream>

void exportCamPose(Mfg& m, string fname) {
   ofstream file(fname);
   // ----- camera poses -----
   for (int i = 0; i < m.views.size(); ++i) {
      file << m.views[i].id << '\t' << m.views[i].frameId << '\t'
              << m.views[i].R.at<double>(0, 0) << '\t'
              << m.views[i].R.at<double>(0, 1) << '\t'
              << m.views[i].R.at<double>(0, 2) << '\t'
              << m.views[i].R.at<double>(1, 0) << '\t'
              << m.views[i].R.at<double>(1, 1) << '\t'
              << m.views[i].R.at<double>(1, 2) << '\t'
              << m.views[i].R.at<double>(2, 0) << '\t'
              << m.views[i].R.at<double>(2, 1) << '\t'
              << m.views[i].R.at<double>(2, 2) << '\t'
              << m.views[i].t.at<double>(0) << '\t'
              << m.views[i].t.at<double>(1) << '\t'
              << m.views[i].t.at<double>(2) << '\t'
              << m.views[i].errPt << '\t'
              //   << m.views[i].errVp << '\t'
              << m.views[i].errLn << '\t'
              << m.views[i].errAll << '\t'
              << m.views[i].errPl << '\t'
              << m.views[i].errLnMean << '\n';
   }
  
   file.close();
}

void exportMfgNode(Mfg& m, string fname)
// output mfg nodes to file
{
   ofstream file(fname);

   // point
   int kpNum = 0;
   for (int i = 0; i < m.keyPoints.size(); ++i) {
      if (!m.keyPoints[i].is3D || m.keyPoints[i].gid < 0) continue; // only output 3d pt
      ++kpNum;
   }
   file << kpNum << '\n';
   for (int i = 0; i < m.keyPoints.size(); ++i) {
      if (!m.keyPoints[i].is3D || m.keyPoints[i].gid < 0) continue; // only output 3d pt
      file << m.keyPoints[i].x << '\t' << m.keyPoints[i].y << '\t' << m.keyPoints[i].z
              << m.keyPoints[i].gid << '\t' << m.keyPoints[i].estViewId << '\t'
              << m.keyPoints[i].pGid << '\n';
   }

   // ideal line
   int ilNum = 0;
   for (int i = 0; i < m.idealLines.size(); ++i) {
      if (!m.idealLines[i].is3D || m.idealLines[i].gid < 0) continue; // only output 3d
      ++ilNum;
   }
   file << ilNum << '\n';
   for (int i = 0; i < m.idealLines.size(); ++i) {
      if (!m.idealLines[i].is3D || m.idealLines[i].gid < 0) continue; // only output 3d
      file << m.idealLines[i].extremity1().x << '\t' << m.idealLines[i].extremity1().y << '\t'
              << m.idealLines[i].extremity1().z << '\t'
              << m.idealLines[i].extremity2().x << '\t' << m.idealLines[i].extremity2().y << '\t'
              << m.idealLines[i].extremity2().z << '\t'
              << m.idealLines[i].gid << '\t' << m.idealLines[i].estViewId << '\t'
              << m.idealLines[i].vpGid << '\t' << m.idealLines[i].pGid << '\n';
   }

   file << m.vanishingPoints.size() << '\n';
   for (int i = 0; i < m.vanishingPoints.size(); ++i) {
      file << m.vanishingPoints[i].x << '\t' << m.vanishingPoints[i].y << '\t'
              << m.vanishingPoints[i].z << '\t' << m.vanishingPoints[i].gid << '\t'
              << m.vanishingPoints[i].estViewId << '\n';
   }
   file.close();
}
