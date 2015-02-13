#include "utils.h"
#include "export.h"
#include <fstream>
#include <QDir>
#include <QString>

void exportCamPose(Mfg& m, QString fname) {
   ofstream file(fname.toLocal8Bit().data());
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

void exportMfgNode(Mfg& m, QString fname)
// output mfg nodes to file
{
   ofstream file(fname.toLocal8Bit().data());

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

void Mfg::exportAll (QString root_dir)
{
  //======== 0. make/check directory ========
  if(!QDir(root_dir).exists()) {
    QDir().mkdir(root_dir);
  }
  //======== 1. export 3D features ========
  QString feat3d_fname = root_dir + "/features3d.txt";
  ofstream feat3d_ofs(feat3d_fname.toLocal8Bit().data());

  feat3d_ofs<<"keyframe_number: "<<views.size()<<'\n';
  feat3d_ofs<<"intrinsic_camera_param: "<<K.at<double>(0,0)<<'\t'<<K.at<double>(0,1)<<'\t'<<K.at<double>(0,2)<<'\t'
    <<K.at<double>(1,0)<<'\t'<<K.at<double>(1,1)<<'\t'<<K.at<double>(1,2)<<'\t'
    <<K.at<double>(2,0)<<'\t'<<K.at<double>(2,1)<<'\t'<<K.at<double>(2,2)<<'\n';

  // === point ===
  int kpNum = 0;
  for (int i = 0; i < keyPoints.size(); ++i) {
    if (!keyPoints[i].is3D || keyPoints[i].gid < 0) continue; // only output 3d pt
    ++kpNum;
  }
  feat3d_ofs << "3D_point_number: "<<kpNum<<'\t'<<keyPoints.size()<<'\n';
  feat3d_ofs <<"#format: global_id\tx\ty\tz\tplane_id\test_view_id\tviewId_ptLid_number\tviewId_ptLid_pairs\n";
  for (int i = 0; i < keyPoints.size(); ++i) {
    if (!keyPoints[i].is3D || keyPoints[i].gid < 0) continue; // only output 3d pt
    feat3d_ofs<< keyPoints[i].gid <<'\t' << keyPoints[i].x << '\t' << keyPoints[i].y << '\t' << keyPoints[i].z << '\t'
       << keyPoints[i].pGid <<'\t' << keyPoints[i].estViewId << '\t';
    feat3d_ofs<<keyPoints[i].viewId_ptLid.size()<<'\t';
    for(int j=0; j<keyPoints[i].viewId_ptLid.size(); ++j) {
      feat3d_ofs<<keyPoints[i].viewId_ptLid[j][0]<<'\t'<<keyPoints[i].viewId_ptLid[j][1]<<'\t';
    }
    feat3d_ofs<<'\n';
  }

    // === lines ===
  int ilNum = 0;
  for (int i = 0; i < idealLines.size(); ++i) {
    if (!idealLines[i].is3D || idealLines[i].gid < 0) continue; // only output 3d pt
    ++ilNum;
  }
  feat3d_ofs <<"3D_line_number: "<< ilNum<<'\t'<<idealLines.size() <<'\n';
  feat3d_ofs <<"#format: global_id\tx1\ty1\tz1\tx2\ty2\tz2\tplane_id\tvp_id\test_view_id\tviewId_lnLid_number\tviewId_lnLid_pairs\n";
  for (int i = 0; i < idealLines.size(); ++i) {
    if (!idealLines[i].is3D || idealLines[i].gid < 0) continue; // only output 3d pt
    feat3d_ofs<< idealLines[i].gid <<'\t' << idealLines[i].extremity1().x << '\t' << idealLines[i].extremity1().y << '\t' << idealLines[i].extremity1().z << '\t'
        << idealLines[i].extremity2().x <<'\t'<< idealLines[i].extremity2().y <<'\t'<< idealLines[i].extremity2().z <<'\t'
        << idealLines[i].pGid<<'\t'<< idealLines[i].vpGid << '\t'<< idealLines[i].estViewId << '\t';
    feat3d_ofs<<idealLines[i].viewId_lnLid.size()<<'\t';
    for(int j=0; j<idealLines[i].viewId_lnLid.size(); ++j) {
      feat3d_ofs<<idealLines[i].viewId_lnLid[j][0]<<'\t'<<idealLines[i].viewId_lnLid[j][1]<<'\t';
    }
    feat3d_ofs<<'\n';
  }

  // === vanishing points ===
  feat3d_ofs <<"3D_vanishing_point_number: "<< vanishingPoints.size() <<'\n';
  feat3d_ofs <<"#format: global_id\tx\ty\tz\tw\test_view_id\tviewId_vpLid_number\tviewId_vpLid_pairs\n";
  for (int i = 0; i < vanishingPoints.size(); ++i) {
    feat3d_ofs << vanishingPoints[i].gid <<'\t'
      << vanishingPoints[i].x << '\t' << vanishingPoints[i].y << '\t' << vanishingPoints[i].z << '\t'<<vanishingPoints[i].w<<'\t'
      << vanishingPoints[i].estViewId<<'\t';
    feat3d_ofs<<vanishingPoints[i].viewId_vpLid.size()<<'\t';
    for(int j=0; j<vanishingPoints[i].viewId_vpLid.size(); ++j) {
      feat3d_ofs<<vanishingPoints[i].viewId_vpLid[j][0]<<'\t'<<vanishingPoints[i].viewId_vpLid[j][1]<<'\t';
    }
    feat3d_ofs<<'\n';
  }

  // === planes ===
  feat3d_ofs<<"3D_plane_number: "<<primaryPlanes.size()<<'\n';
  feat3d_ofs<<"#format:global_id\tnormal_vec_3d\tdepth\test_view_id\trecent_view_id\tchild_kpt_number\tchild_kpt_gids\tchild_line_number\tchild_line_gids\n";
  for(int i=0; i<primaryPlanes.size();++i) {
    feat3d_ofs<<primaryPlanes[i].gid<<'\t'
      <<primaryPlanes[i].n.at<double>(0)<<'\t'<<primaryPlanes[i].n.at<double>(1)<<'\t'<<primaryPlanes[i].n.at<double>(2)<<'\t'
      <<primaryPlanes[i].d<<'\t'<<primaryPlanes[i].estViewId<<'\t'<<primaryPlanes[i].recentViewId<<'\t';
    feat3d_ofs<<primaryPlanes[i].kptGids.size()<<'\t';
    for(int j=0; j<primaryPlanes[i].kptGids.size();++j) feat3d_ofs<<primaryPlanes[i].kptGids[j]<<'\t';
    feat3d_ofs<<primaryPlanes[i].ilnGids.size()<<'\t';
    for(int j=0; j<primaryPlanes[i].ilnGids.size();++j) feat3d_ofs<<primaryPlanes[i].ilnGids[j]<<'\t';
    feat3d_ofs<<'\n';
  }

  //======== 2. export views =========
  QString views_dir = root_dir + "/views";
  if(QDir(views_dir).exists()) {
    QDir(views_dir).removeRecursively();
  }
  QDir().mkdir(views_dir);

  for(int i=0; i<views.size(); ++i) {
    QString view_fname = views_dir + QString("/views_%1.txt").arg(i, 3, 10, QChar('0'));
    ofstream view_ofs(view_fname.toLocal8Bit().data());
    view_ofs<<"keyframe_id:\t"<<views[i].id<<'\n';
    view_ofs<<"rawframe_id:\t"<<views[i].frameId<<'\n';
    view_ofs<<"image_path:\t"<<views[i].filename<<'\n';
    view_ofs<<"rotation_mat:\t"<<views[i].R.at<double>(0,0)<<'\t'<<views[i].R.at<double>(0,1)<<'\t'<<views[i].R.at<double>(0,2)<<'\t'
        <<views[i].R.at<double>(1,0)<<'\t'<<views[i].R.at<double>(1,1)<<'\t'<<views[i].R.at<double>(1,2)<<'\t'
        <<views[i].R.at<double>(2,0)<<'\t'<<views[i].R.at<double>(2,1)<<'\t'<<views[i].R.at<double>(2,2)<<'\n';
    view_ofs<<"translation:\t"<<views[i].t.at<double>(0)<<'\t'<<views[i].t.at<double>(1)<<'\t'<<views[i].t.at<double>(2)<<'\n';

    // === 2d points ===
    view_ofs<<"feat_pts_number: "<<views[i].featurePoints.size()<<'\n';
    view_ofs<<"#format: local_id\tglobal_id\tx\ty\tdesc_dim\tdescriptor_vector\n";
    for(int j=0; j<views[i].featurePoints.size();++j) {
      view_ofs<<views[i].featurePoints[j].lid<<'\t'<<views[i].featurePoints[j].gid<<'\t'<<views[i].featurePoints[j].x<<'\t'<<views[i].featurePoints[j].y<<'\t';
      int desc_dim =  views[i].featurePoints[j].siftDesc.rows*views[i].featurePoints[j].siftDesc.cols;
      view_ofs<<desc_dim<<'\t';
      if(views[i].featurePoints[j].siftDesc.type()==CV_32FC1)
        for(int k=0; k<desc_dim; ++k) view_ofs<<views[i].featurePoints[j].siftDesc.at<float>(k)<<'\t';
      else if (views[i].featurePoints[j].siftDesc.type()==CV_64FC1)
        for(int k=0; k<desc_dim; ++k) view_ofs<<views[i].featurePoints[j].siftDesc.at<double>(k)<<'\t';
      else {
        cout<<"pt descriptor type error in exportAll\n"; exit(0);
      }
      view_ofs<<'\n';
    }

    // === line segments ===
    view_ofs<<"line_segment_number: "<<views[i].lineSegments.size()<<'\n';
    view_ofs<<"format: local_id\tx1\ty1\tx2\ty2\tvp_lid\tideal_line_lid\tgx\tgy\tdesc_dim\tmsld_desc\n";
    for(int j=0; j<views[i].lineSegments.size(); ++j) {
      view_ofs<<views[i].lineSegments[j].lid<<'\t'<<views[i].lineSegments[j].endpt1.x<<'\t'<<views[i].lineSegments[j].endpt1.y<<'\t'
        <<views[i].lineSegments[j].endpt2.x<<'\t'<<views[i].lineSegments[j].endpt2.y<<'\t'
        <<views[i].lineSegments[j].vpLid<<'\t'<<views[i].lineSegments[j].idlnLid<<'\t'
        <<views[i].lineSegments[j].gradient.x<<'\t'<<views[i].lineSegments[j].gradient.y<<'\t';
      int desc_dim = views[i].lineSegments[j].msldDesc.rows * views[i].lineSegments[j].msldDesc.cols;
      view_ofs<<desc_dim<<'\t';
      if(views[i].lineSegments[j].msldDesc.type()==CV_32FC1)
        for(int k=0; k<desc_dim;++k) view_ofs<<views[i].lineSegments[j].msldDesc.at<float>(k)<<'\t';
      else if(views[i].lineSegments[j].msldDesc.type()==CV_64FC1)
        for(int k=0; k<desc_dim;++k) view_ofs<<views[i].lineSegments[j].msldDesc.at<double>(k)<<'\t';
      else {
        cout<<"line descriptor type error in exportAll\n"; exit(0);
      }
      view_ofs<<'\n';
    }

    // === ideal lines ===
    view_ofs<<"ideal_line_number: "<<views[i].idealLines.size()<<'\n';
    view_ofs<<"format: local_id\tglobal_id\tvp_lid\tplane_gid\tx1\ty1\tx2\ty2\tgx\tgy\tchild_segment_number\tchild_segment_lids\n";
    for(int j=0; j<views[i].idealLines.size();++j) {
      view_ofs<<views[i].idealLines[j].lid<<'\t'<<views[i].idealLines[j].gid<<'\t'<<views[i].idealLines[j].vpLid<<'\t'<<views[i].idealLines[j].pGid<<'\t'
        <<views[i].idealLines[j].extremity1.x<<'\t'<<views[i].idealLines[j].extremity1.y<<'\t'<<views[i].idealLines[j].extremity2.x<<'\t'<<views[i].idealLines[j].extremity2.y<<'\t'
        <<views[i].idealLines[j].gradient.x<<'\t'<<views[i].idealLines[j].gradient.y<<'\t';
      view_ofs<<views[i].idealLines[j].lsLids.size()<<'\t';
      for(int k=0; k<views[i].idealLines[j].lsLids.size();++k) view_ofs<<views[i].idealLines[j].lsLids[k]<<'\t';
      view_ofs<<'\n';
    }

    // === vanishing points ===
    view_ofs<<"vanishing_points_number: "<<views[i].vanishPoints.size()<<'\n';
    view_ofs<<"format: local_id\tglobal_id\tx\ty\tw\tcov_2x2\tchild_iline_number\tchild_iline_lids\n";
    for(int j=0; j<views[i].vanishPoints.size();++j) {
      view_ofs<<views[i].vanishPoints[j].lid<<'\t'<<views[i].vanishPoints[j].gid<<'\t'
        <<views[i].vanishPoints[j].x<<'\t'<<views[i].vanishPoints[j].y<<'\t'<<views[i].vanishPoints[j].w<<'\t'
        <<views[i].vanishPoints[j].cov_ab.at<double>(0,0)<<'\t'<<views[i].vanishPoints[j].cov_ab.at<double>(0,1)<<'\t'
        <<views[i].vanishPoints[j].cov_ab.at<double>(1,0)<<'\t'<<views[i].vanishPoints[j].cov_ab.at<double>(1,1)<<'\t';
      view_ofs<<views[i].vanishPoints[j].idlnLids.size()<<'\t';
      for(int k=0; k<views[i].vanishPoints[j].idlnLids.size();++k) view_ofs<<views[i].vanishPoints[j].idlnLids[k]<<'\t';
      view_ofs<<'\n';
    }

    // === misc ===
    view_ofs<<"angular_velocity: "<<views[i].angVel<<" deg_per_sec\n";
    view_ofs<<"line_segment2d_len_thresh: "<<views[i].lsLenThresh<<'\n';
  }
  cout<<"MFG exported.\n";
}
