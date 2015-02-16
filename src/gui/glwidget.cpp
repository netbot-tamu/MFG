#include "glwidget.h"

#include <QtGui>
#include <math.h>


#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif


double scale_const = 2000;

GLWidget::GLWidget(QWidget *parent)
: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
   xRot = 270*16;
   yRot = 0;
   zRot = 0;
   scale = 0.1*scale_const;// it has to equal that in sSlider->setValue(1*100);

   qtGreen = QColor::fromCmykF(0.40, 0.0, 1.0, 0.0);
   //   qtPurple = QColor::fromCmykF(0.39, 0.39, 0.0, 0.0);
   qtPurple = QColor::fromRgb(255, 255, 255);
}

GLWidget::~GLWidget()
{
}

QSize GLWidget::minimumSizeHint() const
{
   return QSize(50, 50);
}

QSize GLWidget::sizeHint() const
{
   return QSize(600, 600);
}

static void qNormalizeAngle(int &angle)
{
   while (angle < 0)
      angle += 360 * 16;
   while (angle > 360 * 16)
      angle -= 360 * 16;
}

void GLWidget::setXRotation(int angle)
{
   qNormalizeAngle(angle);
   if (angle != xRot) {
      xRot = angle;
      emit xRotationChanged(angle);
      updateGL();
   }
}

void GLWidget::setYRotation(int angle)
{
   qNormalizeAngle(angle);
   if (angle != yRot) {
      yRot = angle;
      emit yRotationChanged(angle);
      updateGL();
   }
}

void GLWidget::setZRotation(int angle)
{
   qNormalizeAngle(angle);
   if (angle != zRot) {
      zRot = angle;
      //		 cout<<"zRot="<<zRot<<endl;
      emit zRotationChanged(angle);
      updateGL();
   }
}

void GLWidget::setScale(int s)
{
   if (scale != s) {
      scale = s;
      //		 cout<<"scale="<<scale<<endl;
      emit scaleChanged(s);
      updateGL();
   }
}

void GLWidget::initializeGL()
{
   qglClearColor(qtPurple);

   glEnable(GL_DEPTH_TEST);
   //     glEnable(GL_CULL_FACE);
   glShadeModel(GL_SMOOTH);
   //     glEnable(GL_LIGHTING);
   //     glEnable(GL_LIGHT0);
   glEnable(GL_MULTISAMPLE);
   //     static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
   //     glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
}

void GLWidget::paintGL()
{
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glLoadIdentity();
   glTranslatef(0.0, 0.0, -1.0);
   glRotatef(xRot / 16.0, 1.0, 0.0, 0.0);
   glRotatef(yRot / 16.0, 0.0, 1.0, 0.0);
   glRotatef(zRot / 16.0, 0.0, 0.0, 1.0);
   glScalef(scale/scale_const,scale/scale_const,scale/scale_const);

   if (mfg) {
      drawMfg();
   }

}

void GLWidget::resizeGL(int width, int height)
{
   int side = qMin(width, height);
   glViewport((width - side) / 2, (height - side) / 2, side, side);

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
#ifdef QT_OPENGL_ES_1
   glOrthof(-0.5, +0.5, -0.5, +0.5, 4.0, 15.0);
#else
   glOrtho(-1, +1, -1, +1, -0.0, 100.0);
#endif
   glMatrixMode(GL_MODELVIEW);
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
   lastPos = event->pos();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
   int dx = event->x() - lastPos.x();
   int dy = event->y() - lastPos.y();

   if (event->buttons() & Qt::LeftButton) {
      setXRotation(xRot + 8 * dy);
      setYRotation(yRot + 8 * dx);
   } else if (event->buttons() & Qt::RightButton) {
      setXRotation(xRot + 8 * dy);
      setZRotation(zRot + 8 * dx);
   }
   lastPos = event->pos();
}

void GLWidget::drawMfg()
{
   // plot first camera, small
   glLineWidth(1);
   glBegin(GL_LINES);
   glColor3f(1,0,0); // x-axis
   glVertex3f(0,0,0);
   glVertex3f(1,0,0);
   glColor3f(0,1,0);
   glVertex3f(0,0,0);
   glVertex3f(0,1,0);
   glColor3f(0,0,1);// z axis
   glVertex3f(0,0,0);
   glVertex3f(0,0,1);
   glEnd();

   cv::Mat xw = (cv::Mat_<double>(3,1)<< 0.5,0,0),
      yw = (cv::Mat_<double>(3,1)<< 0,0.5,0),
      zw = (cv::Mat_<double>(3,1)<< 0,0,0.5);

   vector<View> views = mfg->views;
   for (int i=1; i<views.size(); ++i) {
      if(!(views[i].R.dims==2)) continue; // handle the in-process view
      cv::Mat c = -views[i].R.t()*views[i].t;
      cv::Mat x_ = views[i].R.t() * (xw-views[i].t),
         y_ = views[i].R.t() * (yw-views[i].t),
         z_ = views[i].R.t() * (zw-views[i].t);
      glBegin(GL_LINES);

      glColor3f(1,0,0);
      glVertex3f(c.at<double>(0),c.at<double>(1),c.at<double>(2));
      glVertex3f(x_.at<double>(0),x_.at<double>(1),x_.at<double>(2));
      glColor3f(0,1,0);
      glVertex3f(c.at<double>(0),c.at<double>(1),c.at<double>(2));
      glVertex3f(y_.at<double>(0),y_.at<double>(1),y_.at<double>(2));
      glColor3f(0,0,1);
      glVertex3f(c.at<double>(0),c.at<double>(1),c.at<double>(2));
      glVertex3f(z_.at<double>(0),z_.at<double>(1),z_.at<double>(2));
      glEnd();
   }

   vector<KeyPoint3d> keyPoints = mfg->keyPoints;
   glPointSize(3.0);
   glBegin(GL_POINTS);
   for (int i=0; i<keyPoints.size(); ++i){
      if(!keyPoints[i].is3D || keyPoints[i].gid<0) continue;
      if(keyPoints[i].pGid < 0) // red
         glColor3f(0.6, 0.6, 0.6);
      else { // coplanar green
         glColor3f(0.0, 1.0, 0.0);
      }
      glVertex3f(keyPoints[i].x, keyPoints[i].y, keyPoints[i].z);
   }
   glEnd();

   vector<IdealLine3d> idealLines = mfg->idealLines;
   glColor3f(0,1,1);
   glLineWidth(2);
   glBegin(GL_LINES);
   for(int i=0; i<idealLines.size(); ++i) {
      if(!idealLines[i].is3D || idealLines[i].gid<0) continue;
      if(idealLines[i].pGid < 0) {
         glColor3f(0,0,0);
      } else {
         glColor3f(0.0,1.0,0.0);
      }

      glVertex3f(idealLines[i].extremity1().x,idealLines[i].extremity1().y,idealLines[i].extremity1().z);
      glVertex3f(idealLines[i].extremity2().x,idealLines[i].extremity2().y,idealLines[i].extremity2().z);
   }
   glEnd();
}
