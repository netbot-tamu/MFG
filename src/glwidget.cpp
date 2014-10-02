#include <QtGui>

#include <math.h>

#include "glwidget.h"


 #ifndef GL_MULTISAMPLE
 #define GL_MULTISAMPLE  0x809D
 #endif


double scale_const = 2000;

 GLWidget::GLWidget(QWidget *parent)
     : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
 {
	 scene = 0;
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
#ifndef HIGH_SPEED_NO_GRAPHICS
	 if (scene)
		 scene->draw3D();
	 if (map) {
		 map->draw3D();
	 }
#endif
//     glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
//     glEnable ( GL_COLOR_MATERIAL );

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
