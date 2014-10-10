#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QtOpenGL/QGLWidget>
#include "mfg.h"

// class QtLogo;

class GLWidget : public QGLWidget
{
   Q_OBJECT

public:
   GLWidget(QWidget *parent = 0);
   ~GLWidget();

   QSize minimumSizeHint() const;
   QSize sizeHint() const;

   TwoView *scene;
   Mfg*	  map;	

public slots:
   void setXRotation(int angle);
   void setYRotation(int angle);
   void setZRotation(int angle);
   void setScale(int s);

signals:
   void xRotationChanged(int angle);
   void yRotationChanged(int angle);
   void zRotationChanged(int angle);
   void scaleChanged(int s);

protected:
   void initializeGL();
   void paintGL();
   void resizeGL(int width, int height);
   void mousePressEvent(QMouseEvent *event);
   void mouseMoveEvent(QMouseEvent *event);

private:

   int xRot;
   int yRot;
   int zRot;
   int scale;
   QPoint lastPos;
   QColor qtGreen;
   QColor qtPurple;
   QColor qtWhite;
};

#endif
