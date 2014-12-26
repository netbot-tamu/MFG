#include "window.h"

//#include <QtGui>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QSlider>
#include <QWidget>

#include "glwidget.h"


extern double scale_const;

Window::Window()
{
   glWidget = new GLWidget;

   xSlider = createSlider();
   ySlider = createSlider();
   zSlider = createSlider();
   sSlider = createScaleSlider();

   connect(xSlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setXRotation(int)));
   connect(glWidget, SIGNAL(xRotationChanged(int)), xSlider, SLOT(setValue(int)));
   connect(ySlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setYRotation(int)));
   connect(glWidget, SIGNAL(yRotationChanged(int)), ySlider, SLOT(setValue(int)));
   connect(zSlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setZRotation(int)));
   connect(glWidget, SIGNAL(zRotationChanged(int)), zSlider, SLOT(setValue(int)));
   connect(sSlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setScale(int)));
   connect(glWidget, SIGNAL(scaleChanged(int)), sSlider, SLOT(setValue(int)));

   QHBoxLayout *mainLayout = new QHBoxLayout;
   mainLayout->addWidget(glWidget);
   mainLayout->addWidget(xSlider);
   mainLayout->addWidget(ySlider);
   mainLayout->addWidget(zSlider);
   mainLayout->addWidget(sSlider);
   setLayout(mainLayout);

   xSlider->setValue(270 * 16);
   ySlider->setValue(0 * 16);
   zSlider->setValue(0 * 16);
   sSlider->setValue(0.1*scale_const);
   setWindowTitle(tr("Multilayer Feature Graph - 3D"));
}

QSlider *Window::createSlider()
{
   QSlider *slider = new QSlider(Qt::Vertical);
   slider->setRange(0, 360 * 16);
   slider->setSingleStep(16);
   slider->setPageStep(15 * 16);
   slider->setTickInterval(15 * 16);
   slider->setTickPosition(QSlider::TicksRight);
   return slider;
}

QSlider *Window::createScaleSlider()
{
   QSlider *slider = new QSlider(Qt::Vertical);
   slider->setRange(0, 0.1*scale_const);
   slider->setSingleStep(1);
   slider->setPageStep(100);
   slider->setTickInterval(20);
   slider->setTickPosition(QSlider::TicksRight);
   return slider;
}

void Window::keyPressEvent(QKeyEvent *e)
{
   if (e->key() == Qt::Key_Escape)
      close();
   else
      QWidget::keyPressEvent(e);
}

void Window::setMfgScene (Mfg* pMap)
{
   glWidget->map = pMap;
}
