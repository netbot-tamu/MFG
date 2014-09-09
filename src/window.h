#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include "mfg.h"

class QSlider;

class GLWidget;

class Window : public QWidget
{
	Q_OBJECT

public:
	Window();
	void setMfgScene(TwoView* pMfg);
	void setMfgScene(Mfg* pMap);

protected:
	void keyPressEvent(QKeyEvent *event);

private:
	QSlider *createSlider();
	QSlider *createScaleSlider();

	GLWidget *glWidget;
	QSlider *xSlider;
	QSlider *ySlider;
	QSlider *zSlider;
	QSlider *sSlider; // scale
};

#endif
