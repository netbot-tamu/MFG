#ifndef MFGWIN_H
#define MFGWIN_H

#include <QtGui/QWidget>
#include "ui_mfgwin.h"

class mfgWin : public QWidget
{
	Q_OBJECT

public:
	mfgWin(QWidget *parent = 0, Qt::WFlags flags = 0);
	~mfgWin();

private:
	Ui::mfgWinClass ui;
};

#endif // MFGWIN_H
