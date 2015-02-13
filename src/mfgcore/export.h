/*
 * File:   export.h
 * Author: madtreat
 *
 * Created on November 7, 2014, 10:29 AM
 */

#ifndef EXPORT_H
#define EXPORT_H

#include <QString>
#include "mfg.h"

void exportCamPose(Mfg& m, QString fname);
void exportMfgNode(Mfg& m, QString fname);

#endif /* EXPORT_H */

