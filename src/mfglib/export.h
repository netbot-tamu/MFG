/* 
 * File:   export.h
 * Author: madtreat
 *
 * Created on November 7, 2014, 10:29 AM
 */

#ifndef EXPORT_H
#define EXPORT_H

#include "mfg.h"

void exportCamPose(Mfg& m, std::string fname);
void exportMfgNode(Mfg& m, std::string fname);

#endif /* EXPORT_H */

