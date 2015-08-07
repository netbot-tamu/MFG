/////////////////////////////////////////////////////////////////////////////////
//
//  Multilayer Feature Graph (MFG), version 1.0
//  Copyright (C) 2011-2015 Yan Lu, Madison Treat, Dezhen Song
//  Netbot Laboratory, Texas A&M University, USA
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//
/////////////////////////////////////////////////////////////////////////////////

/********************************************************************************
 * Constants
 ********************************************************************************/


#ifndef CONSTS_H_
#define CONSTS_H_

// Key Point/Feature Point Detection algorithms: SIFT, SURF, GFTT
enum FeatureDetectionAlgorithm
{
    KPT_NONE = 0,
    KPT_SIFT = 1,
    KPT_SURF = 2,
    KPT_GFTT = 3
};

static const double PI = 3.14159265;

#endif

