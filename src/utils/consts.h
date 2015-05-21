/*
 * File:   consts.h
 * Author: madtreat
 *
 * Created on February 16, 2015, 2:12 PM
 *
 * This header is for defining constants and re-usable types to simplify
 * code construction.
 */

#ifndef CONSTS_H_
#define CONSTS_H_

#include <vector>
#include <opencv2/core/core.hpp>

// TODO: Make these actual pairs instead of vectors

// A pair of Feature Points, and a list of them
typedef std::vector<cv::Point2d> FeaturePointPair;
typedef std::vector<FeaturePointPair> FeaturePointPairs;


// A pair of Vanishing Point, and a list of them
typedef std::vector<cv::Mat> VPointPair;
typedef std::vector<VPointPair> VPointPairs;


// Key Point/Feature Point Detection algorithms: SIFT, SURF, GFTT
enum FeatureDetectionAlgorithm {
   KPT_NONE = 0,
   KPT_SIFT = 1,
   KPT_SURF = 2,
   KPT_GFTT = 3
};

static const double PI = 3.14159265;

#endif

