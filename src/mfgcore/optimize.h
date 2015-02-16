/*
 * File:   optimize.h
 * Author: madtreat
 *
 * Created on February 16, 2015, 1:54 PM
 */

#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include "consts.h"

////////////////////////////////////////////////////////////////////////////////
// Optimization of R^T with Vanishing Points
////////////////////////////////////////////////////////////////////////////////
struct Data_optimizeRt_withVP
{
	FeaturePointPairs pointPairs;
	cv::Mat K;
	VPointPairs	vpPairs;
	double weightVP;
};
void costFun_optimizeRt_withVP (double *p, double *error, int N, int M, void *adata);
void optimizeRt_withVP (cv::Mat K, VPointPairs vppairs, double weightVP,
                        FeaturePointPairs& featPtMatches, cv::Mat R, cv::Mat t);


////////////////////////////////////////////////////////////////////////////////
// Optimization of T Given R
////////////////////////////////////////////////////////////////////////////////
struct Data_optimize_t_givenR
{
	FeaturePointPairs pointPairs;
	cv::Mat K, R;
};
void costFun_optimize_t_givenR (double *p, double *error, int N, int M, void *adata);
void optimize_t_givenR (cv::Mat K, cv::Mat R, FeaturePointPairs& featPtMatches, cv::Mat t);

#endif	/* OPTIMIZE_H */

