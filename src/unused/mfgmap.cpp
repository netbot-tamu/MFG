
#include "mfg.h"
#include "mfg_utils.h"

void MfgMap::registerLandmark(vector<MfgSingleView>& singleViewSeq,
								vector<MfgTwoView>& twoViewSeq)
{
	int n1 = singleViewSeq.size(), n2 = twoViewSeq.size(); // num of mfgs
	int p = twoViewSeq[n2-1].onPlaneLineMatchesIdx.size(); // plane number
	for (int i=0; i < p; ++i) { // for each plane
		for (int j=0; j<twoViewSeq[n2-1].onPlaneLineMatchesIdx[i].size(); ++j){
			int group = twoViewSeq[n2-1].onPlaneLineMatchesIdx[i][j][0],
				// Note: idx1 corresponds to I_k, i.e. the k-th view,
				// idx2 corresponds to I_k-1, the (k-1) th view,
				// because MFG-2view is built using I_k as first view!!!!!!
				idx1  = twoViewSeq[n2-1].onPlaneLineMatchesIdx[i][j][1],// I_k
				idx2  = twoViewSeq[n2-1].onPlaneLineMatchesIdx[i][j][2];// I_k-1  

			if ( singleViewSeq[n1-2].idealLineGroups[group][idx2].gID < 0) {
				// setup the view ID for the new line landmark 
				singleViewSeq[n1-1].idealLineGroups[group][idx1].viewID = n1-1;
				// assign a gID to new line landmark			
				singleViewSeq[n1-1].idealLineGroups[group][idx1].gID = lines.size();
				// Note that line equation is still w.r.t. current CCS !!!!!
				// In slam state, this equation must be converted to WCS.
				lines.push_back (singleViewSeq[n1-1].idealLineGroups[group][idx1]);
			} else { // when line landmark exists already
				singleViewSeq[n1-1].idealLineGroups[group][idx1].gID = 
					singleViewSeq[n1-2].idealLineGroups[group][idx2].gID;
				singleViewSeq[n1-1].idealLineGroups[group][idx1].pID = 
					singleViewSeq[n1-2].idealLineGroups[group][idx2].pID;
				singleViewSeq[n1-1].idealLineGroups[group][idx1].viewID = 
					singleViewSeq[n1-2].idealLineGroups[group][idx2].viewID;

				//  Match plane between two 2view-MFGs
			}
		}
	}
	

}

void MfgMap::initialize (vector<MfgSingleView>& singleViewSeq,
						vector<MfgTwoView>& twoViewSeq)
// input : a vector of two 1-view-MFG from I_0, I_1
// input : a empty vector of 2-view-MFG
{
	if ( singleViewSeq.size() == 2 )
		return;

	MfgTwoView m0(singleViewSeq[0],singleViewSeq[1]);
	// for each plane in MFG, set them to be the landmark planes of map
	// for each line associated with plane, also put them in map
	for (int i=0; i < m0.primaryPlanes.size(); ++i) {
		m0.primaryPlanes[i].gID = planes.size();		
		for (int j=0; j<m0.onPlaneLineMatchesIdx[i].size(); ++j) {
			int g, id1, id2;
			g   = m0.onPlaneLineMatchesIdx[i][j][0]; // parallel line group No.
			id1 = m0.onPlaneLineMatchesIdx[i][j][1]; // line idx in first view
			id2 = m0.onPlaneLineMatchesIdx[i][j][2];

			singleViewSeq[0].idealLineGroups[g][id1].pID = m0.primaryPlanes[i].gID;
			singleViewSeq[1].idealLineGroups[g][id2].pID = m0.primaryPlanes[i].gID;
			singleViewSeq[0].idealLineGroups[g][id1].gID = lines.size();
			singleViewSeq[1].idealLineGroups[g][id2].gID = lines.size();
			singleViewSeq[0].idealLineGroups[g][id1].viewID = 0;
			singleViewSeq[1].idealLineGroups[g][id2].viewID = 0;
			lines.push_back (singleViewSeq[0].idealLineGroups[g][id1]);
		}
		planes.push_back (m0.primaryPlanes[i]);
	}


	
}
