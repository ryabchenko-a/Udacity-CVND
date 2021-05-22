#include "headers/blur.h"

using namespace std;

// OPTIMIZATION: Pass large variable by reference
vector < vector <float> > blur(vector < vector < float> > &grid, float blurring) {

	// OPTIMIZATION: window, DX and  DY variables have the 
    // same value each time the function is run.
  	// It's very inefficient to recalculate the vectors
    // every time the function runs. 
    // 
    // The const and/or static operator could be useful.
  	// Define and declare window, DX, and DY using the
    // bracket syntax: vector<int> foo = {1, 2, 3, 4} 
    // instead of calculating these vectors with for loops 
    // and push back

	int height;
	int width;
	float center, corner, adjacent;

	height = grid.size();
	width = grid[0].size();

	center = 1.0 - blurring;
	corner = blurring / 12.0;
	adjacent = blurring / 6.0;
	static const vector < vector <float> > window{vector <float> {corner, adjacent, corner}, 
	  											vector <float> {adjacent, center, adjacent}, 
												vector <float> {corner, adjacent, corner}};

	static const vector <int> DXY{-1, 0, 1};

	int dx, dy, i, j, ii, jj, new_i, new_j;
	float multiplier, val, newVal;

	// OPTIMIZATION: Use your improved zeros function

	for (i=0; i< height; i++ ) {
		for (j=0; j<width; j++ ) {
			val = grid[i][j];
			newVal = val;
			for (ii=0; ii<3; ii++) {
				dy = DXY[ii];
				for (jj=0; jj<3; jj++) {
					dx = DXY[jj];
					new_i = (i + dy + height) % height;
					new_j = (j + dx + width) % width;
					multiplier = window[ii][jj];
					grid[new_i][new_j] += newVal * multiplier;
				}
			}
		}
	}

	return grid;
}