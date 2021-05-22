#include "headers/zeros.h"

using namespace std;

vector < vector <float> > zeros(int height, int width) {
	int i, j;
  
	// OPTIMIZATION: Reserve space in memory for vectors
	vector <float> newRow;
	newRow.reserve(width);
  	newRow.assign(width, 0.0);
  	vector < vector <float> > newGrid;
  	newGrid.reserve(height);
  	newGrid.assign(height, newRow);
  	// OPTIMIZATION: nested for loop not needed
    // because every row in the matrix is exactly the same
	return newGrid;
}