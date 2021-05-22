#include "headers/initialize_beliefs.h"

using namespace std;

// OPTIMIZATION: pass large variables by reference
vector< vector <float> > initialize_beliefs(vector< vector <char> > &grid) {

	// OPTIMIZATION: Which of these variables are necessary?
	// OPTIMIZATION: Reserve space in memory for vectors
	int i, j, height, width, area;
	float total, prob_per_cell;

	height = grid.size(); 
	width = grid[0].size();
	area = height * width; 
  	prob_per_cell = 1.0 / ( (float) area) ; // all 3 vars make code clearer and don't affect speed much
  
	vector<float> newRow;
  	newRow.reserve(width);
  	newRow.assign(width, prob_per_cell);
  	vector< vector <float> > newGrid;
  	newGrid.reserve(height);
    newGrid.assign(height, newRow);
  
  	return newGrid;

  	// OPTIMIZATION: Is there a way to get the same results 	// without nested for loops?
}