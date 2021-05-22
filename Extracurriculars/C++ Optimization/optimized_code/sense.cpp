#include "headers/sense.h"

using namespace std;

// OPTIMIZATION: Pass larger variables by reference
vector< vector <float> > sense(char color, vector< vector <char> > &grid, vector< vector <float> > &beliefs,  float p_hit, float p_miss) 
{
	// OPTIMIZATION: Is the newGrid variable necessary?
  	// Could the beliefs input variable be updated directly?

	float prior, p;

	char cell;

	int i, j, size;
	size = grid.size();

	for (i=0; i<size; i++) {
		for (j=0; j<size; j++) {
          	// OPTIMIZATION: Which of these variables are needed?
			prior = beliefs[i][j];
			cell = grid[i][j];
          	// again, these lines make code clearer and don't affect speed much
          
			if (cell == color) {
				beliefs[i][j] = prior * p_hit;
			}
            // OPTIMIZATION: if else statements might be 
          	// 	faster than two if statements
			else {
				p = prior * p_miss;
			}
		}
	}
	return beliefs;
}
