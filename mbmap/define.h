
#ifndef ALL_DEFINES
#define ALL_DEFINES

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <stdint.h>
#include<cmath>

// Label Reader ..
#include <sstream>
#include <string>
#include <fstream>
#include <map>

// Boost incs:
// #include <boost/assign/list_inserter.hpp> // for 'insert()'
// #include <boost/assert.hpp>

#include <string>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include<time.h>


// Default
#define MB_FRAME_DIM        512
#define MAX_GRID            512
#define MAX_STRING_SIZE     512
#define MV_CHANNELS	    2
#define MAX_FRAME	    1024

using namespace std;
// using namespace boost::assign ;

extern int FRAME_WIDTH ; extern int FRAME_HEIGHT ;
extern bool GRID_8X8, SHOW_HELP ;
extern const char *INPUT_PATH, *OUTPUT_PATH ;

struct JMacroBlock 
{ int  frame, type, x, y, dx, dy ; } ; 



struct Frame 
{ 
const static size_t MAX_GRID_SIZE  ; 
 
bool mb[MB_FRAME_DIM][MB_FRAME_DIM] ;
bool empty[MB_FRAME_DIM][MB_FRAME_DIM] ;
signed char mv[MV_CHANNELS][MB_FRAME_DIM][MB_FRAME_DIM] ;
int  pts ;  int index ; char type ;  
int width ; int height ; 

void setup(vector<JMacroBlock>& ) ; 
void print(FILE*) ; 
public:
Frame() ; 

} ; 

class Helper {
public:
static FILE *openFile(char *) ;
static void parse_input(int argc, char *argv[], float *d, float *fixed_thres,   
                float *adapt_thres_coef_shift, int *n_max, int *block_size) ;
static void parse_options(int argc, const char* argv[]) ;

 } ;


#endif
