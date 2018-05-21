
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
#include <boost/assign/list_inserter.hpp> // for 'insert()'
#include <boost/assert.hpp>

extern "C"
{
        #include <libavcodec/avcodec.h>
        #include <libavformat/avformat.h>
        #include <libswscale/swscale.h>
        #include <libavutil/motion_vector.h>
}

#include <string>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include<time.h>


// Default
#define SELDEC_THRESHOLD    0
#define SLEDEC_REF          5
#define SELDEC_WRITE        1
#define MAX_MAP            512
#define MAX_STRING_SIZE     512
#define MV_CHANNELS	    2

using namespace std;
using namespace boost::assign ;

// extern AVFrame* av_frame ;
// extern int av_stream_index;
// extern AVFormatContext* av_context ; extern AVStream* av_stream ;
// extern AVPixelFormat pixel_format ;
// extern size_t frame_width, frame_height;
// 
// 
// extern bool ARG_OUTPUT_RAW_MOTION_VECTORS, ARG_FORCE_GRID_8, ARG_FORCE_GRID_4, ARG_FORCE_GRID_16, ARG_OUTPUT_OCCUPANCY, ARG_QUIET, ARG_HELP, ARG_MORE ;
// extern const char* ARG_VIDEO_PATH; extern const char* OUT_PATH ; extern const char* LABEL_PATH ; extern const char* MORE_PATH ;  extern const char* RGB_PATH ;
// extern int MV_THRESH ; extern int REF_INTERVAL ; extern int WRITE_INTERVAL ;



extern AVFrame *av_frame ;
extern AVFormatContext *av_context ; extern AVStream *av_stream ;
extern int av_stream_index ; extern size_t frame_width, frame_height ;
extern bool MAP_TO_8X8, SHOW_HELP ;
extern const char* VIDEO_PATH, *OUT_PATH, *RGB_PATH ;
extern int RGB_THRESH, REF_INTERVAL, WRITE_INTERVAL ;


struct Frame 
{ 
const static size_t MAX_MAP_SIZE  ; 
 
AVFrame* pFrame ;
bool mb[MAX_MAP][MAX_MAP] ;
bool empty[MAX_MAP][MAX_MAP] ;
signed char mv[MV_CHANNELS][MAX_MAP][MAX_MAP] ;
int  pts ;  int index ; 
char type ;  // AVFrame* avframe ; 
int width ; int height ; 

void setup(AVFrame*, vector<AVMotionVector>& ) ; 
void print(FILE*, FILE*) ; 
public:
Frame() ; 

} ; 



class AVL_H264{

AVFormatContext* avcontext ; 
AVStream* avstream ; int avstream_index ;
Frame frame ;
AVFrame* avframe ;
vector<AVMotionVector> avmv ;
int pts ; char type ;

FILE* mvout ; FILE* rgbout ;
public:
AVL_H264() ;
void initialize() ;
bool decode_packets() ;
bool get_motion_vectors()   ;
// void output_frame(int , int64_t , char , vector<AVMotionVector>& motionVectors, AVFrame& avframe , FILE* , FILE* ) ;

};


class Helper {
public:
static FILE *openFile(char *) ;
static void openFolder(const char *, bool) ;
static void parse_input(int argc, char *argv[], float *d, float *fixed_thres,   
                float *adapt_thres_coef_shift, int *n_max, int *block_size) ;
static void parse_options(int argc, const char* argv[]) ;

 } ;


#endif
