
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
#define MAX_GRID            512
#define MAX_STRING_SIZE     512
#define MVCH		    2

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
extern bool GRID_8X8, SHOW_HELP ;
extern const char* VIDEO_PATH, *OUT_PATH, *RGB_PATH ;
extern int RGB_THRESH, REF_INTERVAL, WRITE_INTERVAL ;



class AVL_H264{
public:
static void initialize_av() ;
static bool process_frame(AVPacket *pkt) ;
static bool read_packets(FILE* mout) ;
static bool read_frame(int64_t& pts, char& pictType, vector<AVMotionVector>& motion_vectors, AVFrame& avframe,  FILE* mout) ;
// static void output_frame(int , int64_t , char , vector<AVMotionVector>& motionVectors, AVFrame& avframe , FILE* , FILE* ) ;

};


class Helper {
public:
static FILE *openFile(char *) ;
static void openFolder(const char *, bool) ;
static void parse_input(int argc, char *argv[], float *d, float *fixed_thres,   
                float *adapt_thres_coef_shift, int *n_max, int *block_size) ;
static void parse_options(int argc, const char* argv[]) ;

 } ;


struct Frame 
{ 
const static size_t MAX_GRID_SIZE  ; 
 
size_t GridStep; 
pair<size_t, size_t> Shape; 
 

bool mb[MAX_GRID][MAX_GRID] ;
bool empty[MAX_GRID][MAX_GRID] ;
signed char mv[MAX_GRID][MAX_GRID][MVCH] ;
int64_t Pts; 
int FrameIndex; 
char PictType; 
const char* Origin; 
bool Empty; 
bool Printed; 
AVFrame avframe ; Frame() ; 



void InterpolateFlow(Frame& prev, Frame& next) ; 
void FillInSomeMissingVectorsInGrid8(); 
void FillInSomeMissingVectorsInGrid4(); 
void PrintIfNotPrinted(FILE* fout, FILE* mout); 
void setup(AVFrame avframe, int index, int64_t pts, char type, vector<AVMotionVector>, size_t w, size_t h) ; 
void print(AVFrame*, FILE*, FILE*) ; 

public:
static void output_frame(int frameIndex, int64_t pts, char pictType, vector<AVMotionVector>& motionVectors, AVFrame& avframe , FILE* fout, FILE* mout) ;

} ; 


#endif
