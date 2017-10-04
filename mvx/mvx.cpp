#include "all_defines.h"

// Required by AVLib
int av_stream_index ; size_t frame_width, frame_height ;
AVFrame* av_frame ; AVFormatContext* av_context ; AVStream* av_stream ;

// Extraction settings
bool GRID_8X8, SHOW_HELP ; 
int RGB_THRESH, REF_INTERVAL, WRITE_INTERVAL ;
const char *VIDEO_PATH, *OUT_PATH, *RGB_PATH ;


int main(int argc, const char* argv[])
{
    Helper::parse_options(argc, argv);
    clock_t begin_stdout = clock();

    FILE* mvout ; mvout = fopen(OUT_PATH,"wb") ;
    FILE* rgbout ; rgbout = fopen(RGB_PATH,"wb") ;

    AVL_H264::initialize_av();
    
    char type ; int64_t pts, pts_ = -1 ;
    AVFrame avframe ; vector<AVMotionVector> motionVectors ;
    
    int index = 0 ; clock_t begin_ffinit = clock();
    while(true){
	// AVL_H264::read_frame(pts, type, motionVectors, avframe, rgbout) ;
	if (pts <= pts_ && pts_ != -1) continue ;
	Frame::output_frame(index, pts, type, motionVectors, avframe, mvout, rgbout) ; 

	pts_ = pts ; index = index + 1 ;
    }

 clock_t end_stdout = clock();
 double time_spent_stdout = (double)(begin_stdout - end_stdout) / CLOCKS_PER_SEC;

 fprintf(stdout, "%f\n", (time_spent_stdout) * -1);

}

