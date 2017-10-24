#include "all_defines.h"

// Required by AVLib
int av_stream_index ; size_t frame_width, frame_height ;

// Extraction settings
bool GRID_8X8, SHOW_HELP ; 
int RGB_THRESH, REF_INTERVAL, WRITE_INTERVAL ;
const char *VIDEO_PATH, *OUT_PATH, *RGB_PATH ;


int main(int argc, const char* argv[])
{
	Helper::parse_options(argc, argv);
	clock_t begin = clock();

	AVL_H264 avlh = AVL_H264() ;
	avlh.initialize() ;	
	clock_t begin_ffinit = clock();
	while(avlh.get_motion_vectors()){continue ;}

	clock_t end = clock();
	double time_spent_stdout = (double)(end - begin) / CLOCKS_PER_SEC;

	fprintf(stdout, "Runtime: %f\n", (time_spent_stdout));

}

