#include "all_defines.h"

Frame::Frame()
{}

        void Frame::setup(vector<AVMotionVector>& avmv, index, type)
        {
	memset(empty, 0, sizeof(bool)) ;
	memset(mv, 0, sizeof(signed char)) ; memset(mb, 0, sizeof(bool)) ;
        this->type = type ; this->index = index ; this->pts = -1;

        AVMotionVector& avmv_ ;
        for(int i = 0; i < avmv.size(); i++)
        	{
        	        avmv_ = motionVectors[i];
			size_t MAX = h - 1 ; size_t MIN = 0 ;
        	        size_t pos_x = max(MIN, min(avmv_.dst_x / mbsize , MAX)) ;
        	        size_t pos_y = max(MIN, min(avmv_.dst_y / mbsize , MAX)) ;

			mb[pos_x][pos_y] = 1 ;
        	        mv[pos_x][pos_y][0] = (signed char) (avmv_.dst_x - avmv_.src_x) ;
        	        mv[pos_x][pos_y][1] = (signed char) (avmv_.dst_y - avmv_.src_y) ;
        	}
	}

        void Frame::print(AVFrame *avframe, FILE *fout, FILE *mout)
        {
                static int64_t FirstPts = -1;

                if(Printed)
                        return;

                if(FirstPts == -1)
                        FirstPts = Pts;

        fprintf(stdout,"Called for frame  -- %d\n", FrameIndex) ;
        // Offset PTS
        int PTS_Offset = (int) Pts - (int) FirstPts ;
        // Define xdim and ydim ..      
        int ydim = (((int) Shape.first) * 2) ; int xdim  = ((int) Shape.second) ;

        // Write frame header..
        fwrite(&PTS_Offset, sizeof(int), 1, fout) ;
        fwrite(&FrameIndex, sizeof(int), 1, fout) ;
        fwrite(&xdim, sizeof(int), 1, fout) ;
        fwrite(&ydim, sizeof(int), 1, fout) ;
        fwrite(&PictType, sizeof(char), 1, fout) ;

        // Write DXY values ..
        for(int i = 0; i < (ydim/2) ; i++){
                fwrite(&dx[i], sizeof(signed char), xdim , fout) ;
        }
        for(int i = 0; i < (ydim/2) ; i++){
                fwrite(&dy[i], sizeof(signed char), xdim , fout) ;
        }

        // -- Y Data
        int absx ; int absy ;
        int frame_data_pos ; int rgb_data_pos ;

        AVFrame* pFrame = &avframe ;
        int image_size = (xdim*GridStep)*((ydim/2)*GridStep) ;

        unsigned char* Y = new unsigned char[image_size]() ;

        unsigned char* R = new unsigned char[image_size]() ;
        unsigned char* G = new unsigned char[image_size]() ;
        unsigned char* B = new unsigned char[image_size]() ;

        bool is_reference = ((FrameIndex % REF_INTERVAL) == 0 || FrameIndex == 1) ;

        for(int i = 0 ; i < xdim ; i++){
        for(int j = 0; j < (ydim/2) ; j++){

        absx = i * GridStep ; absy = j * GridStep ;

        if ((dx[j][i]*dx[j][i] + dy[j][i]*dy[j][i]) > RGB_THRESH){
                for ( int k = 0 ; k < GridStep ; k++ ){

                frame_data_pos = (pFrame->linesize[0]* (absy + k)) + absx;
                rgb_data_pos = (pFrame->linesize[0]* (absy + k)) + absx ;

                for (int l = 0 ; l < GridStep ; l++) {
                /*
                *(R + rgb_data_pos + l) = (unsigned char) pFrame->data[0][frame_data_pos+l*3+0];
                *(G + rgb_data_pos + l) = (unsigned char) pFrame->data[1][frame_data_pos+l*3+1];
                *(B + rgb_data_pos + l) = (unsigned char) pFrame->data[2][frame_data_pos+l*3+2]; }*/

                if(is_reference){

                // Update RGB as Reference
                *(R + rgb_data_pos + l) = (unsigned char) pFrame->data[0][frame_data_pos+l];
                *(G + rgb_data_pos + l) = (unsigned char) pFrame->data[1][frame_data_pos+l];
                *(B + rgb_data_pos + l) = (unsigned char) pFrame->data[2][frame_data_pos+l]; }

                else{
                if (std::abs(dx[j][i]) > RGB_THRESH || std::abs(dy[j][i]) > RGB_THRESH){

                *(R + rgb_data_pos + l) = (unsigned char) pFrame->data[0][frame_data_pos+l];
                *(G + rgb_data_pos + l) = (unsigned char) pFrame->data[1][frame_data_pos+l];
                *(B + rgb_data_pos + l) = (unsigned char) pFrame->data[2][frame_data_pos+l]; }}


                // fwrite(&(pFrame->data[0][frame_data_pos+l]), sizeof(unsigned char), 1, mout) ;
                // fwrite(&(pFrame->data[1][frame_data_pos+l]), sizeof(unsigned char), 1, mout) ;
                // fwrite(&(pFrame->data[2][frame_data_pos+l]), sizeof(unsigned char), 1, mout) ; }

                }}} // extra ?

        }}
        if((FrameIndex % WRITE_INTERVAL) == 0){
                fwrite(R, sizeof(unsigned char), image_size, mout) ;
                fwrite(G, sizeof(unsigned char), image_size, mout) ;
                fwrite(B, sizeof(unsigned char), image_size, mout) ; }

        Printed = true ; }


void Frame::output_frame(int frameIndex, int64_t pts, char type, vector<AVMotionVector>& motionVectors, AVFrame& avframe , FILE* fout, FILE* mout)
{

        Frame frame ; AVL_H264::read_frame(pts, type, motionVectors, avframe) ;
	
        size_t mbsize = 16 ; mbsize = GRID_8X8 ? 8 : mbsize ;

	const size_t max_grid = MAX_GRID ;
	size_t w = min(frame_width / mbsize, max_grid) ;
	size_t h = min(frame_height / mbsize, max_grid) ;
	frame.setup(avframe, index, type, mv, w, h) ;

        for(int i = 0; i < motionVectors.size(); i++)
        {
                AVMotionVector& mv = motionVectors[i];
                int mvdx =  mv.dst_x - mv.src_x;
                int mvdy = mv.dst_y - mv.src_y;

                size_t i_clipped = max(size_t(0), min(mv.dst_y / frame.GridStep, frame.Shape.first - 1));
                size_t j_clipped = max(size_t(0), min(mv.dst_x / frame.GridStep, frame.Shape.second - 1));

                frame.Empty = false;
                frame.dx[i_clipped][j_clipped] = ((signed char) mvdx);
                frame.dy[i_clipped][j_clipped] = ((signed char) mvdy);
                frame.occupancy[i_clipped][j_clipped] = true;
        }

        if(!motionVectors.empty()) {
        if (frame.GridStep == 8) frame.FillInSomeMissingVectorsInGrid8() ;
        frame.PrintIfNotPrinted(fout, mout) ; }
        }
 
