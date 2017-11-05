#include "all_defines.h"

Frame::Frame()
{ 
	pts = -1 ;
	memset(empty, 0, sizeof(empty)) ;
	memset(mv, 0, sizeof(mv)) ; memset(mb, 0, sizeof(mb)) ;
}

void Frame::setup(AVFrame* avframe, vector<MacroBlock>& mbvec )
{
	size_t mbsize  = 16 ;
	pFrame = avframe ;
	int x_mv_length = MAX_GRID ;
	int y_mv_length = MAX_GRID ;


	signed char *pos ;
	signed char *x_mv_start_pos = &mv[0][0][0] ;
	signed char *y_mv_start_pos = x_mv_start_pos + MAX_GRID * MAX_GRID ;
	
	for(int i = 0; i < avmv.size(); i++)
	{
		size_t MAX  ; size_t MIN  ;
	        MacroBlock& mbvec = mbvec[i];
		MAX = (width/16) - 1 ; MIN = 0 ;
	        size_t pos_x = max(MIN, min(avmv_.dst_x / mbsize , MAX)) ;
		MAX = (height/16) - 1 ; MIN = 0 ;
	        size_t pos_y = max(MIN, min(avmv_.dst_y / mbsize , MAX)) ;
		
		mb[pos_x][pos_y] = 1 ;
		pos = x_mv_start_pos + pos_y * MAX_GRID + pos_x ;
                *(pos) = (unsigned char) (avmv_.dst_x - avmv_.src_x) ;

		pos = y_mv_start_pos + pos_y * MAX_GRID + pos_x ;
                *(pos) = (unsigned char) (avmv_.dst_y - avmv_.src_y) ;
	}
}

void Frame::print(FILE *fout, FILE *mout)
        {
        int ydim = (height/16)*2 ; int xdim  = width/16 ;

        // Write frame header..
	
        fwrite(&index, sizeof(int), 1, fout) ;
        fwrite(&index, sizeof(int), 1, fout) ;
        fwrite(&xdim, sizeof(int), 1, fout) ;
        fwrite(&ydim, sizeof(int), 1, fout) ;
        fwrite(&type, sizeof(char), 1, fout) ;
	for (int i = 0 ; i < ydim/2 ; i++){

        	fwrite(&mv[0][i], sizeof(signed char), xdim , fout) ;
	}
	for (int i = 0 ; i < ydim/2 ; i++){

        	fwrite(&mv[1][i], sizeof(signed char), xdim , fout) ;
	}

       //  fwrite(&mv[1], sizeof(signed char), xdim*ydim/2 , fout) ;
        
	int absx ; int absy ;
        int frame_data_pos ; int rgb_data_pos ;

        int image_size = height * width ;

         unsigned char* R = new unsigned char[image_size]() ;
         unsigned char* G = new unsigned char[image_size]() ;
         unsigned char* B = new unsigned char[image_size]() ;
 
         bool is_reference = ((index % REF_INTERVAL) == 0 || index == 1) ;
 
         for(int i = 0 ; i < xdim ; i++){
         for(int j = 0; j < (ydim/2) ; j++){
 
         absx = i * 16 ; absy = j * 16 ;
 
         if ((mv[0][j][i]*mv[0][j][i] + mv[1][j][i]*mv[1][j][i]) > RGB_THRESH){
                 for ( int k = 0 ; k < 16 ; k++ ){
 
                 frame_data_pos = (pFrame->linesize[0]* (absy + k)) + absx;
                 rgb_data_pos = (pFrame->linesize[0]* (absy + k)) + absx ;
 
                 for (int l = 0 ; l < 16 ; l++) {
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
                 if (std::abs(mv[0][j][i]) > RGB_THRESH || std::abs(mv[1][j][i]) > RGB_THRESH){
 
                 *(R + rgb_data_pos + l) = (unsigned char) pFrame->data[0][frame_data_pos+l];
                 *(G + rgb_data_pos + l) = (unsigned char) pFrame->data[1][frame_data_pos+l];
                 *(B + rgb_data_pos + l) = (unsigned char) pFrame->data[2][frame_data_pos+l]; }}
 
 
                 // fwrite(&(pFrame->data[0][frame_data_pos+l]), sizeof(unsigned char), 1, mout) ;
                 // fwrite(&(pFrame->data[1][frame_data_pos+l]), sizeof(unsigned char), 1, mout) ;
                 // fwrite(&(pFrame->data[2][frame_data_pos+l]), sizeof(unsigned char), 1, mout) ; }
 
                 }}} // extra ?
 
         }}
         // fprintf(stdout,"Called for RAND  -- %d %d\n", xdim, ydim) ;
         if((index % WRITE_INTERVAL) == 0){
                 fwrite(R, sizeof(unsigned char), image_size, mout) ;
                 fwrite(G, sizeof(unsigned char), image_size, mout) ;
                 fwrite(B, sizeof(unsigned char), image_size, mout) ; }

        }
