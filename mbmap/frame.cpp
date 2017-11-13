#include "define.h"

Frame::Frame()
{ 
	memset(empty, 0, sizeof(empty)) ;
	memset(mv, 0, sizeof(mv)) ; memset(mb, 0, sizeof(mb)) ;
}


void Frame::smooth(){

	size_t mbsize  = GRID_8X8 ? 8:16  ;

        int ydim = (height/mbsize)*2 ; int xdim  = width/mbsize ;

for (int l = 1 ; l > 0 ; l--){
  for(int k = 0; k < 2 ; k++)
  {       
        for(int i = 1; i < (ydim/2) - 1; i++)
        {       
                for(int j = 1; j < xdim - 1; j++)
                {       
                        if(mb[i][j] == 0)
			
                                if(mb[i][j - l] != 0 && mb[i][j + l] != 0)
                                {        
                                        mv[0][i][j] = (mv[0][i][j -l] + mv[0][i][j + l]) / 2;
                                        mv[1][i][j] = (mv[1][i][j -l] + mv[1][i][j + l]) / 2;
                                        mb[i][j] = 2;
                                }
                                else if(mb[i - l][j] != 0 && mb[i + l][j] != 0)
                                {        
                                        mv[0][i][j] = (mv[0][i - l][j] + mv[0][i + l][j]) / 2;
                                        mv[1][i][j] = (mv[1][i - l][j] + mv[1][i + l][j]) / 2;
                                        mb[i][j] = 2;
                                }
				// else{mv[0][i][j] = 5 ; mv[1][i][j] = 5 ;}
                        }
                }
        }
  }
}



void Frame::setup(vector<JMacroBlock>& avmv)
{

	memset(mv, 0, sizeof(mv)) ; 
	memset(mb, 0, sizeof(mb)) ; 
	size_t mbsize  = GRID_8X8 ? 8:16  ;
	int x_mv_length = MAX_GRID ;
	int y_mv_length = MAX_GRID ;


	signed char *pos ;
	signed char *x_mv_start_pos = &mv[0][0][0] ;
	signed char *y_mv_start_pos = x_mv_start_pos + MAX_GRID * MAX_GRID ;
	
	for(int i = 0; i < avmv.size(); i++)
	{
		size_t MAX  ; size_t MIN  ;
	        JMacroBlock& avmv_ = avmv[i] ;
		if (avmv_.dx == 0 && avmv_.dy == 0) continue ;

		MAX = (width/mbsize) - 1 ; MIN = 0 ;
	        size_t pos_x = max(MIN, min(avmv_.x / mbsize , MAX)) ;
		MAX = (height/mbsize) - 1 ; MIN = 0 ;
	        size_t pos_y = max(MIN, min(avmv_.y / mbsize , MAX)) ;
		
		mb[pos_y][pos_x] = true ;
		pos = x_mv_start_pos + pos_y * MAX_GRID + pos_x ;
                *(pos) = (unsigned char) (avmv_.dx) ;

		pos = y_mv_start_pos + pos_y * MAX_GRID + pos_x ;
                *(pos) = (unsigned char) (avmv_.dy) ;
	}
}

void Frame::print(FILE *fout)
        {

	size_t mbsize  = GRID_8X8 ? 8:16  ;

        int ydim = (height/mbsize)*2 ; int xdim  = width/mbsize ;

        // Write frame header..
        fwrite(&index, sizeof(int), 1, fout) ;
        fwrite(&index, sizeof(int), 1, fout) ;
        fwrite(&xdim, sizeof(int), 1, fout) ;
        fwrite(&ydim, sizeof(int), 1, fout) ;
        fwrite(&type, sizeof(char), 1, fout) ;
	fprintf(stdout, "Printing frame -- xdim: %d, ydim: %d, width: %d, height: %d\n", xdim, ydim, width, FRAME_HEIGHT) ;
	for (int i = 0 ; i < ydim/2 ; i++){
        	fwrite(&mv[0][i], sizeof(signed char), xdim , fout) ; }

	for (int i = 0 ; i < ydim/2 ; i++){
        	fwrite(&mv[1][i], sizeof(signed char), xdim , fout) ; }
        }
