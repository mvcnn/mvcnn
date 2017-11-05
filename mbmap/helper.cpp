#include "define.h"

void Helper::parse_options(int argc, const char* argv[])
{
        for(int i = 1; i < argc; i++)
        {
                if(strcmp(argv[i], "--help") == 0 )
                        SHOW_HELP = true;
                else if(strcmp(argv[i], "-8") == 0)
                        GRID_8X8 = true ;
		else if (strcmp(argv[i], "-w") == 0)
                        FRAME_WIDTH = atoi(argv[++i]);
		else if (strcmp(argv[i], "-h") == 0)
                        FRAME_HEIGHT = atoi(argv[++i]);
                else if (strcmp(argv[i], "-o") == 0)
                        OUTPUT_PATH = argv[++i];
                else
                        INPUT_PATH = argv[i];
        }
}

