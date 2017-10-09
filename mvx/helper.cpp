#include "all_defines.h"


FILE *Helper::openFile(char *fn){
        FILE *f = fopen(fn, "wt") ;
        if (f == NULL) { printf("File %s could not be created.\n", fn) ; exit(-1) ; }
	else { return f ; }
	}

void Helper::parse_options(int argc, const char* argv[])
{
        for(int i = 1; i < argc; i++)
        {
                if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
                        SHOW_HELP = true;
                else if(strcmp(argv[i], "-8") == 0)
                        GRID_8X8 = true;
                else if (strcmp(argv[i], "-o") == 0)
                        OUT_PATH = argv[++i];
                else if (strcmp(argv[i], "-t") == 0)
                        RGB_THRESH = atoi(argv[++i]);
                else if (strcmp(argv[i], "-r") == 0)
                        REF_INTERVAL = atoi(argv[++i]);
                else if (strcmp(argv[i], "-w") == 0)
                        WRITE_INTERVAL = atoi(argv[++i]);
                else if (strcmp(argv[i], "--rgb") == 0)
                        RGB_PATH = argv[++i];
                else
                        VIDEO_PATH = argv[i];
        }
}

