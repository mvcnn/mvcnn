CFLAGS = -O3 -D__STDC_CONSTANT_MACROS
LDFLAGS = -lswscale -lavformat -lavcodec -lswresample -lavutil -lpthread -lbz2 -lz -lc -lrt
INSTALLED_DEPS = -Idependencies/include -Ldependencies/lib

mvx: mvx.cpp frame.cpp helper.cpp av_dec_h264.cpp
	g++ $^ -o $@ $(CFLAGS) $(LDFLAGS) $(INSTALLED_DEPS)

