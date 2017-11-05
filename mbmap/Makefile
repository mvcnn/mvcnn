CFLAGS = -O3 -D__STDC_CONSTANT_MACROS
LDFLAGS = -lswscale # -lbz2 -lz -lc
INSTALLED_DEPS = -Idependencies/include -Ldependencies/lib

mvx: main.cpp helper.cpp frame.cpp
	g++ $^ -g -o $@ $(CFLAGS) $(LDFLAGS) $(INSTALLED_DEPS)

