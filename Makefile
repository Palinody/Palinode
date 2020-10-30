IDIR = headers
CC = g++
CXXFLAGS = -std=c++20 -O3 -march=native
DEBUG_CFLAGS = -Wall -Werror -pedantic -ggdb3 -Wno-error=unknown-pragmas
#SRC1 = smth1.cpp smth2.cpp \
		smth3.cpp
SRC2 = main.cpp
#SRC = $(SRC1) $(SRC2)
SRC = $(SRC2)
main: $(SRC2)

.PHONY: clean
clean:
	rm -f *.o main