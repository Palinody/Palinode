IDIR = headers
CC = g++-10
CFLAGS = -O3 -march=native
DEBUG_CFLAGS = -Wall -Werror -pedantic -ggdb3 -Wno-error=unknown-pragmas
#SRC1 = something1.cpp something2.cpp \
		something3.cpp
SRC2 = main.cpp
#SRC = $(SRC1) $(SRC2)
SRC = $(SRC2)
main: $(SRC2)

.PHONY: clean
clean:
	rm -f *.o main