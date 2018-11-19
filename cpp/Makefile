CC = g++
CFLAGS = -g -O2 -Wall
SRCS = main.cpp InfiniteHorizonGP.cpp Matern32model.cpp
PROG = main

EIGEN = `pkg-config --libs --cflags eigen3`
LIBS = $(EIGEN)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
