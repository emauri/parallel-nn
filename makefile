OBJS = main.o ShallowNetwork.o IsingDataLoader.o NetworkTrainer.o
CC = g++ --std=c++11
OPTIMIZE = -O3
CFLAGS = -Wall -c $(OPTIMIZE)
LFLAGS = -Wall $(OPTIMIZE)

LIBS = ../MulticoreBSP/lib/libmcbsp1.2.0.a -I../MulticoreBSP
LINK = -I../MulticoreBSP

main : $(OBJS)
		$(CC) $(LFLAGS) $(OBJS) -o main $(LIBS)

main.o : main.cpp ShallowNetwork.h IsingDataLoader.h NetworkTrainer.h
		$(CC) $(CFLAGS) main.cpp $(LINK)

ShallowNetwork.o : ShallowNetwork.h ShallowNetwork.cpp NetworkTrainer.h
		$(CC) $(CFLAGS) ShallowNetwork.cpp $(LINK)

IsingDataLoader.o : IsingDataLoader.h IsingDataLoader.cpp
		$(CC) $(CFLAGS) IsingDataLoader.cpp $(LINK)

NetworkTrainer.o : NetworkTrainer.h NetworkTrainer.cpp ShallowNetwork.h
		$(CC) $(CFLAGS) NetworkTrainer.cpp $(LINK)

clean:
		\rm *.o main
