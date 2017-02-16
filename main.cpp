#include <iostream>
#include <chrono>
#include <cstdint>
#include "mcbsp.hpp"
extern "C" {
  #include "mcbsp-affinity.h"
}
#define SIZET (sizeof(float))

#include "NetworkTrainer.h"
#include "IsingDataLoader.h"
//#include "cblas.h"

//rows of processors
uint32_t M;

//column of processors
uint32_t N;

//total number of processors
uint32_t nProcessors;

void parallelNN() {
  //Initiate parallel program on nProcessors
  bsp_begin(M * N);
  nProcessors = M * N;
  uint32_t pId = bsp_pid();  //processor id number

  //define network parameters
  uint32_t inputNeurons = 2500;
  uint32_t hiddenNeurons = 100;
  uint32_t outputNeurons = 2;

  //define training set parameters
  uint32_t setSize = 33000;

  //allocate memory for the input array and label
  //----------------------------------------------------------------------------

  //local number of components using cyclic distribution of the input configuration
  uint32_t localInputNeurons = (inputNeurons + nProcessors - pId - 1) / nProcessors;
  float * input = new float[localInputNeurons * setSize];

  //register memory
  bsp_push_reg(input, SIZET * localInputNeurons * setSize);
  bsp_sync();

  //load data sets
  //----------------------------------------------------------------------------
  IsingDataLoader training;

  //load data and distribute them cyclically
  //----------------------------------------------------------------------------
  char fileName[100];
  char filePath[100];
  sprintf(filePath, "../Final Project/dataParallel/trainingParallel");
  char fileNumber[10];
  sprintf(fileNumber, "_%02d.txt", pId);
  strcpy(fileName, filePath);
  strcat(fileName, fileNumber);
  if ( !training.loadData(setSize, fileName) ) { return; };

  float * set = training.getDataSet();

  float * labels = training.getLabels();

  //distribute the input vector cyclically on each processor

  for (uint32_t i = 0; i < setSize; ++i) {
    for (uint32_t j = 0; j < inputNeurons; ++j) {
      if (j % nProcessors == pId) {
        input[j / nProcessors + i * localInputNeurons] = set[j + i * inputNeurons];
      }
    }
  }

  delete[] set;
  //network declaration;
  ShallowNetwork network(pId, M, N, inputNeurons, hiddenNeurons, outputNeurons);
  NetworkTrainer trainer(&network, 0.02, 20, 10, 0.5, false, setSize);

  //train the network
  //----------------------------------------------------------------------------

  //measure training time
  auto t1 = std::chrono::high_resolution_clock::now();

  //training
  trainer.trainNetwork(input , labels);

  auto t2 = std::chrono::high_resolution_clock::now();

  //print training time
  if (pId == 0) {
    std::cout << "Training took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()/1000 << " seconds.\n";
  }

  //save network weights and biases
  //network.saveNetwork("data");

  //test network accuracy
  //----------------------------------------------------------------------------
  //std::cout << "Test data accuracy: " << network.getAccuracyOfSet( test.getDataSet() ) << std::endl;

  bsp_end();
}

int main(int argc, char * argv[]) {

  bsp_init(parallelNN, argc, argv);
  mcbsp_set_affinity_mode(COMPACT);

  //Set number of cores
  M = 2;
  N = 2;

  //call the parallel program

  parallelNN();

  return 0;
}
