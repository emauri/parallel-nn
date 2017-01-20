#include <iostream>
#include <chrono>
#include <cstdint>
#include "mcbsp.hpp"
extern "C" {
  #include "mcbsp-affinity.h"
}
#define SIZET (sizeof(double))

#include "NetworkTrainer.h"
#include "IsingDataLoader.h"
//#include "cblas.h"

//rows of processors
uint32_t M;

//column of processors
uint32_t N;

//total number of processors
uint32_t nCores = M * N;

void parallelNN() {

  //Initiate parallel program on n_cores
  bsp_begin(nCores);
  uint32_t pId = bsp_pid();  //processors id number

  //define network parameters
  uint32_t inputNeurons = 2500;
  uint32_t hiddenNeurons = 2000;
  uint32_t outputNeurons = 2;

  //define training set parameters
  uint32_t setSize = 33000;

  //network declaration;
  ShallowNetwork(inputNeurons, hiddenNeurons, outputNeurons, nCores);

  //allocate memory for the input array and label
  //----------------------------------------------------------------------------

  //local number of components using cyclic distribution of the input configuration
  uint32_t localInputNeurons = (inputNeurons + nCores - pId -1) / nCores;
  double * input = new double[localInputNeurons * setSize];
  double * label = new double[setSize];

  //register memory
  bsp_push_reg(input, SIZET * localInputNeurons);
  bsp_push_reg(label, SIZET * setSize);
  bsp_sync();

  //load data sets
  //----------------------------------------------------------------------------
  IsingDataLoader training;

  //load data on processor 0 and distribute them cyclically to all the other processors
  //----------------------------------------------------------------------------
  if (pid == 0) {
    if ( !training.loadData(setSize, "../Final Project/dataParallel/trainingParallel_00.txt") ) { return; };

    double * set = training.getDataSet();

    label = training.getLabels();

    //put all the labels in each processor
    for (uint32_t i  = 0; i < nCores; ++i) {
      bsp_put(i, label, label, SIZET * setSize);
    }

    //distribute the input vector cyclically on each processor
    for (uint32_t i = 0; i < setSize) {
      for (uint32_t j = 0; j < inputNeurons; ++j) {
              bsp_put(j % nCores, (set + j + i * inputNeurons), (input + j + (inputNeurons + nCores - (j % nCores) -1) / nCores * i), 0, SIZET);
              //(inputNeurons + nCores - (j % nCores) -1) / nCores gives the number of elements for each input on processor (j % nCores)
      }
    }
  }
  bsp_sync();

  //free memory for set
  bsp_pop_reg(set);
  delete[] set;

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
