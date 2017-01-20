//Implementation for the ShallowNetwork class
#include<iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <cassert>

//Include header
#include "ShallowNetwork.h"

#define SIZET (sizeof(double))

//constructor
ShallowNetwork::ShallowNetwork(uint32_t pId, uint32_t pR, uint32_t pC, uint32_t iN, uint32_t hN, uint32_t oN) : pId(pId), processorsRows(pR), processorsCols(pC), inputNeurons(iN), hiddenNeurons(hN), outputNeurons(oN) {

  //total number of processors
  nProcessors = pR * pC;

  //determine the local number of neurons
  localInputNeurons = (iN < 10) ? iN : (iN + nProcessors - pId -1) / nProcessors;  //cyclic distribution

  localHiddenNeurons = (hN < 10) ? hN : (hN + nProcessors - pId -1) / nProcessors;  //cyclic distribution

  //No localOutputNeurons, since output neurons are stored entirely on each processors

  //allocate memory for storing vector elements received during matix-vector multiplication
  localStore = new double[hiddenNeurons - localHiddenNeurons];

  //allocate memory for storing partial results received from all the processors
  allResults = new double[localHiddenNeurons * nProcessors];

  allResultsOutput = new double[outputNeurons * nProcessors];

  //allocate memory for the layers and initialize them to zero
  hidden = new double[localHiddenNeurons];

  output = new double[localOutputNeurons];

  //initialize neurons to zero
  for (uint32_t i = 0; i < localHiddenNeurons; ++i) {
    hidden[i] = 0;
  }

  for (uint32_t i = 0; i < localOutputNeurons; ++i) {
    output[i] = 0;
  }

  //initialize biases vectors, weights matrices and related quantities
  initializeWeightsAndBiases();
}

//destructor
ShallowNetwork::~ShallowNetwork()
{
	//delete neurons
	delete[] hidden;
	delete[] output;

  //delete weight and biases
  delete[] weightInputHidden;
  delete[] weightHiddenOutput;
	delete[] hiddenBias;
  delete[] outputBias;

  //delete storing arrays
  delete[] matrixIndecesIH;
  delete[] localStore;
  delete[] partialResults;
  delete[] allResults;
  delete[] partialResultsOutput;
  delete[] allResultsOutput;
}


//biases initializer
void ShallowNetwork::initializeWeightsAndBiases() {

  //set the seed to a random value
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  //generate Normal distribution
  std::normal_distribution<float> distribution(0.0,1.0);

  //set biases
  //----------------------------------------------------------------------------

  //Allocate memory
  hiddenBias = new double[localHiddenNeurons];

  outputBias = new double[localOutputNeurons];

  //Biases initialization using Normal distribution
  for (uint32_t i = 0; i < localHiddenNeurons; ++i) {
    hiddenBias[i] = distribution(generator);
  }

  for (uint32_t i = 0; i < localOutputNeurons; ++i) {
    outputBias[i] = distribution(generator);
  }

  //Store matrix indices of elements in the processor for the weight of the Input-Hidden layer.
  //The row index and column index of an element are store contiguosly
  matrixIndecesIH = new uint32_t[2 * (uint32_t)std::ceil(inputNeurons * 1.0 / nProcessors) * (uint32_t)std::ceil(hiddenNeurons * 1.0 / nProcessors)];

  //keep trak of how many matrix elements are assigned to the processor (for the Input-Hidden matrix)
  countElements = 0;

  //initialize matrixIndeces
  for(uint32_t i = 0; i < inputNeurons; ++i) {
    for (uint32_t j = 0; j < hiddenNeurons; ++j) {
      if ( (i % processorsCols) + processorsRows * ((j % nProcessors) / processorsCols) == pId ) {
        matrixIndecesIH[countElements * 2] = i;
        matrixIndecesIH[countElements * 2 + 1] = j;
        ++countElements;
      }
    }
  }

  //count number of row elements (Js) and column elements (Is)
  uint32_t currentI = matrixIndecesIH[0];
  for (uint32_t i = 0; i < countElements; ++i) {
    if (currentI != matrixIndecesIH[i]) { break; }
    ++countJ;
  }
  countI = countElements / countJ;
  assert(countElements % countJ == 0);

  //allocate memory for storing partial results of the matrix-vector computations
  partialResults = new double[countI];

  partialResultsOutput = new double[outputNeurons];

  //Set weights between input and hidden layers
  //----------------------------------------------------------------------------

  //Allocate memory
  weightInputHidden = new double[countElements];

  //standard deviation of the Gaussian distribution
  float stdDev = 1.0 / (double)sqrt(inputNeurons);

  //initialize the weights and rescale them
  for (uint32_t i = 0; i < countElements; ++i) {
    weightInputHidden[i] = distribution(generator) * stdDev;
  }

  //set weights between hidden and output layers.
  //----------------------------------------------------------------------------

  //For the output layer, do to the small number of output neurons, we use
  //the cylclic column distribution to distribute the matrix elements

  //number of local elements for the cyclic column distribution
  uint32_t localSize = localHiddenNeurons * outputNeurons;

  //Allocate memory
  weightHiddenOutput = new double[localHiddenNeurons * outputNeurons];

  //Standard deviation of the Gaussian
  stdDev = 1 / (double)sqrt(hiddenNeurons);

  //Initialize the weights and rescale them
  for (uint32_t i = 0; i < localSize; ++i) {
    weightHiddenOutput[i] = distribution(generator) * stdDev;
  }
}

/*
//Save network configuration
bool ShallowNetwork::saveNetwork(const char * directoryName) {

  //convert directoryName to std::string
  std::string stringName = std::string(directoryName);

  //save weights matrices to .txt file in arma_binary format
  //----------------------------------------------------------------------------

  //save input-hidden weigths matrix
  bool ihWStatus = weightInputHidden.save(stringName + "/ih_weights.txt");

  //save hidden-output weights matrix
  bool hoWStatus = weightHiddenOutput.save(stringName + "/ho_weights.txt");

  //save biases vectors to .txt file in arma_binary format
  //----------------------------------------------------------------------------

  //save hidden layer biases vector
  bool hBStatus = hiddenBias.save(stringName + "/h_bias.txt");

  //save output layer biases vector
  bool oBStatus = outputBias.save(stringName + "/o_bias.txt");

  return (hoWStatus && ihWStatus && hBStatus && oBStatus);
}

//load network configuration
bool ShallowNetwork::loadNetwork(const char * directoryName) {

  //convert directoryName to std::string
  std::string stringName = std::string(directoryName);

  //load weights matrices from .txt file in arma_binary format
  //----------------------------------------------------------------------------

  //load input-hidden weigths matrix
  bool ihWStatus = weightInputHidden.load(stringName + "/ih_weights.txt", arma_binary);

  //load hidden-output weigths matrix
  bool hoWStatus = weightHiddenOutput.load(stringName + "/ho_weights.txt", arma_binary);

  //load biases vectors to .txt file in arma_binary format
  //----------------------------------------------------------------------------

  //load hidden layer biases vector
  bool hBStatus = hiddenBias.load(stringName + "/h_bias.txt", arma_binary);

  //load output layer biases vector
  bool oBStatus = outputBias.load(stringName + "/o_bias.txt", arma_binary);

  return (hoWStatus && ihWStatus && hBStatus && oBStatus);
}
*/

//Activation function
void ShallowNetwork::activationFunction(double & input) {
  //sigmoid function
  input = 1.0 / (1.0 + exp(-input));
}

//Feed Forward procedure
void ShallowNetwork::feedForward(double * input) {

  //calculate output from hidden layer
  //----------------------------------------------------------------------------

  //Superstep 0: Fanout
  uint32_t count = 0;
  for (uint32_t i = 0; i < countJ; ++i) {
    if ( matrixIndecesIH[i * 2 + 1] % nProcessors != pId) {
      ++count;
      bsp_get( (matrixIndecesIH[i * 2 + 1] % nProcessors), input + matrixIndecesIH[i * 2 + 1] / nProcessors, 0, localStore + count, SIZET);
    }
  } //localStore stores the vector elements not owned by the processors in increasing index order

  //Superstep 1: local matrix-vector multiplication
  uint32_t currentI;
  uint32_t currentJ;

  for (uint32_t i = 0; i < countI; ++i) {
    count = 0;
    currentI = matrixIndecesIH[i * countJ * 2];
    partialResults[i] = 0;
    for (uint32_t j = 0; j < countJ; ++j) {
      currentJ = matrixIndecesIH[j * 2 + 1];
      if (currentJ % nProcessors == pId) { //if the corresponding vector element is in the current processor
        partialResults[i] += weightInputHidden[i * countJ + j] * input[currentJ / nProcessors];
      }
      else { //the corresponding vector element was taken from another processor
        partialResults[i] += weightInputHidden[i * countJ + j] * localStore[count];
        ++count;
        assert(count <= countJ);
      }
    }
    //superstep 2: Fanin
    if (partialResults[i] != 0) {
      bsp_put(currentI % nProcessors, partialResults + i, (allResults + currentI * nProcessors + pId), 0, SIZET);
    }
  }
  bsp_sync();

  //superstep 3: Summation of partial sums
  for (uint32_t i = 0; i < localHiddenNeurons; ++i) {
    hidden[i] = 0;
    currentI = matrixIndecesIH[i * 2];
    for (uint32_t j = 0; j < nProcessors; ++j) {
      if (allResults[currentI + j]) {
        hidden[i] += allResults[currentI + j];
      }
    }
    hidden[i] += hiddenBias[i];
    activationFunction(hidden[i]);
  }

  //calculate output from hidden layer
  //----------------------------------------------------------------------------
  //The hidden-output matrix is stored in a column cyclic fashion
  //Fanout is not required

  //Superstep 0: local matrix-vector multiplication
  for (uint32_t i = 0; i < localHiddenNeurons; ++i) {
    for (uint32_t j = 0; j < localOutputNeurons; ++j) {
      partialResultsOutput[j] += hidden[i] * weightHiddenOutput[i  + j * localHiddenNeurons];
    }
  }

  //superstep 1: All-to-all communication. We want to store the output on every processor
  for (uint32_t i = 0; i < outputNeurons; ++i) {
    for (uint32_t j = 0; j < nProcessors; ++j) {
      if (partialResultsOutput[i]) {
        bsp_put(j, partialResultsOutput + i, (allResultsOutput + pId + i * nProcessors), 0, SIZET);
      }
    }
  }
  bsp_sync();

  //superstep 2: Summation of partial results
  for (uint32_t j = 0; j < outputNeurons; ++j) {
    output[j] = 0;
    for (uint32_t i = 0; i < nProcessors; ++i) {
      output[j] += allResultsOutput[i + j * nProcessors];
    }
    output[j] += outputBias[j];
    activationFunction(output[j]);
  }
}

//get the output neuron with the highest output value
uint32_t ShallowNetwork::getResult(double * input) {

  //feedforward input
  ShallowNetwork::feedForward(input);

  //find the index of the neuron with the highest output value
  uint32_t indexMax = 0;
  double currentMax = 0;
  for (uint32_t i = 0; i < outputNeurons; ++i) {
    if (output[i] > currentMax) {
      currentMax = output[i];
      indexMax = i;
    }
  }

  return indexMax;
}

/*
//accuracy on an input set
float ShallowNetwork::getAccuracyOfSet(double * set) {
  float incorrectResults = 0;

  //compare result of each input with corresponding label
  uint32_t size = set->n_elem;
  for (uint32_t i = 0; i < size; ++i) {
    if (getResult(set->at(i)(0)) != set->at(i)(1).index_max()) {
       ++incorrectResults;
    }
  }

  //return percentage of correct results
  return 100 - (incorrectResults / size * 100);
}
*/
