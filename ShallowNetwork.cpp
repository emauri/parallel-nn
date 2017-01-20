//Implementation for the ShallowNetwork class
#include<iostream>
#include <fstream>
#include <math.h>
//Include header
#include "ShallowNetwork.h"

//constructor
ShallowNetwork::ShallowNetwork(uint32_t pId, uint32_t pR, uint32_t pC, uint32_t iN, uint32_t hN1, uint32_t hN2, uint32_t oN) : pId(pId), processorsRows(pR), processorsCols(pC), inputNeurons(iN), hiddenNeurons(hN), outputNeurons(oN) {

  //total number of processors
  nProcessors = pR * pC;

  //determine the local number of neurons
  localInputNeurons = (iN < 10) ? iN : (iN + nP - pId -1) / nP;

  localHiddenNeurons = (hN < 10) ? hN : (hN + nP - pId -1) / nP;

  localOutputNeurons = (oN < 10) ? oN : (oN + nP - pId -1) / nP;

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

  //initialize biases vectors and weights matrices
  initializeWeightsAndBiases();
}

//biases initializer
void ShallowNetwork::initializeWeightAndBiases() {

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
  matrixIndecesIH = new uint32_t[2 * std::ceil(inputNeurons) / nProcessors * std::ceil(hiddenNeurons) / nProcessors];

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

  //number of local elements for the cyclic distribution
  uint32_t localSize = localHiddenNeurons * outputNeurons

  //Allocate memory
  weightHiddenOtput = new double[localHiddenNeurons * outputNeurons];

  //Standard deviation of the Gaussian
  stdDev = 1 / (double)sqrt(hiddenNeurons);

  //Initialize the weights and rescale them
  for (uint32_t i = 0; i < localSize; ++i) {
    weightHiddenOtput[i] = distribution(generator) * stdDev;
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
void ShallowNetwork::activationFunction(double * input, uint32_t inputSize) {

  //sigmoid function
  for (uint32_t i = 0; i < inputSize; ++i) {
    input[i] = 1.0 / (1.0 + exp(-input[i]));
  }
}

//Feed Forward procedure
void ShallowNetwork::feedForward(double * input) {

  //calculate output from hidden layer
  //----------------------------------------------------------------------------

  //Superstep 0: Fanout
  uint32_t count;
  for (uint32_t i = 0; i < countElements; ++i) {
    if ( matrixIndecesIH[i * 2 + 1] % nProcessors != pId) {
      ++count;
    }
    bsp_get( (matrixIndecesIH[i * 2 + 1] % nProcessors), hidden + matrixIndecesIH[i * 2 + 1] / nProcessors, 0, localStore + count, SIZET);
  }

  //local matrix-vector multiplication
  for (uint32_t i = 0; i < countElements; ++i) {
    partialResults[i] = 0;
    for (uint32_t i = 0; i < )
  }


  //calculate output from hidden layer
  //----------------------------------------------------------------------------
  for (uint32_t i = 0; i < hiddenNeurons; ++i) {
    hidded[i] = 0;
    for (uint32_t j = 0; j < inputNeurons; ++j) {
      hidden[i] += weightInputHidden[i * input + j] * input[j];
    }

    hidden[i] = activationFunction(hidden[i]);
  }

  //calculate output
  //----------------------------------------------------------------------------
  for (uint32_t i = 0; i < outputNeurons; ++i) {
    output[i] = 0;
    for (uint32_t j = 0; j < inputNeurons; ++j) {
      output[i] += weightHiddenOutput[i * input + j] * input[j];
    }

    output[i] = activationFunction(ouput[i]);
  }
}

//get the output neuron with the highest output value
uint32_t ShallowNetwork::getResult(float * input) {

  //feedforward input
  ShallowNetwork::feedForward(input);

  //find the index of the neuron with the highest output value
  uint32_t indexMax = 0;
  float currentMax = 0;
  for (uint32_t i = 0; i < outputNeurons; ++i) {
    if (ouput[i] > currentMax) {
      currentMax = output[i];
      indexMax = i;
    }
  }

  return indexMax;
}

//accuracy on an input set
float ShallowNetwork::getAccuracyOfSet(field< field<fvec> > * set) {
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
