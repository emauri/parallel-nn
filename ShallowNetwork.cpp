//Implementation for the ShallowNetwork class
#include<iostream>
#include <fstream>
#include <math.h>
//Include header
#include "ShallowNetwork.h"

//constructor
ShallowNetwork::ShallowNetwork(uint32_t nP, iN, uint32_t hN1, uint32_t hN2, uint32_t oN) : nProcessors(nP), inputNeurons(iN), hiddenNeurons(hN), outputNeurons(oN) {

  //determine the local number of neurons
  localInputNeurons = (iN < 10) ? iN : std::ceil(iN * 1.0 / nP);

  localHiddenNeurons = (hN < 10) ? hN : std::ceil(hN * 1.0 / nP);

  localOutputNeurons = (oN < 10) ? oN : std::ceil(oN * 1.0 / nP);

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

  //Set weights between input and hidden layers
  //----------------------------------------------------------------------------
  uint32_t localSize = std::ceil( std::max(inputNeurons, hiddenNeurons) * std::max(localInputNeurons, localHiddenNeurons) ); // VERIFY THIS

  //Allocate memory
  weightInputHidden = new double[localSize];

  //standard deviation of the Gaussian distribution
  float stdDev = 1.0 / (double)sqrt(inputNeurons);

  //initialize the weights and rescale them
  for (uint32_t i = 0; i < localSize; ++i) {
    weightInputHidden[i] = distribution(generator) * stdDev;
  }

  //set weights between input and hidden layers
  //----------------------------------------------------------------------------
  //CHECK THE MAXIMUM SIZE OF THE MATRX
  localSize = std::ceil( std::max(outputNeurons, hiddenNeurons) * std::max(localOutputNeurons, localHiddenNeurons) ); // VERIFY THIS

  //Allocate memory
  weightHiddenOtput = new double[localSize];

  //Standard deviation of the Gaussian
  stdDev = 1 / (double)sqrt(hiddenNeurons);

  //Initialize the weights and rescale them
  for (uint32_t i = 0; i < localSize; ++i) {
    weightHiddenOtput[i] = distribution(generator) * stdDev;
  }
}

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

  //save biases vectors to .txt file in arma_binary format
  //----------------------------------------------------------------------------

  //load hidden layer biases vector
  bool hBStatus = hiddenBias.load(stringName + "/h_bias.txt", arma_binary);

  //load output layer biases vector
  bool oBStatus = outputBias.load(stringName + "/o_bias.txt", arma_binary);

  return (hoWStatus && ihWStatus && hBStatus && oBStatus);
}

//Activation function
void ShallowNetwork::activationFunction(double * input, uint32_t inputSize) {

  //sigmoid function
  for (uint32_t i = 0; i < inputSize; ++i) {
    input[i] = 1.0 / (1.0 + exp(-input[i]));
  }
}

//Feed Forward procedure
void ShallowNetwork::feedForward(double * input) {

  assert(M * N = nProcessors);
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
