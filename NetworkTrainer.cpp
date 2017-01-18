//Implementation for the NetworkTrainer class
#include<iostream>
#include <math.h>
//Include header
#include "NetworkTrainer.h"

using namespace arma;

//constructor
NetworkTrainer::NetworkTrainer(ShallowNetwork * network, float lR, uint32_t nOE, uint32_t bS, float r, bool uV) : network(network), learningRate(lR), numberOfEpochs(nOE), batchSize(bS), regularizer(r), useValidation(uV) {

  //allocate memory and initialize deltaBias vectors
  deltaHiddenBias = new double[network->localHiddenNeurons];

  for (uint32_t i = 0; i < network->localHiddenNeurons; ++i) {
    deltaHiddenBias[i] = 0;
  }

  deltaOutputBias = new double[network->localOutputNeurons];
  for (uint32_t i = 0; i < network->localOutputNeurons; ++i) {
    deltaOutputBias[i] = 0;
  }

  //allocate memory and initialize deltaWeight matrices
  deltaWeightInputHidden = new double[network->localSizeIH];
  for (uint32_t i = 0; i < network->localSizeIH; ++i) {
    deltaWeightInputHidden[i] = 0;
  }

  deltaWeightHiddenOutput = new double[network->localSizeHO];
  for (uint32_t i = 0; i < network->localSizeHO; ++i) {
    deltaWeightInputHidden[i] = 0;
  }

  //initialize vectors for monitoring progress
  trainingAccuracy = new bool [nOE];
  for (uint32_t i = 0; i < nOE; ++i) {
    trainingAccuracy[i] = 0;
    if (uV) {
      validationAccuracy[i] = 0;
      validationCost[i] = 0;
    }
  }
}

//setters
void NetworkTrainer::setTrainingParameters(float learningRate, uint32_t numberOfEpochs, uint32_t batchSize, float regularizer, bool useValidation) {
  this->learningRate = learningRate;
  this->numberOfEpochs = numberOfEpochs;
  this->batchSize = batchSize;
  this->regularizer = regularizer;
  this->useValidation = useValidation;
}

//compute output error using cross-entropy cost function
float NetworkTrainer::getOutputError(float output, float label) {

  //return output error
  return (output - label);
}

//implementation of backpropagation algorithm
void NetworkTrainer::backpropagation(float * input, float * label) {

  //feed forward
  network->feedForward(input);

  float delta;

  for (uint32_t i = 0; i < network->outputNeurons; ++i) {
    deltaOutputBias[i] += getOutputError(networ->output[i], label[i]);
  }
    getOutputError(network->output, label);
  deltaOutputBias += delta;

  //compute error in the hidden-output weights
  deltaWeightHiddenOutput += (delta * network->hidden.t() );

  //compute error of the hidden layers
  delta = (network->weightHiddenOutput.t() *  delta) % ( (network->hidden) % (1 - network->hidden) ); //using sigmoid activation function

  //add error in the hidden biases
  deltaHiddenBias += delta;

  //compute error in the input-hidden weights
  deltaWeightInputHidden += (delta * input.t());
}

//update network weights and biases
void NetworkTrainer::updateNetwork(field< field<fvec> > * trainingSet, uint32_t currentBatchStart, uint32_t size) {

  //if the current batch is not the first one, reset all the errors in weights and biases to zero
  if (currentBatchStart) {
    deltaHiddenBias.zeros();
    deltaOutputBias.zeros();
    deltaWeightInputHidden.zeros();
    deltaWeightHiddenOutput.zeros();
  }

  uint32_t stop = (currentBatchStart + batchSize > size) ? size : currentBatchStart + batchSize;

  for (uint32_t i = currentBatchStart; i < stop; ++i) {

    //backpropagation
    backpropagation(trainingSet->at(shuffleData(i))(0), trainingSet->at(shuffleData(i))(1));
    //check output against label
    if (network->output.index_max() != trainingSet->at(shuffleData(i))(1).index_max()) {
      ++incorrectResults;
    }
  }

  //update weights
  float prefactor = learningRate / batchSize;
  float regularizationTerm = (1 - learningRate * regularizer / size);

  network->weightInputHidden = regularizationTerm * ( network->weightInputHidden ) - prefactor * deltaWeightInputHidden;

  network->weightHiddenOutput = regularizationTerm * ( network->weightHiddenOutput ) - prefactor * deltaWeightHiddenOutput;

  //update biases
  network->hiddenBias -= prefactor * deltaHiddenBias;

  network->outputBias -= prefactor * deltaOutputBias;
}

//implement stocastic gradient descent to train the network
void NetworkTrainer::stochasticGradientDescent(field< field<fvec> > * trainingSet, uint32_t size) {

  //shuffle data in the training set
  shuffleData = shuffle(shuffleData);

  incorrectResults = 0;

  //update network based on gradient descent on a mininbatch

  //this 'for loop' constitute an epoch
  for (uint32_t i = 0; i < size; i += batchSize) {
    updateNetwork(trainingSet, i, size);
  }
}

//train the neural network
void NetworkTrainer::trainNetwork(field< field <fvec> > * trainingSet, field < field<fvec> > * validationSet) {

  //initialize shuffleData
  uint32_t trainingSize = trainingSet->n_elem;
  shuffleData.set_size(trainingSize);
  for (uint32_t i = 0; i < trainingSize; ++i) {
    shuffleData(i) = i;
  }
  std::cout	<< std::endl << " Neural network ready to start training: " << std::endl
			<< "==========================================================================" << std::endl
			<< " LR: " << learningRate << ", Number of Epochs: " << numberOfEpochs << ", Batch size: " << batchSize << std::endl
			<< " " << network->inputNeurons << " Input Neurons, " << network->hiddenNeurons << " Hidden Neurons, " << network->outputNeurons << " Output Neurons" << std::endl
			<< "==========================================================================" << std::endl << std::endl;

  //loop over the number of epochs
  for (uint32_t i = 0; i < numberOfEpochs; ++i) {

    //each call to stochasticGradientDescent perform an epoch of traning
    stochasticGradientDescent(trainingSet, trainingSize);

    //store training accuracy for the epoch
    trainingAccuracy(i) = 100 - (incorrectResults/trainingSize * 100);

    if (useValidation && validationSet != NULL) {
      validationAccuracy(i) = network->getAccuracyOfSet(validationSet);

      validationCost(i) = monitorCost(validationSet);
    }

    //print accuracy for the epoch
    std::cout << "==========================================================================" << std::endl
    << "Epoch: " << i + 1 << " of " << numberOfEpochs << std::endl
    << "Training set accuracy: " << trainingAccuracy(i) << std::endl;
    if (useValidation && validationSet != NULL) {
      std::cout << "Validation set accuracy: " << validationAccuracy(i) << " Total cost: " << validationCost(i) << std::endl;
    }
    std::cout << "==========================================================================" << std::endl;

  }
}

//define cross-entropy cost function
float NetworkTrainer::crossEntropy(fvec & output, fvec & label) {

  log(ouput).print();
  label.print();

  return accu(-label % log(output) + (label - 1) % log(1 - output));
}

float NetworkTrainer::monitorCost(field< field<fvec> > * set) {

  float totalCost = 0;

  uint32_t size = set->n_elem;

  for (uint32_t i = 0; i < size; ++i) {
    network->feedForward(set->at(i)(0)); //feedforward to compute output
    totalCost += crossEntropy(network->output, set->at(i)(1));
  }
  return totalCost;
}
