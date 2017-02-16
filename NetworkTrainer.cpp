//Implementation for the NetworkTrainer class
#include<iostream>
#include <math.h>
#include <algorithm>
#include <vector>
#include<cassert>
//Include header
#include "NetworkTrainer.h"

#define SIZET (sizeof(float))

//constructor
NetworkTrainer::NetworkTrainer(ShallowNetwork * network, float lR, uint32_t nOE, uint32_t bS, float r, bool uV, uint32_t tS) : network(network), learningRate(lR), numberOfEpochs(nOE), batchSize(bS), regularizer(r), useValidation(uV), trainingSize(tS) {

  //allocate memory and initialize deltaBias vectors
  deltaHiddenBias = new float[network->localHiddenNeurons];

  for (uint32_t i = 0; i < network->localHiddenNeurons; ++i) {
    deltaHiddenBias[i] = 0;
  }

  deltaOutputBias = new float[network->outputNeurons];
  for (uint32_t i = 0; i < network->outputNeurons; ++i) {
    deltaOutputBias[i] = 0;
  }

  //allocate memory and initialize deltaWeight matrices
  deltaWeightInputHidden = new float[network->countElements];
  for (uint32_t i = 0; i < network->countElements; ++i) {
    deltaWeightInputHidden[i] = 0;
  }

  deltaWeightHiddenOutput = new float[network->outputNeurons * network->localHiddenNeurons];
  for (uint32_t i = 0; i < network->outputNeurons * network->localHiddenNeurons; ++i) {
    deltaWeightInputHidden[i] = 0;

  //allocate memory to store variables from other processors during backpropagation
  localStore = new float[network->hiddenNeurons - network->localHiddenNeurons];

  bsp_push_reg(localStore, ( (network->hiddenNeurons - network->localHiddenNeurons) * SIZET ));
  bsp_sync();
  }

  for (uint32_t i = 0; i < network->hiddenNeurons - network->localHiddenNeurons; ++i);

  //initialize vectors for monitoring progress
  trainingAccuracy = new float[nOE];
  validationAccuracy = new float[nOE];
  validationCost = new float[nOE];

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

//implementation of backpropagation algorithm
void NetworkTrainer::backpropagation(float * input, float label) {

  //feed forward
  network->feedForward(input);

  float * delta = new float[network->outputNeurons];


  for (uint32_t i = 0; i < network->outputNeurons; ++i) {
    delta[i] = (i == label) ? (network->output[i] - 1) : network->output[i];

    deltaOutputBias[i] += delta[i];
  }

  //compute error in the hidden-output weights
  for (uint32_t i = 0; i < network->outputNeurons; ++i) {
    for (uint32_t j = 0; j < network->localHiddenNeurons; ++j) {
      deltaWeightHiddenOutput[j + network->localHiddenNeurons * i] += network->hidden[j] * delta[i];
    }
  }

  float * deltaHidden = new float[network->localHiddenNeurons];
  bsp_push_reg(deltaHidden, network->localHiddenNeurons * SIZET);
  bsp_sync();

  //compute error of the hidden layers
  for (uint32_t i = 0; i < network->localHiddenNeurons; ++i) {
    deltaHidden[i] = 0;
    for (uint32_t j = 0; j < network->outputNeurons; ++j) {
      deltaHidden[i] += network->weightHiddenOutput[i + j * network->localHiddenNeurons] * delta[j];
    }
    deltaHidden[i] *= ( network->hidden[i] * (1 - network->hidden[i]) ); //using sigmoid activation function

    //add error in the hidden bias
    deltaHiddenBias[i] += delta[i];
  }

  //compute the error in the input-hidden weights

  //store the current element of the hidden vector for multiplication
  uint32_t count = 0;
  for (uint32_t i = 0; i < network->hiddenNeurons; ++i) {
    if (i % network->nProcessors == network->pId) { continue; }
    bsp_get(i % network->nProcessors, deltaHidden, (i / network->nProcessors) * SIZET, localStore + count, SIZET);
    ++count;
  }
  bsp_sync();

  //compute error in the hidden-input weights
  uint32_t currentI;
  uint32_t currentJ;
  uint32_t countR = 0;
  bool updateR = false;

  for (uint32_t i = 0; i < network->countI; ++i) {
    count = 0;
    currentI = network->matrixIndecesIH[i * network->countJ * 2];

    for (uint32_t j = 0; j < network->countJ; ++j) {

      currentJ = network->matrixIndecesIH[j * 2 + 1];
      if (currentI % network->nProcessors == network->pId) {
        if (currentJ % network->nProcessors == network->pId) {
          deltaWeightInputHidden[i * network->countJ + j] += deltaHidden[currentI / network->nProcessors] * network->input[currentJ / network->nProcessors];
        }
        else {
          deltaWeightInputHidden[i * network->countJ + j] += deltaHidden[currentI / network->nProcessors] * network->localStore[count];
          ++count;
        }
      }
      else {
        if (currentJ % network->nProcessors == network->pId) {
          deltaWeightInputHidden[i * network->countJ + j] += localStore[countR] * network->input[currentJ / network->nProcessors];
          updateR = true;
        }
        else {
          deltaWeightInputHidden[i * network->countJ + j] += localStore[countR] * network->localStore[count];
          ++count;
          updateR = true;
        }
      }
    }
    if (updateR) {
      ++countR;
    }
  }
}

//update network weights and biases
void NetworkTrainer::updateNetwork(float * trainingSet, float * labels, uint32_t currentBatchStart, uint32_t size) {

  //if the current batch is not the first one, reset all the errors in weights and biases to zero
  if (currentBatchStart) {
    resetWeightsAndBiases();
  }

  uint32_t stop = (currentBatchStart + batchSize > size) ? size : currentBatchStart + batchSize;

  for (uint32_t i = currentBatchStart; i < stop; ++i) {
    //backpropagation
    backpropagation(trainingSet + i * network->localInputNeurons,  labels[i]);

    //find the index of the neuron with the highest output value
    uint32_t indexMax = 0;
    float currentMax = 0;
    for (uint32_t i = 0; i < network->outputNeurons; ++i) {
      if (network->output[i] > currentMax) {
        currentMax = network->output[i];
        indexMax = i;
      }
    }

    //check output against label
    if (indexMax != labels[i]) {
      ++incorrectResults;
    }
  }

  //update weights
  float prefactor = learningRate / batchSize;
  float regularizationTerm = (1 - learningRate * regularizer / size);

  for (uint32_t i = 0; i < network->countElements; ++i) {
    network->weightInputHidden[i] = regularizationTerm * ( network->weightInputHidden[i] ) - prefactor * deltaWeightInputHidden[i];
  }

  for (uint32_t i = 0; i < network->outputNeurons * network->localHiddenNeurons; ++i) {
    network->weightHiddenOutput[i] = regularizationTerm * ( network->weightHiddenOutput[i] ) - prefactor * deltaWeightHiddenOutput[i];
  }

  //update biases
  for (uint32_t i = 0; i < network->localHiddenNeurons; ++i) {
    network->hiddenBias[i] -= prefactor * deltaHiddenBias[i];
  }

  for (uint32_t i = 0; i < network->outputNeurons; ++i) {
    network->outputBias[i] -= prefactor * deltaOutputBias[i];
  }
}

//implement stocastic gradient descent to train the network
void NetworkTrainer::stochasticGradientDescent(float * trainingSet, float * labels, uint32_t size) {

  //shuffle data in the training set
  random_shuffle(shuffleData.begin(), shuffleData.end());;

  incorrectResults = 0;

  //update network based on gradient descent on a mininbatch

  //this 'for loop' constitute an epoch
  for (uint32_t i = 0; i < size; i += batchSize) {
    updateNetwork(trainingSet, labels, i, size);
  }
}

//train the neural network
void NetworkTrainer::trainNetwork(float * trainingSet, float * labels, float * validationSet) {

  //initialize shuffleData
  shuffleData.resize(trainingSize);
  for (uint32_t i = 0; i < trainingSize; ++i) {
    shuffleData[i] = i;
  }
  if (network->pId == 0) {
    std::cout	<< std::endl << " Neural network ready to start training: " << std::endl
        << "==========================================================================" << std::endl
        << " LR: " << learningRate << ", Number of Epochs: " << numberOfEpochs << ", Batch size: " << batchSize << std::endl
        << " " << network->inputNeurons << " Input Neurons, " << network->hiddenNeurons << " Hidden Neurons, " << network->outputNeurons << " Output Neurons" << std::endl
        << "==========================================================================" << std::endl << std::endl;
  }
  //loop over the number of epochs
  for (uint32_t i = 0; i < numberOfEpochs; ++i) {

    //each call to stochasticGradientDescent perform an epoch of traning
    stochasticGradientDescent(trainingSet, labels, trainingSize);

    //store training accuracy for the epoch
    trainingAccuracy[i] = 100 - (incorrectResults/trainingSize * 100);

    if (useValidation && validationSet != NULL) {
      //validationAccuracy[i] = network->getAccuracyOfSet(validationSet);

      //validationCost[i] = monitorCost(validationSet);
    }

    //print accuracy for the epoch
    if (network->pId == 0) {
      std::cout << "==========================================================================" << std::endl
      << "Epoch: " << i + 1 << " of " << numberOfEpochs << std::endl
      << "Training set accuracy: " << trainingAccuracy[i] << std::endl;
      if (useValidation && validationSet != NULL) {
        std::cout << "Validation set accuracy: " << validationAccuracy[i] << " Total cost: " << validationCost[i] << std::endl;
      }
      std::cout << "==========================================================================" << std::endl;
    }
  }
}

//define cross-entropy cost function
float NetworkTrainer::crossEntropy(float * output, float label) {

  //vectorize the label in this particular case of two ouptut neurons
  //WARNING: This function is not general anymore
  std::vector<float> currentLabel(2);
  if(label == 0) {
    currentLabel[0] = 1;
    currentLabel[1] = 0;
  }
  else {
    currentLabel[0] = 0;
    currentLabel[1] = 1;
  }

  float cost = 0;
  for(uint32_t i = 0; i < network->outputNeurons; ++i) {
    cost += (-currentLabel[i] * log(output[i]) + (currentLabel[i] - 1) * log(1 - output[i]));
  }
  return cost;
}

/*
float NetworkTrainer::monitorCost(field< field<fvec> > * set) {

  float totalCost = 0;

  uint32_t size = set->n_elem;

  for (uint32_t i = 0; i < size; ++i) {
    network->feedForward(set->at(i)(0)); //feedforward to compute output
    totalCost += crossEntropy(network->output, set->at(i)(1));
  }
  return totalCost;
}
*/
void NetworkTrainer::resetWeightsAndBiases() {
  for (uint32_t i = 0; i < network->localHiddenNeurons; ++i) {
    deltaHiddenBias[i] = 0;
  }
  for (uint32_t i = 0; i < network->outputNeurons; ++i) {
    deltaOutputBias[i] = 0;
  }
  for (uint32_t i = 0; i < network->countElements; ++i) {
    deltaWeightInputHidden[i] = 0;
  }
  for (uint32_t i = 0; i < network->outputNeurons * network->localHiddenNeurons; ++i) {
    deltaWeightHiddenOutput[i] = 0;
  }
}
