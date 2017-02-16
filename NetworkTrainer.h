//Header for the NetworkTrainer class
#ifndef NETWORKTRAINER_H
#define NETWORKTRAINER_H
#include <vector>

#include "ShallowNetwork.h"

class NetworkTrainer {

  //Class members
  //----------------------------------------------------------------------------
private:
  //Network to train
  ShallowNetwork * network;

  //vectors storing changes in biases during gradient descent
  float * deltaHiddenBias;
  float * deltaOutputBias;

  //matrices storing cahnges in weights during gradient descent
  float * deltaWeightInputHidden;
  float * deltaWeightHiddenOutput;

  //learning parameters
  float learningRate;
  uint32_t numberOfEpochs;
  uint32_t batchSize;
  float regularizer;

  //bool monitorTrainingCost;
  bool useValidation;

  //monitor progress during learning
  float incorrectResults = 0;

  float * trainingAccuracy;
  float * validationAccuracy;
  float * validationCost;

  //store shuffled indeces to access data in random order
  std::vector<uint32_t> shuffleData;

  //number of training samples
  uint32_t trainingSize;

  //store variables from other processors during backpropagation
  float * localStore;

  //public Methods
  //----------------------------------------------------------------------------
public:

  //constructor with default values
  NetworkTrainer(ShallowNetwork * network, float learningRate = 0.01, uint32_t numberOfEpochs = 30, uint32_t miniBatchSize = 10, float regularizer = 0.0, bool useValidation = false, uint32_t trainingSize = 33000);

  //setters
  void setTrainingParameters(float learningRate, uint32_t numberOfEpochs, uint32_t batchSize, float regularizer = 0.0, bool useValidation = false);

  //getters for monitorig vector
  float * getTrainingAccuracy() const;
  float * getValidationAccuracy() const;
  float * getValidationCost() const;

  //network trainer
  void trainNetwork(float * trainingSet, float * labels, float * validationSet = NULL);

  //private Methods
  //----------------------------------------------------------------------------
private:

  void stochasticGradientDescent(float * trainingSet, float * labels, uint32_t size);
  void updateNetwork(float * trainingSet, float * labels, uint32_t currentBatchStart, uint32_t size);
  void backpropagation(float * input, float label);
  float crossEntropy(float * output, float label);
  //float monitorCost(float * set, float * labels); (yet to be implemented in this parallel version)
  void resetWeightsAndBiases();
};

#endif
