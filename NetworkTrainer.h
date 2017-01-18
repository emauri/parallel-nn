//Header for the NetworkTrainer class
#ifndef NETWORKTRAINER_H
#define NETWORKTRAINER_H

#include<armadillo>

#include "ShallowNetwork.h"

class NetworkTrainer {

  //Class members
  //----------------------------------------------------------------------------
private:
  //Network to train
  ShallowNetwork * network;

  //vectors storing changes in biases during gradient descent
  arma::fvec deltaHiddenBias, deltaOutputBias;

  //matrices storing cahnges in weights during gradient descent
  arma::fmat deltaWeightInputHidden, deltaWeightHiddenOutput;

  //learning parameters
  float learningRate;
  uint32_t numberOfEpochs;
  uint32_t batchSize;
  float regularizer;

  //bool monitorTrainingCost;
  bool useValidation;

  //monitor progress during learning
  float incorrectResults = 0;

  arma::fvec trainingAccuracy;
  arma::fvec validationAccuracy;
  arma::fvec validationCost;

  //store shuffled indeces to access data in random order
  arma::Col<uint32_t> shuffleData;

  //public Methods
  //----------------------------------------------------------------------------
public:

  //constructor with default values
  NetworkTrainer(ShallowNetwork * network, float learningRate = 0.01, uint32_t numberOfEpochs = 30, uint32_t miniBatchSize = 10, float regularizer = 0.0, bool useValidation = false);

  //setters
  void setTrainingParameters(float learningRate, uint32_t numberOfEpochs, uint32_t batchSize, float regularizer = 0.0, bool useValidation = false);

  //getters for monitorig vector
  arma::fvec getTrainingAccuracy() const;
  arma::fvec getValidationAccuracy() const;
  arma::fvec getValidationCost() const;

  //network trainer
  void trainNetwork(arma::field< arma::field<arma::fvec> > * trainingSet, arma::field< arma::field<arma::fvec> > * validationSet = NULL);

  //private Methods
  //----------------------------------------------------------------------------
private:

  arma::fvec getOutputError(arma::fvec & output, arma::fvec & label);
  void stochasticGradientDescent(arma::field< arma::field<arma::fvec> > * trainingSet, uint32_t size);
  void updateNetwork(arma::field< arma::field<arma::fvec> > * trainingSet, uint32_t currentBatchStart, uint32_t size);
  void backpropagation(arma::fvec & input, arma::fvec & label);
  float crossEntropy(arma::fvec & output, arma::fvec & label);
  float monitorCost(arma::field< arma::field<arma::fvec> > * set);
};

#endif
