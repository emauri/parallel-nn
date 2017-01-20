//Header for the ShallowNetwork class
#ifndef SHALLOWNETWORK_H
#define SHALLOWNETWORK_H
#include "mcbsp.hpp"
extern "C" {
  #include "mcbsp-affinity.h"
}

//friend class
class NetworkTrainer;

class ShallowNetwork {

  //Class members
  //----------------------------------------------------------------------------
private:

  //processor ID
  uint32_t pId;

  //Number of processors rows, columns and total
  uint32_t processorsRows, processorsCols, nProcessors;

  //Number of neurons in each layer
  uint32_t inputNeurons, hiddenNeurons, outputNeurons;

  //number of local neurons in the processor
  uint32_t localInputNeurons, localHiddenNeurons, localOutputNeurons;

  //Neurons layers as vectors
  double * hidden;
  double * output;

  //Neurons biases stored as vectors for each layer
  double * hiddenBias;
  double * outputBias;

  //Weight matrices within layers
  double * weightInputHidden;
  double * weightHiddenOutput;

  //store the indeces of the matrix elements contained in the processor
  uint32_t * matrixIndecesIH;

  //count matrix elements on the processor: rows, columns and total
  uint32_t countI, countJ, countElements;

  //store vector elements received during matix-vector multiplication
  double * localStore;

  //store results of the matrix vector multiplication
  double * partialResults;
  double * allResults;
  double * partialResultsOutput;
  double * allResultsOutput;

  //Public Methods
  //----------------------------------------------------------------------------
public:

  //Constructor with default values for data members
  ShallowNetwork(uint32_t pId, uint32_t processorsRows, uint32_t processorsCols, uint32_t inputNeurons = 2500, uint32_t hiddenNeurons = 1000, uint32_t outputNeurons = 2);

  //destructor
  ~ShallowNetwork();

  //getters and setters
  //arma::Col<uint32_t> getStructure() const;

  //As of now, I am not going to allow setters, initialize the network with the right values using the constructor.
  /*
  void setInputNeurons(uint32_t inputNeurons);
  void setHiddenNeurons(uint32_t hiddenNeurons);
  void setOutputNeurons(uint32_t outputNeurons);
  void setStructure(arma::Col<uint32_t> & structure);
*/

  //save to and load a network form the given directory. If used with no arguments it saves to and load from the same directory as the file.
  //bool saveNetwork(const char * directoryName = ".");
  //bool loadNetwork(const char * directoryName = ".");

  //get the result of the network evaluation;
  uint32_t getResult(double * input);

  //training evaluation
  //float getAccuracyOfSet(double * set);

  //Friends
	//--------------------------------------------------------------------------------------------
	friend NetworkTrainer;

  //Private methods
  //----------------------------------------------------------------------------
private:

  void initializeWeightsAndBiases();
  void activationFunction(double & input);
  void feedForward(double * input);
};

#endif
