//Header for the ShallowNetwork class
#ifndef SHALLOWNETWORK_H
#define SHALLOWNETWORK_H
#include<armadillo>

//friend class
class NetworkTrainer;

class ShallowNetwork {

  //Class members
  //----------------------------------------------------------------------------
private:

  //Number of processors
  uint32_t nProcessors;

  //Number of neurons in each layer
  uint32_t inputNeurons, hiddenNeurons, outputNeurons;

  //number of local neurons in each processor
  uint32_t localInputNeurons, localHiddenNeurons, localOutputNeurons;

  //Neurons layers as vectors
  double * hidden, output;

  //Neurons biases stored as vectors for each layer
  double * hiddenBias, outputBias;

  //Weight matrices within layers
  double * weightInputHidden, weightHiddenOutput;

  //Public Methods
  //----------------------------------------------------------------------------
public:

  //Constructor with default values for data members
  ShallowNetwork(inputNeurons = 2500, uint32_t hiddenNeurons = 1000, uint32_t outputNeurons = 2, uint32_t nProcessors = 1);

  //getters and setters
  arma::Col<uint32_t> getStructure() const;

  //As of now, I am not going to allow setters, initialize the network with the right values using the constructor.
  /*
  void setInputNeurons(uint32_t inputNeurons);
  void setHiddenNeurons(uint32_t hiddenNeurons);
  void setOutputNeurons(uint32_t outputNeurons);
  void setStructure(arma::Col<uint32_t> & structure);
*/

  //save to and load a network form the given directory. If used with no arguments it saves to and load from the same directory as the file.
  bool saveNetwork(const char * directoryName = ".");
  bool loadNetwork(const char * directoryName = ".");

  //get the result of the network evaluation;
  uint32_t getResult(double * input, uint32_t inputSize = 2500);

  //training evaluation
  float getAccuracyOfSet(double * set);

  //Friends
	//--------------------------------------------------------------------------------------------
	friend NetworkTrainer;

  //Private methods
  //----------------------------------------------------------------------------
private:

  void initializeBiases();
  void initializeWeights();
  void activationFunction(double * input);
  void feedForward(double * input);
};

#endif