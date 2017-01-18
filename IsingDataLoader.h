//Header for a simple IsingDataLoader class, to be improved later
#ifndef ISINGDATALOADER_H
#define ISINGDATALOADER_H
#include<vector>
#include<string>
#include "mcbsp.hpp"
extern "C" {
  #include "mcbsp-affinity.h"
}
#define SIZET (sizeof(double))

class IsingDataLoader {

  //Class members
  //----------------------------------------------------------------------------
private:

  uint32_t numberOfFiles;
  uint32_t latticeSize;
  //store all the configurations contiguosly
  double * set;

  //store all the labels
  double * label;

  //Public methods
  //----------------------------------------------------------------------------
public:

  //constructor
  IsingDataLoader(uint32_t numberOfFiles = 33000, uint32_t latticeSize = 2500);

  //Destructor
  ~IsingDataLoader();

  //Load the data set.
  bool loadData(uint32_t numberOfFiles, const char * listFile);

  //load saved data set
  bool loadData(const char * fileName);

  //Getter for the data set. Return a pointer to the loaded dat set
  double * getDataSet();

  //Private methods
  //----------------------------------------------------------------------------
private:

  //initialize one element of the set with input data and output label
  void setData(std::string & fileName, uint32_t fileNumber);
};

#endif
