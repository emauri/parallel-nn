//implementation of the IsingDataLoader class

//include libraries
#include <iostream>
#include <fstream>
#include <cmath>

//include class declaration
#include "IsingDataLoader.h"

//constructor
IsingDataLoader::IsingDataLoader(uint32_t numberOfFiles, uint32_t latticeSize) : numberOfFiles(numberOfFiles), latticeSize(latticeSize) {
  set = new double[numberOfFiles * latticeSize];
  label = new double[numberOfFiles];
}

//destructor
IsingDataLoader::~IsingDataLoader() {

  //delete dataset
  delete[] set;
}

//Getter
double * IsingDataLoader::getDataSet() {
  return set;
}

//Read data from one file and initialize one elemnt of set, i.e. it initializes
//two vectors, one with the input configuration and one with the corresponding ouput label.
void IsingDataLoader::setData(std::string & fileName, uint32_t fileNumber) {
  //load the data in the file, the first line contain the size of the Ising lattice,
  //the second line contains the temperature of the system. From the third line
  //onwards the file contains the Ising spin configuration
  std::string path = "../Final Project/";
  std::ifstream file (path + fileName);

  if (file.is_open()) {
    std::string line;
    //skip first line
    std::getline(file, line);

    //Set the output label
    //----------------------------------------------------------------------------
    //If the temperature is less than 2.269 it is below the critical temperature: label = 0,
    //otherwise it is above: label = 1
    if ( std::getline(file, line) ) {
      label[fileNumber] = (std::stod(line) < 2.269) ? 0 : 1;
    }
    else {
      std::cout << "Something is wrong, the data are not being loaded" << std::endl;
    }

    //Set the input configuration
    //----------------------------------------------------------------------------
    for (uint32_t i = 0; i < latticeSize; ++i) {
      if ( std::getline(file, line) ) {
        set[fileNumber * latticeSize + i] = std::stod(line);
      }
      else {
        std::cout << "Something is wrong, the data are not being loaded" << std::endl;
      }
    }
  }
}

bool IsingDataLoader::loadData(uint32_t numberOfFiles, const char * listFile) {

  //check if all files are loaded
  bool allLoaded = true;

  //read data files names from file
  std::ifstream myfile (listFile);
  std::string fileName;
  if (myfile.is_open()) {
    for(uint32_t i = 0; i < numberOfFiles; ++i) {
      if (std::getline(myfile, fileName)) {
        setData(fileName, i);
      }
      else {
        allLoaded = false;
      }
    }
    if (std::getline(myfile, fileName)) {
      std::cout << "Warning: loaded less configurations than available." << std::endl;
      allLoaded = false;
    }
    myfile.close();
  }
  else {
    std::cout << "Couldn't open the file." << std::endl;
    allLoaded = false;
  }

  if (allLoaded) {
    std::cout << "Set properly loaded." << std::endl;
  } else {
    std::cout << "Unable to properly load the data set." << std::endl;
  }
  return allLoaded;
}
