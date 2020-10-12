#include <iostream>
#include <fstream>
#include <string>
#include "Net.hpp"
#include <algorithm>
#include <random>

/// Really messy func just to pull inputs and labels for training example
void getSampleData(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& labels)
{
  std::string line;
  std::ifstream datafile ("iris.data");
  if(!datafile.is_open())
  {
    std::cout << "Unable to open file" << std::endl;
    return;
  }
  int line_num = 1;
  while ( std::getline(datafile, line) )
  {
    inputs.emplace_back(std::vector<double>());
    // explode string
    // for num inputs
    for(int i = 0; i < 4; ++i)
    {
      std::string value = line.substr(0, line.find(','));
      line.erase(0, line.find(',') + 1);
      try
      {
        inputs.back().emplace_back(std::stod(value)); // TODO: Fix
      }
      catch (std::exception& ex)
      {
        std::cout << "Bad file read @ line " << line_num << std::endl;
      }
    }
    
    if(line == "Iris-setosa")
    {
      labels.emplace_back(std::vector<double>() = { 1, 0, 0 });
    }
    else if(line == "Iris-versicolor")
    {
      labels.emplace_back(std::vector<double>() = { 0, 1, 0 });
    }
    else if(line == "Iris-virginica")
    {
      labels.emplace_back(std::vector<double>() = { 0, 0, 1 });
    }
    else
    {
      std::cout << "Label Missing @ line " << line_num << std::endl;
    }
    line_num += 1;
  }
}

int main()
{
  // Using deterministic randomness for the sake of debugging
  srand(0);
  
  std::vector<int> topology = { 4, 4, 4, 3 };
  Net* my_net = new Net(topology);
  
  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> labels;
  getSampleData(inputs, labels);
  
  my_net->run(inputs, labels);
  
  my_net->save("demo_net");
  Net* clone = new Net("demo_net");
  
  std::cout << "breakpoint" << std::endl;
  delete my_net;
  delete clone;
  return 0;
}

