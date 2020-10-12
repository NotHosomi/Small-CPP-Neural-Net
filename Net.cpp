//
// Created by Hosomi on 02/10/2020.
//

#include "Net.hpp"
#include <iostream>
#include <math.h>       /* sqrt */
#include <fstream>


/***********
 *  BUILD  *
 ***********/
Net::Net(std::vector<int> topology)
{
  for(int i = 0; i < topology.size(); ++i)
  {
    layers.emplace_back(std::vector<Neuron>());
    int num_outputs = (i == topology.size() - 1) ? 0 : topology[i+1];
    
    for(int n = 0; n < topology[i]; ++n)
    {
      layers.back().emplace_back(num_outputs);
      std::cout << "Added a neuron! L:" << i << "\t N:" << n << std::endl;
    }
  }
}

/******************
 *  FEED FORWARD  *
 ******************/
void Net::feedForward(const std::vector<double>& inputs)
{
  // set the input values
  for(int i = 0; i < inputs.size(); ++i)
  {
    layers[0][i].setActivation(inputs[i]);
  }
  
  /// run through
  // for each layer
  for(int i = 1; i < layers.size(); ++i)
  {
    std::vector<Neuron>& prev_layer = layers[i - 1];
    
    // for each neuron
    for(int n = 0; n < layers[i].size(); ++n)
    {
      double sum = 0;
      for(Neuron& in : prev_layer)
      {
        sum += in.getActivation() * in.getConnectionWeight(n);
      }
      // set my activation value
      layers[i][n].setActivation(Neuron::activationFunction(sum));
    }
  }
  /// DEBUG
  //std::cout << "O: ";
  //for(Neuron& n : layers.back())
  //{
  //  std::cout << std::to_string(n.getActivation()) << ", ";
  //}
  //std::cout << std::endl;
}

/**********************
 *  BACK PROPAGATION  *
 **********************/
void Net::backProp(const std::vector<double> &targets)
{
  // ref: https://www.youtube.com/watch?v=tIeHLnjs5U8
  std::vector<Neuron>& outputs = layers.back();
  
  for(int n = 0; n < outputs.size(); ++n)
  {
    double diff = targets[n] - outputs[n].getActivation();
    diff *= Neuron::activationFuncDerivative(outputs[n].getActivation());
    outputs[n].addGradient(diff);   // TODO: look into this more
  }
  
  // calculate gradient across hidden layers
  for(int L = layers.size() - 2; L > 0; --L)
  {
    for(Neuron& k : layers[L]) // 'K' as per calculus notation
    {
      k.calcGradient(layers[L+1]);
    }
  }
  
  // use gradients to update weights (Can be done in batches)
  for(int L = layers.size() - 1; L > 0; --L)
  {
    for(int j = 0; j < layers[L].size(); ++j)
    {
      for(Neuron& k : layers[L - 1])
      {
        k.updateWeight(j, layers[L][j].getCurrentGradient()); // use Sum? Or Average?
      }
    }
  }
  //clear gradients (for stochastic)
  for(std::vector<Neuron>& layer : layers)
  {
    for(Neuron& neuron : layer)
    {
      neuron.clearGradients();
    }
  }
}


/*****************
 *  OTHER UTILS  *
 *****************/
void Net::evaluate(const std::vector<double> &targets)
{
  std::cout << "O: ";
  for(Neuron& n : layers.back())
  {
    std::cout << std::to_string(n.getActivation()) << ", ";
  }
  std::cout << std::endl;
  std::cout << "T: ";
  for(double value : targets)
  {
    std::cout << std::to_string(value) << ", ";
  }
  std::cout << std::endl;
  
  /// track the recent average error:
  double error = 0.0;
  for (unsigned n = 0; n < layers.back().size() - 1; ++n)
  {
    double delta = targets[n] - layers.back()[n].getActivation();
    error += delta * delta;
  }
  error /= layers.back().size() - 1; // Avrg of error squared
  error = sqrt(error); // RMS
  recentAverageError =
          (recentAverageError * recentAverageSmoothingFactor + error)
          / (recentAverageSmoothingFactor + 1);
}

void Net::run(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>>& labels)
{
  // catch problems
  if(inputs[0].size() != layers[0].size())
  {
    std::cout << "ERROR: input dimensions mismatch" << std::endl;
    return;
  }
  if(labels[0].size() != layers.back().size())
  {
    std::cout << "ERROR: label dimensions mismatch" << std::endl;
    return;
  }
  
  // train
  for(int i = 0; i < inputs.size(); ++i)
  {
    feedForward(inputs[i]);
    evaluate(labels[i]);
    backProp(labels[i]);
  }
  
  //conclude
  std::cout << "Recent Average Error: " << recentAverageError << std::endl;
}

/// PERSISTENCE

void Net::save(const std::string &filename) const
{
  std::ofstream file;
  file.open("nets/" + filename + ".dat", std::ios::binary | std::ios::out | std::ios::trunc);
  if(!file)
  {
    std::cout << "Failed to save file " << filename << ".dat" << std::endl;
    return;
  }
  
  /// Encode Topology
  // num layers
  auto num_layers = static_cast<unsigned int>(layers.size());
  file.write(reinterpret_cast<char*>(&num_layers), sizeof(unsigned int));
  // layer dimensions
  for (auto& layer : layers)
  {
    auto num_neurons = static_cast<unsigned int>(layer.size());
    file.write(reinterpret_cast<char*>(&num_neurons), sizeof(unsigned int));
  }
  /// Encode Connections
  // layer dimensions
  for (int L = 0; L < layers.size() - 1; ++L)
  {
    for (auto& k : layers[L])
    {
      for (int j = 0; j < layers[L + 1].size(); ++j)
      {
        double w = k.getConnectionWeight(j);
        file.write(reinterpret_cast<char*>(&w), sizeof(double));
      }
      double bias = k.getBias();
      file.write(reinterpret_cast<char*>(&bias), sizeof(double));
    }
  }
  file.close();
  std::cout << "File saved!" << std::endl;
}

Net::Net(const std::string &filename)
{
  std::ifstream file;
  file.open("nets/" + filename + ".dat", std::ios::binary | std::ios::in);
  if(!file)
  {
    std::cout << "Failed to open file " << filename << ".dat" << std::endl;
    return;
  }
  
  /// Load Topology
  unsigned int num_layers = 0;
  file.read(reinterpret_cast<char*>(&num_layers), sizeof(unsigned int));
  //layers.reserve(num_layers);
  std::cout << "Num layers: " << num_layers;
  // get layer sizes
  std::vector<int> topology;
  for(int l = 0; l < num_layers; ++l)
  {
    int layer_size = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(unsigned int));
    topology.emplace_back(num_layers);
  }
  
  
  for(int i = 0; i < topology.size(); ++i)
  {
    layers.emplace_back(std::vector<Neuron>());
    int num_outputs = (i == topology.size() - 1) ? 0 : topology[i+1];
    
    for(int k = 0; k < topology[i]; ++k)
    {
      std::cout << "Added a neuron! L:" << i << "\t N:" << k << std::endl;
      std::vector<double> connections;
      for(int j = 0; j < num_outputs; ++j)
      {
        double weight = 0;
        file.read(reinterpret_cast<char*>(&weight), sizeof(double));
        connections.emplace_back(weight);
      }
      double bias = 0;
      file.read(reinterpret_cast<char*>(&bias), sizeof(double));
      
      layers.back().emplace_back(connections, bias);
    }
  }
  std::cout << std::endl;
  file.close();
  std::cout << "Net loaded!" << std::endl;
}













