//
// Created by Hosomi on 02/10/2020.
//
#pragma once

#include <vector>
#include <string>
#include "Neuron.hpp"

class Net {
public:
  explicit Net(std::vector<int> topology);
  void feedForward(const std::vector<double>& inputs);
  void backProp(const std::vector<double>& targets);
  void evaluate(const std::vector<double>& targets);
  
  void run(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& labels);
  
  void save(const std::string& filename) const;
  explicit Net(const std::string& filename);
  
private:
  std::vector<std::vector<Neuron>> layers;
  
  double recentAverageError = 0;
  double recentAverageSmoothingFactor = 0;
};
