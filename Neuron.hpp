//
// Created by Hosomi on 02/10/2020.
//
#pragma once
#include <vector>

class Neuron {
public:
  explicit Neuron(int outputs);
  explicit Neuron(std::vector<double> outputs, double bias);
  
  // Gradient
  void addGradient(double amount);
  void clearGradients();
  double getCurrentGradient() const;
  
  void calcGradient(const std::vector<Neuron>& j_layer);
  void updateWeight(int j, double j_gradient);
  
  // GET and SET
  double getActivation() const { return activation; };
  void setActivation(double value) { activation = value; };
  double getConnectionWeight(int index) const { return connections[index]; };
  double getBias() const { return bias; };
  
  // Statics
  static constexpr double eta = 0.15; // learning rate
  static constexpr double alpha = 0.5; // momentum
  static double activationFunction(double x);
  static double activationFuncDerivative(double x);
private:
  double activation = 0; // start 0 (activation range is -1 to 1)
  std::vector<double> connections; // connection weight to the NEXT layer
  double bias = 0; // TODO: research better initialization technique
  
  std::vector<double> gradients;
};
