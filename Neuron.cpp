//
// Created by Hosomi on 02/10/2020.
//

#include <cmath>
#include "Neuron.hpp"
#include <random>

Neuron::Neuron(int outputs) {
  for(int i = 0; i < outputs; ++i)
  {
    connections.emplace_back((double)rand() / RAND_MAX); // could be random
    delta_weights.emplace_back(0);
  }
  bias = (double)rand() / RAND_MAX;
}

double Neuron::activationFunction(double z)
{
  // tanh
  return tanh(z);
}

double Neuron::activationFuncDerivative(double z)
{
  // tanh derivative
  return (1 / cosh(z)) * (1 / cosh(z));
}

void Neuron::addGradient(double amount)
{
  gradients.emplace_back(amount);
}

void Neuron::clearGradients()
{
  gradients.clear();
}

double Neuron::getCurrentGradient() const {
  return gradients.back();
}

void Neuron::updateWeight(int j, double j_gradient)
{
  // double old_delta_weight // TODO: reimplement momentum
  
  double delta_weight =
          Neuron::eta
          * activation
          * j_gradient
          + alpha // add momentum
          * delta_weights[j];
  
  connections[j] += delta_weight;
  delta_weights[j] = delta_weight;
}

void Neuron::calcGradient(const std::vector<Neuron>& j_layer)
{
  // sum of contribution to next layer costs
  double sum = 0;
  for(int j = 0; j < j_layer.size(); ++j)
  {
    sum += getConnectionWeight(j) * j_layer[j].getCurrentGradient();
  }
  double new_gradient = sum * Neuron::activationFuncDerivative(activation); // TODO: confirm this
  gradients.emplace_back(new_gradient);
  
  double bias_delta = Neuron::eta * new_gradient;
  bias += bias_delta;
  
}

// File load contructor
Neuron::Neuron(std::vector<double> weights, double bias)
{
  for(auto& weight : weights)
  {
    connections.emplace_back(weight); // could be random
  }
  this->bias = bias;
}
