#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <functional>

#include "neuronBase.h"

class Neuron : public NeuronBase {
public:
    std::vector<double> weights;
    double bias;
    double score;
    // Pointer to activation function
    std::function<double(double)> activationFunction; 

    Neuron(int numberOfWeight, std::function<double(double)> activationFunction);

    const double forward(const std::vector<double>& inputs) override;
};

#endif // NEURON_H
