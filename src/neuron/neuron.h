#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <functional>

#include "neuronBase.h"
#include "../activation/activation.h"

class Neuron : public NeuronBase {
public:
    std::vector<double> weights;
    double bias;
    double score;
    // Pointer to activation function
    ActivationFunction activationFunction; 

    Neuron(int numberOfWeight, ActivationFunction activationFunction);

    const double forward(const std::vector<double>& inputs) override;
};

#endif // NEURON_H
