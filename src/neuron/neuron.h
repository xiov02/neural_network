#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <functional>

#include "neuronBase.h"

class Neuron : public NeuronBase {
public:
    std::vector<float> weights;
    float bias;
    float score;
    // Pointer to activation function
    std::function<float(float)> activationFunction; 

    Neuron(int numberOfWeight, std::function<float(float)> activationFunction);

    const float forward(const std::vector<float>& inputs);
};

#endif // NEURON_H
