#include "neuron.h"
#include <random>



Neuron::Neuron(int numberOfWeight, ActivationFunction activationFunction)
    : NeuronBase(), activationFunction(activationFunction)
{
    // Initialize weights vector with random values between -1 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0f, 1.0f);

    weights.resize(numberOfWeight);
    for (int i = 0; i < numberOfWeight; ++i) {
        weights[i] = dis(gen);
    }

    // Initialize bias with a random value between -1 and 1
    bias = dis(gen);
}

const double Neuron::forward(const std::vector<double>& inputs) {
    double totalInput = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        totalInput += weights[i] * inputs[i];
    }
    output = activationFunction.function(totalInput);
    return output;
}
