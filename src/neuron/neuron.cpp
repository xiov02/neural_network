#include "neuron.h"
#include <random>

Neuron::Neuron(int numberOfWeight, std::function<float(float)> activationFunction)
    : NeuronBase(), activationFunction(activationFunction)
{
    // Initialize weights vector with random values between -1 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    weights.resize(numberOfWeight);
    for (int i = 0; i < numberOfWeight; ++i) {
        weights[i] = dis(gen);
    }

    // Initialize bias with a random value between -1 and 1
    bias = dis(gen);
}

const float Neuron::forward(const std::vector<float>& inputs) {
    float totalInput = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        totalInput += weights[i] * inputs[i];
    }
    output = activationFunction(totalInput);
    return output;
}
