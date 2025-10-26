#include "neuron.h"
#include <random>

int Neuron::global_id_counter = 0; // Define and initialize static member

Neuron::Neuron(int numberOfWeight, std::function<float(float)> activationFunction)
    : output(0.0f), score(0.0f), activationFunction(activationFunction)
{
    id = global_id_counter++;
    
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
