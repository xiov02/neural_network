#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <functional>

class Neuron {
public:
    static int global_id_counter;

    int id;
    std::vector<float> weights;
    float bias;
    float output;
    float score;
    // Pointer to activation function
    std::function<float(float)> activationFunction; 

    Neuron(int numberOfWeight, std::function<float(float)> activationFunction);

    const float forward(const std::vector<float>& inputs);
};

#endif // NEURON_H
