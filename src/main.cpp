#include <iostream>
#include <cmath>

#include "neuron/neuron.h"

float sigmoid(float input) {return 1.0f / (1.0f + std::exp(-input));} // Sigmoid function
float relu(float x) { return x > 0 ? x : 0; } // ReLU Function

int main() {

    int x = 10;
    Neuron neuron = Neuron(x, relu);

    printf("Neuron ID = %d\n",neuron.id);

    for (size_t i = 0; i < x; ++i) {
        printf("Neuron Weight[%zu] = %f\n", i, neuron.weights[i]);
    }
    
    printf("Test activation function output for input 0.5: %f\n", neuron.activationFunction(0.5f));

    return 0;
}