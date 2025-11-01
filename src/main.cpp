#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>

#include "neuron/neuron.h"
#include "neuron/inputNeuron.h"
#include "layer/hiddenLayer.h"
#include "network/multilayerPerceptron.h"

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

    HiddenLayer hiddenLayer = HiddenLayer(5, x, sigmoid);
    printf("Hidden Layer ID = %d\n", hiddenLayer.id);

    MultilayerPerceptron mlp = MultilayerPerceptron({3, 4, 2}, sigmoid);
    printf("Multilayer Perceptron created with %zu hidden layers.\n", mlp.hiddenLayers.size());

    const std::vector<float>& input = {0.4f, 0.6f, 0.8f};
    const std::vector<float>& output = mlp.forward(input);

    mlp.training({{0.4f, 0.6f, 0.8f, 1.0f, 0.0f}, {0.1f, 0.2f, 0.3f, 0.0f, 1.0f}}, 10, 0.01f);

    return 0;
}   