#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>

#include "neuron/neuron.h"
#include "neuron/inputNeuron.h"
#include "layer/hiddenLayer.h"
#include "network/multilayerPerceptron.h"
#include "data.h"

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

    MultilayerPerceptron mlp = MultilayerPerceptron({4, 8, 3}, sigmoid);
    printf("Multilayer Perceptron created with %zu hidden layers.\n", mlp.hiddenLayers.size());

    const std::vector<float>& input = {0.4f, 0.6f, 0.8f, 0.2f};
    const std::vector<float>& output = mlp.forward(input);

    //Loss avec [1, 0, 0] comme target
    float loss = mlp.computeLoss(output, 0);
    printf("Computed loss for target class 0: %f\n", loss);



    printf("MLP output for given input:\n");
    for (size_t i = 0; i < output.size(); ++i) {
        printf("Output[%zu] = %f\n", i, output[i]);
    }

    std::vector<Flower> dataSet = data("../data/IRIS.csv");
    std::vector<std::vector<float>> dataPrepared = prepareTrainingData(dataSet);

    //shuffle data before splitting
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(dataPrepared.begin(), dataPrepared.end(), g);

    std::vector<std::vector<float>> trainingData(dataPrepared.begin(), dataPrepared.end() - 20);
    std::vector<std::vector<float>> testData(dataPrepared.end() - 37, dataPrepared.end());

    for (size_t i = 0; i < 5; ++i) {
        printf("Sample %zu: ", i);
        for (float val : trainingData[i]) {
            printf("%f ", val);
        }
        printf("\n");
    }

    printf("Number of training points = %zu\n", trainingData.size());

    for (size_t i = 0; i < trainingData.size(); ++i) {
        printf("Training point %zu size = %zu\n", i, trainingData[i].size());
    }
    fflush(stdout);

    printf("Total data samples: %zu\n", trainingData.size());

    // Train the MLP
    printf("Starting training...\n");

    mlp.training(trainingData, 300, 0.05f);

    //verification on test data
    int correct = 0;
    for (const auto& dataPoint : testData) {
        std::vector<float> inputs(dataPoint.begin(), dataPoint.end() - 1);
        float target = dataPoint.back();
        const std::vector<float> predicted = mlp.forward(inputs);
        int predictedClass = std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end()));
        if (predictedClass == static_cast<int>(target)) {
            correct++;
        }
    }

    printf("Test Accuracy: %f%%\n", (static_cast<float>(correct) / testData.size()) * 100.0f);

    return 0;
}   