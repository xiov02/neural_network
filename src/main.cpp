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

double sigmoid(double input) {return 1.0f / (1.0f + std::exp(-input));} // Sigmoid function
double relu(double x) { return x > 0 ? x : 0; } // ReLU Function

int main() {

    MultilayerPerceptron mlp = MultilayerPerceptron({4, 8, 3}, sigmoid);
    printf("Multilayer Perceptron created with %zu hidden layers.\n", mlp.hiddenLayers.size());

    std::vector<Flower> dataSet = data("../data/IRIS.csv");
    std::vector<std::vector<double>> dataPrepared = prepareTrainingData(dataSet);

    //shuffle data before splitting
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(dataPrepared.begin(), dataPrepared.end(), g);

    std::vector<std::vector<double>> trainingData(dataPrepared.begin(), dataPrepared.end() - 37);
    std::vector<std::vector<double>> testData(dataPrepared.end() - 37, dataPrepared.end());

    printf("Total data samples: %zu\n", trainingData.size());

    std::vector<double> x = mlp.forward({5.1f, 3.5f, 1.4f, 0.2f, 0.0f});
    printf("Initial forward pass output: [%f, %f, %f]\n", x[0], x[1], x[2]);

    std::vector<double> y = mlp.forward({3.1f, 3.5f, 1.4f, 0.2f, 0.0f});
    printf("Initial forward pass output: [%f, %f, %f]\n", y[0], y[1], y[2]);

    // Train the MLP
    printf("Starting training...\n");

    mlp.training(trainingData, 500, 0.05f);

    //verification on test data
    int correct = 0;
    for (const auto& dataPoint : testData) {
        std::vector<double> inputs(dataPoint.begin(), dataPoint.end() - 1);
        double target = dataPoint.back();
        const std::vector<double> predicted = mlp.forward(inputs);
        int predictedClass = std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end()));
        if (predictedClass == static_cast<int>(target)) {
            correct++;
        }
    }

    printf("Test Accuracy: %f%%\n", (static_cast<double>(correct) / testData.size()) * 100.0f);

    return 0;
}   