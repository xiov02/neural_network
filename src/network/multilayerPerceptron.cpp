#include <random>
#include <algorithm>

#include "multilayerPerceptron.h"

MultilayerPerceptron::MultilayerPerceptron(
    std::vector<int> layerSizes,
    std::function<float(float)> activationFunction
)
    : Network(layerSizes, activationFunction)
{
    id = global_id_counter++;
}

const std::vector<float> MultilayerPerceptron::forward(const std::vector<float>& inputs) {
    std::vector<float> currentOutputs = inputLayer.forward(inputs);

    // Forward through others layers
    for (HiddenLayer& layer : hiddenLayers) {
        currentOutputs = layer.forward(currentOutputs);
    }

    return currentOutputs;
}

float MultilayerPerceptron::computeLoss(const std::vector<float>& predicted, float target) {
    // categorical cross-entropy loss
    float loss = 0.0f;
    for (float p : predicted) {
        loss += -target * std::log(p + 1e-9f);
    }
    return loss;
}

void MultilayerPerceptron::training(std::vector<std::vector<float>> trainingData, int epochs, float learningRate) {
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        //shuffle training data for each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(trainingData.begin(), trainingData.end(), g);

        float epochLoss = 0.0f;

        for (const auto& dataPoint : trainingData) {
            // Assuming the last element is the target output
            std::vector<float> inputs(dataPoint.begin(), dataPoint.end() - 1);
            float target = dataPoint.back();

            const std::vector<float> output = MultilayerPerceptron::forward(inputs);

            float loss = computeLoss(output, target);
            epochLoss += loss;

            for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
                HiddenLayer* layer = &hiddenLayers[i];
                HiddenLayer* nextLayer = (i < hiddenLayers.size() - 1) ? &hiddenLayers[i + 1] : nullptr;


                if (nextLayer) {
                    for (size_t j = 0; j < layer->neuronLayer.size(); ++j) {
                        float error = 0.0f;
                        for (Neuron* nextNeuron : nextLayer->neuronLayer) {
                            error += nextNeuron->weights[j] * nextNeuron->score;
                        }
                        layer->neuronLayer[j]->score = error;
                    }
                }
                else {
                    for (size_t i = 0; i < output.size(); ++i) {
                        layer->neuronLayer[i]->score = output[i] - target; // Simplified error
                    }
                }

                for (Neuron* neuron : layer->neuronLayer) {
                    float gradient = neuron->score * neuron->output * (1 - neuron->output); // Sigmoid derivative
                    for (size_t w = 0; w < neuron->weights.size(); ++w) {
                        neuron->weights[w] -= learningRate * gradient * inputs[w]; // Update weights
                    }
                    neuron->bias -= learningRate * gradient; // Update bias
                }
            }
        }
        printf("Epoch %zu, Loss: %f\n", epoch + 1, epochLoss / trainingData.size());
    }
}