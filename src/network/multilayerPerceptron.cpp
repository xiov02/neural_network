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

const std::vector<float> MultilayerPerceptron::softMax(const std::vector<float>& inputs) {
    std::vector<float> expValues(inputs.size());
    float sumExp = 0.0f;

    for (size_t i = 0; i < inputs.size(); ++i) {
        expValues[i] = std::exp(inputs[i]);
        sumExp += expValues[i];
    }

    for (size_t i = 0; i < expValues.size(); ++i) {
        expValues[i] /= sumExp;
    }

    return expValues;
}

const std::vector<float> MultilayerPerceptron::forward(const std::vector<float>& inputs) {
    std::vector<float> currentOutputs = inputLayer.forward(inputs);

    // Forward through others layers
    for (HiddenLayer& layer : hiddenLayers) {
        currentOutputs = layer.forward(currentOutputs);
    }

    // printf("Raw outputs before SoftMax:\n");
    // for (size_t i = 0; i < currentOutputs.size(); ++i) {
    //     printf("Output[%zu] = %f\n", i, currentOutputs[i]);
    // }

    return softMax(currentOutputs);
}

float MultilayerPerceptron::computeLoss(const std::vector<float>& predicted, int target) {
    // categorical cross-entropy loss
    float p = predicted[target];

    // Éviter log(0)
    float loss = -std::log(p + 1e-15f);

    return loss;
}

void MultilayerPerceptron::training(std::vector<std::vector<float>> trainingData, int epochs, float learningRate) {

    std::random_device rd;
    std::mt19937 g(rd());

    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        //shuffle training data for each epoch
        std::shuffle(trainingData.begin(), trainingData.end(), g);

        float epochLoss = 0.0f;

        for (const auto& dataPoint : trainingData) {
            // Assuming the last element is the target output
            std::vector<float> inputs(dataPoint.begin(), dataPoint.end() - 1);
            int target = static_cast<int>(dataPoint.back());

            const std::vector<float> output = forward(inputs);

            float loss = computeLoss(output, target);
            // printf("Output: %f\n", output[0]);
            epochLoss += loss;

            for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
                HiddenLayer* layer = &hiddenLayers[i];
                HiddenLayer* nextLayer = (i < hiddenLayers.size() - 1) ? &hiddenLayers[i + 1] : nullptr;


                if (nextLayer) {
                    for (size_t j = 0; j < layer->neuronLayer.size(); ++j) {
                        float error = 0.0f;
                        for (Neuron* nextNeuron : nextLayer->neuronLayer) {
                            error += nextNeuron-> weights[j] * nextNeuron->score;
                        }
                        float derivative = layer->neuronLayer[j]->output * (1 - layer->neuronLayer[j]->output); // Sigmoid derivative
                        layer->neuronLayer[j]->score = error * derivative;
                    }
                }
                else {
                    for (size_t j = 0; j < output.size(); ++j) {
                        layer->neuronLayer[j]->score = output[j] - (target==j ? 1 : 0);  // Simplified error
                    }
                }

                std::vector<float> inputsTemp;
                
                if (i > 0) {
                    inputsTemp = hiddenLayers[i-1].getOutputs(); // Assuming getOutputs() returns std::vector<float> of previous layer outputs
                } else {
                    inputsTemp = inputs; // For the first hidden layer, use the original inputs
                }

                for (Neuron* neuron : layer->neuronLayer) {
                    // float gradient = neuron->score * neuron->output * (1 - neuron->output); // Sigmoid derivative
                    for (size_t w = 0; w < inputsTemp.size(); ++w) {
                        neuron->weights[w] -= learningRate * neuron->score * inputsTemp[w]; // Update weights
                    }
                    neuron->bias -= learningRate * neuron->score; // Update bias
                }
            }
        }
        printf("Epoch %zu, Loss: %f\n", epoch + 1, epochLoss / trainingData.size());
    }
}