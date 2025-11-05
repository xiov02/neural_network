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

int MultilayerPerceptron::softMaxInPlace(std::vector<float>& output) {
    float sumExp = 0.0f;

    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = std::exp(output[i]);
        sumExp += output[i];
    }

    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sumExp;
    }

    return 0;
}

const std::vector<float> MultilayerPerceptron::forward(const std::vector<float>& inputs) {
    std::vector<float> bufferA, bufferB;
    bufferA.resize(maxLayerSizes);
    bufferB.resize(maxLayerSizes);
    inputLayer.forward(inputs, bufferA);

    // Forward through others layers
    for (HiddenLayer& layer : hiddenLayers) {
        layer.forward(bufferA, bufferB);
        std::swap(bufferA, bufferB);
    }

    softMaxInPlace(bufferA);
    return bufferA;
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

    std::vector<float> inputsTemp;
    inputsTemp.resize(maxLayerSizes);
    printf("%d", inputsTemp.size());

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
                HiddenLayer* nextLayer = (i+1 < hiddenLayers.size()) ? &hiddenLayers[i + 1] : nullptr;

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
                    for (size_t j = 0; j < layer->neuronLayer.size(); ++j) {
                        layer->neuronLayer[j]->score = output[j] - (target==j ? 1 : 0);  // Simplified error
                    }
                }
                
                if (i > 0) {
                    hiddenLayers[i-1].getOutputs(inputsTemp);
                } else {
                    std::copy(inputs.begin(), inputs.end(), inputsTemp.begin());
                }

                for (Neuron* neuron : layer->neuronLayer) {
                    // float gradient = neuron->score * neuron->output * (1 - neuron->output); // Sigmoid derivative
                    for (size_t w = 0; w < neuron->weights.size(); ++w) {
                        neuron->weights[w] -= learningRate * neuron->score * inputsTemp[w]; // Update weights
                    }
                    neuron->bias -= learningRate * neuron->score; // Update bias
                }
            }
        }
        printf("Epoch %zu, Loss: %f\n", epoch + 1, epochLoss / trainingData.size());
    }
}