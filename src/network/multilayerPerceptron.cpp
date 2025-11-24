#include <random>
#include <algorithm>

#include <string>

#include "multilayerPerceptron.h"

MultilayerPerceptron::MultilayerPerceptron(
    std::vector<int> layerSizes,
    const std::string activationFunction
)
    : Network(layerSizes, activationFunction)
{
    id = global_id_counter++;
}

int MultilayerPerceptron::softMaxInPlace(std::vector<double>& output) {
    double maxVal = *std::max_element(output.begin(), output.end());
    double sumExp = 0.0f;
    int n = hiddenLayers.back().getNeuronSize();
    // printf("Number of output Neuron : %d", n);
    for (size_t i =0; i<n; ++i) {
        output[i] = std::exp(output[i]);  // stabilité numérique
        sumExp += output[i];
    }
    for (size_t i =0; i<n; ++i) {
        output[i] /= sumExp;
    }
    return 0;
}


const std::vector<double> MultilayerPerceptron::forward(const std::vector<double>& inputs) {
    std::vector<double> bufferA, bufferB;
    bufferA.resize(maxLayerSizes);
    bufferB.resize(maxLayerSizes);
    inputLayer.forward(inputs, bufferA);

    // Forward through others layers
    for (HiddenLayer& layer : hiddenLayers) {
        layer.forward(bufferA, bufferB);
        std::swap(bufferA, bufferB);
    }


    // for (size_t i = 0; i<bufferA.size(); ++i) {
    //     printf("%d : %f, ",i, bufferA[i]);
    // }
    // printf("\n");

    softMaxInPlace(bufferA);

    // for (size_t i = 0; i<bufferA.size(); ++i) {
    //     printf("%d : %f, ",i, bufferA[i]);
    // }
    // printf("\n");
    std::vector<double> output(bufferA.begin(), bufferA.begin() + 3);
    return output;
}

double MultilayerPerceptron::computeLoss(const std::vector<double>& predicted, int target) {
    // categorical cross-entropy loss
    double p = predicted[target];

    // Éviter log(0)
    double loss = -std::log(p + 1e-9f);

    // printf("predicted : %f, loss : %f \n", p, loss);

    return loss;
}

void MultilayerPerceptron::training(
    std::vector<std::vector<double>> trainingData,
    int epochs,
    double learningRate
) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<double> inputsTemp(maxLayerSizes);
    double d = 0.0;
    int compte = 0;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(trainingData.begin(), trainingData.end(), g);

        double epochLoss = 0.0;

        for (const auto& dataPoint : trainingData) {
            // ---- [1] Forward ----
            std::vector<double> inputs(dataPoint.begin(), dataPoint.end() - 1);
            int target = static_cast<int>(dataPoint.back());
            const std::vector<double> output = forward(inputs);
            double loss = computeLoss(output, target);
            epochLoss += loss;

            // ---- [2] Backward pass (calcul des deltas / scores) ----
            for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
                HiddenLayer* layer = &hiddenLayers[i];
                HiddenLayer* nextLayer = (i + 1 < hiddenLayers.size()) ? &hiddenLayers[i + 1] : nullptr;

                if (nextLayer) {
                    // Couches cachées : delta = somme (w_jk * delta_k) * f'(output)
                    for (size_t j = 0; j < layer->neuronLayer.size(); ++j) {
                        double error = 0.0;
                        for (Neuron* nextNeuron : nextLayer->neuronLayer)
                            error += nextNeuron->weights[j] * nextNeuron->score;

                        double derivative = layer->neuronLayer[j]->output * (1.0 - layer->neuronLayer[j]->output);
                        layer->neuronLayer[j]->score = error * derivative;
                        d += layer->neuronLayer[j]->score;
                        compte++;
                    }
                } else {
                    // Couche de sortie : delta = (y_pred - y_true)
                    for (size_t j = 0; j < layer->neuronLayer.size(); ++j)
                        layer->neuronLayer[j]->score = output[j] - ((target == static_cast<int>(j)) ? 1.0 : 0.0);
                }
            }

            // ---- [3] Mise à jour des poids ----
            for (size_t i = 0; i < hiddenLayers.size(); ++i) {
                HiddenLayer* layer = &hiddenLayers[i];

                // Entrées de cette couche = sorties de la précédente
                if (i == 0)
                    std::copy(inputs.begin(), inputs.end(), inputsTemp.begin());
                else
                    hiddenLayers[i - 1].getOutputs(inputsTemp);

                // Update poids + biais
                for (Neuron* neuron : layer->neuronLayer) {
                    for (size_t w = 0; w < neuron->weights.size(); ++w)
                        neuron->weights[w] -= learningRate * neuron->score * inputsTemp[w];

                    neuron->bias -= learningRate * neuron->score;
                }
            }
        }

        printf("Epoch %zu - Loss moyenne: %.6f - Score moyen: %.6f\n",
               epoch + 1, epochLoss / trainingData.size(), d / std::max(1, compte));
    }
}


void MultilayerPerceptron::drawNetwork() {
    for (HiddenLayer layer : hiddenLayers) {
        printf("Hidden Layer %d : \n", layer.id);
        for (Neuron *neuron: layer.neuronLayer) {
            printf("\tNeuron number %d : ", neuron->id);
            for (double weight: neuron->weights) {
                printf("%f, ", weight);
            }
            printf("\n");
        }
    }
}