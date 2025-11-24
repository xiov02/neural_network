#include <random>
#include <algorithm>

#include <string>

#include "../activation/activation.h"
#include "network.h"

int Network::global_id_counter = 0; // Define and initialize static member

Network::Network(std::vector<int> layerSizes, const std::string activationFunction)
        : inputLayer(layerSizes[0])
    {
        ActivationFunction activationFunctionInstance = ActivationFunction(activationFunction);
        ActivationFunction identityActivation = ActivationFunction("");
        maxLayerSizes = layerSizes[0];
        for (size_t i = 1; i < layerSizes.size() - 1; ++i) {
            hiddenLayers.emplace_back(layerSizes[i], layerSizes[i - 1], activationFunctionInstance);
            maxLayerSizes = std::max(maxLayerSizes, layerSizes[i]);
        }
        hiddenLayers.emplace_back(layerSizes[layerSizes.size() - 1], layerSizes[layerSizes.size() - 2], identityActivation); // Output layer with identity activation
        maxLayerSizes = std::max(maxLayerSizes, layerSizes[layerSizes.size() - 1]);

        printf("Max layer size determined: %d\n", maxLayerSizes);
    }
