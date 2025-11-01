#include <random>
#include <algorithm>

#include "network.h"

int Network::global_id_counter = 0; // Define and initialize static member

Network::Network(std::vector<int> layerSizes, std::function<float(float)> activationFunction)
        : inputLayer(layerSizes[0], 0, activationFunction)
    {
        for (size_t i = 1; i < layerSizes.size(); ++i)
            hiddenLayers.emplace_back(layerSizes[i], layerSizes[i - 1], activationFunction);
    }