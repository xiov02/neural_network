#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <functional>

#include "../layer/inputLayer.h"
#include "../layer/hiddenLayer.h"

class Network {
public:

    static int global_id_counter;

    int id;
    std::vector<HiddenLayer> hiddenLayers;
    InputLayer inputLayer;
    HiddenLayer outputLayer;

    Network(std::vector<int> layerSizes, std::function<float(float)> activationFunction);

    virtual const std::vector<float> forward(const std::vector<float>& inputs) = 0;
    
};

#endif // NETWORK_H
