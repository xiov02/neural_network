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

    Network() : id(0), hiddenLayers(), inputLayer(InputLayer(0, 0, [](float x){ return x; })), outputLayer(HiddenLayer(0, 0, [](float x){ return x; })) {}

    Network(int id, std::vector<HiddenLayer> hiddenLayers, InputLayer inputLayer, HiddenLayer outputLayer)
        : id(id), hiddenLayers(hiddenLayers), inputLayer(inputLayer), outputLayer(outputLayer) {};

    const std::vector<float> forward(const std::vector<float>& inputs);
};

#endif // NETWORK_H
