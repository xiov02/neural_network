#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <functional>

#include "../layer/inputLayer.h"
#include "../layer/hiddenLayer.h"

class Network {
protected:
    Network(std::vector<int> layerSizes, std::function<float(float)> activationFunction);

public:
    virtual ~Network() = default;

    static int global_id_counter;

    int id;
    std::vector<HiddenLayer> hiddenLayers;
    InputLayer inputLayer;

    virtual const std::vector<float> forward(const std::vector<float>& inputs) = 0;

    virtual void training(const std::vector<std::vector<float>> trainingData, int epochs, float learningRate) = 0; 

    virtual float computeLoss(const std::vector<float>& predicted, int target) = 0;
};

#endif // NETWORK_H
