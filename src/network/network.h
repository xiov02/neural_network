#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <functional>
#include <string>

#include "../layer/inputLayer.h"
#include "../layer/hiddenLayer.h"
#include "../activation/activation.h"

class Network {
protected:
    Network(std::vector<int> layerSizes, const std::string activationFunction);

public:
    virtual ~Network() = default;

    static int global_id_counter;

    int id;
    int maxLayerSizes;
    std::vector<HiddenLayer> hiddenLayers;
    InputLayer inputLayer;

    virtual const std::vector<double> forward(const std::vector<double>& inputs) = 0;

    virtual void training(const std::vector<std::vector<double>> trainingData, int epochs, double learningRate) = 0; 

    virtual double computeLoss(const std::vector<double>& predicted, int target) = 0;
};

#endif // NETWORK_H
