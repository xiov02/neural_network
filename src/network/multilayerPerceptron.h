#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <vector>
#include <functional>

#include "../layer/inputLayer.h"
#include "../layer/hiddenLayer.h"

class MultilayerPerceptron {
public:
    int id;
    std::vector<HiddenLayer> hiddenLayers;
    InputLayer inputLayer;
    HiddenLayer outputLayer;

    MultilayerPerceptron(std::vector<int> layerSizes, std::function<float(float)> activationFunction);
};

#endif // MULTILAYERPERCEPTRON_H
