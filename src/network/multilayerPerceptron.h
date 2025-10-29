#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <vector>
#include <functional>

#include "../layer/inputLayer.h"
#include "../layer/hiddenLayer.h"
#include "network.h"

class MultilayerPerceptron : public Network {
public:

    int id2;

    MultilayerPerceptron(std::vector<int> layerSizes, std::function<float(float)> activationFunction);

    const std::vector<float> forward(const std::vector<float>& inputs) override;
};

#endif // MULTILAYERPERCEPTRON_H
