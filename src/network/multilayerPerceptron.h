#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <vector>
#include <functional>

#include "../layer/inputLayer.h"
#include "../layer/hiddenLayer.h"
#include "network.h"

class MultilayerPerceptron : public Network {
public:

    MultilayerPerceptron(std::vector<int> layerSizes, std::function<float(float)> activationFunction);

    int softMaxInPlace(std::vector<float>& output);

    const std::vector<float> forward(const std::vector<float>& inputs) override;

    void training(const std::vector<std::vector<float>> trainingData, int epochs, float learningRate) override;

    float computeLoss(const std::vector<float>& predicted, int target) override;
    


};

#endif // MULTILAYERPERCEPTRON_H
