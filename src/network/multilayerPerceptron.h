#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <vector>
#include <functional>

#include "../layer/inputLayer.h"
#include "../layer/hiddenLayer.h"
#include "network.h"

class MultilayerPerceptron : public Network {
public:

    MultilayerPerceptron(std::vector<int> layerSizes, std::function<double(double)> activationFunction);

    int softMaxInPlace(std::vector<double>& output);

    const std::vector<double> forward(const std::vector<double>& inputs) override;

    void training(const std::vector<std::vector<double>> trainingData, int epochs, double learningRate) override;

    double computeLoss(const std::vector<double>& predicted, int target) override;
    


};

#endif // MULTILAYERPERCEPTRON_H
