#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>

struct Flower {
    std::string specie;
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
};

std::vector<Flower> data(std::string filePath);

std::vector<std::vector<float>> prepareTrainingData(const std::vector<Flower>& flowers);

std::string trim(const std::string& s);

#endif // DATA_H