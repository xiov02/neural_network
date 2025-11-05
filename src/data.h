#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>

struct Flower {
    std::string specie;
    double sepal_length;
    double sepal_width;
    double petal_length;
    double petal_width;
};

std::vector<Flower> data(std::string filePath);

std::vector<std::vector<double>> prepareTrainingData(const std::vector<Flower>& flowers);

std::string trim(const std::string& s);

#endif // DATA_H