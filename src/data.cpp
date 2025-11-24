#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

#include "data.h"

std::vector<Flower> data(std::string filePath) {
    std::ifstream file(filePath);
    std::string ligne;
    std::vector<Flower> flowers;

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return flowers;
    }

    double maxSepalLength, maxSepalWidth, maxPetalLength, maxPetalWidth = 0.0f;

    std::getline(file, ligne); // Ignorer l'en-tête
    while (std::getline(file, ligne)) {
        std::stringstream ss(ligne);
        std::string specie, sepalLengthStr, sepalWidthStr, petalLengthStr, petalWidthStr;
        double sepalLength, sepalWidth, petalLength, petalWidth;

        std::getline(ss, sepalLengthStr, ',');
        std::getline(ss, sepalWidthStr, ',');
        std::getline(ss, petalLengthStr, ',');
        std::getline(ss, petalWidthStr, ',');
        std::getline(ss, specie, ',');

        sepalLength = std::stof(sepalLengthStr);
        sepalWidth = std::stof(sepalWidthStr);
        petalLength = std::stof(petalLengthStr);
        petalWidth = std::stof(petalWidthStr);

        if (sepalLength > maxSepalLength) maxSepalLength = sepalLength;
        if (sepalWidth > maxSepalWidth) maxSepalWidth = sepalWidth;
        if (petalLength > maxPetalLength) maxPetalLength = petalLength;
        if (petalWidth > maxPetalWidth) maxPetalWidth = petalWidth;

        Flower f{trim(specie), sepalLength, sepalWidth, petalLength, petalWidth};
        flowers.push_back(f);
    }

    printf("Max Sepal Length: %f\n", maxSepalLength);
    printf("Max Sepal Width: %f\n", maxSepalWidth);
    printf("Max Petal Length: %f\n", maxPetalLength);
    printf("Max Petal Width: %f\n", maxPetalWidth);

    return flowers;
}

std::vector<std::vector<double>> prepareTrainingData(const std::vector<Flower>& flowers) {
    std::vector<std::vector<double>> trainingData;

    for (const auto& flower : flowers) {
        std::vector<double> dataPoint;
        dataPoint.push_back(flower.sepal_length / 7.9f);
        dataPoint.push_back(flower.sepal_width / 4.4f);
        dataPoint.push_back(flower.petal_length / 6.9f);
        dataPoint.push_back(flower.petal_width / 2.5f);

        // One-hot encoding for species
        if (flower.specie == "Iris-setosa") {
            dataPoint.push_back(0.0f);
        } else if (flower.specie == "Iris-versicolor") {
            dataPoint.push_back(1.0f);
        } else if (flower.specie == "Iris-virginica") {
            dataPoint.push_back(2.0f);
        }

        trainingData.push_back(dataPoint);
    }

    return trainingData;
}

std::string trim(const std::string& s) {
    std::string res = s;
    res.erase(res.begin(), std::find_if(res.begin(), res.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    res.erase(std::find_if(res.rbegin(), res.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), res.end());
    return res;
}
