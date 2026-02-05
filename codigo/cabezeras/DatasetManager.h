#ifndef DATASET_MANAGER_H
#define DATASET_MANAGER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class DatasetManager {
private:
    std::string rootDir;
    std::vector<std::string> imageIDs;

public:
    DatasetManager(std::string _rootDir);
    
    // Carga la lista desde train.txt
    void init();
    
    // Obtiene imagen y sus bounding boxes (parseando XML internamente)
    // Retorna true si encontró la imagen y el XML
    bool getSample(int index, cv::Mat& outImg, std::vector<cv::Rect>& outBoxes);
    
    int getTotalSamples();
};

#endif