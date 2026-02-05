#ifndef ACF_EXTRACTOR_H
#define ACF_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class ACFExtractor {
private:
    cv::Size modelSize;
    int shrink;
    int numBins;

public:
    // Constructor: shrink=2 es mejor para objetos pequeños/niños
    ACFExtractor(cv::Size _size = cv::Size(32, 64), int _shrink = 2);
    
    // Devuelve una fila (1 x N_Features) lista para el clasificador
    cv::Mat compute(const cv::Mat& img);
    
    cv::Size getModelSize() const { return modelSize; }
};

#endif