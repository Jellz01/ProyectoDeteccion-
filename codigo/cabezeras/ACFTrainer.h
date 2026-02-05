#ifndef ACF_TRAINER_H
#define ACF_TRAINER_H

#include "ACFExtractor.h"
#include "DatasetManager.h"
#include <opencv2/ml.hpp>

class ACFTrainer {
private:
    ACFExtractor* extractor;
    DatasetManager* dataMgr;

public:
    ACFTrainer(ACFExtractor* _ext, DatasetManager* _dm);
    
    // Ejecuta todo el pipeline y guarda el XML
    void runTraining(std::string outputModelPath);
};

#endif