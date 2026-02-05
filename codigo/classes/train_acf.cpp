#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace cv::ml;
using namespace std;
namespace fs = std::filesystem;

// ================= CONFIGURACIÓN =================
const Size WIN_SIZE(64, 64);
const string POS_DIR = "dataset/pos/";
const string NEG_DIR = "dataset/neg/";
const string OUTPUT_MODEL = "hog_wrestling.yml";

// ================= MAIN =================
int main()
{
    // --------- HOG CONFIG (CLAVE) ---------
    HOGDescriptor hog(
        WIN_SIZE,      // winSize
        Size(16,16),   // blockSize
        Size(8,8),     // blockStride
        Size(8,8),     // cellSize
        9              // bins
    );

    vector<vector<float>> descriptors;
    vector<int> labels;

    cout << "[INFO] Cargando positivos..." << endl;

    // --------- POSITIVOS ---------
    for (const auto& entry : fs::directory_iterator(POS_DIR))
    {
        Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        resize(img, img, WIN_SIZE);

        vector<float> desc;
        hog.compute(img, desc);

        descriptors.push_back(desc);
        labels.push_back(+1);
    }

    cout << "[INFO] Positivos cargados: " << labels.size() << endl;
    size_t posCount = labels.size();

    cout << "[INFO] Cargando negativos..." << endl;

    // --------- NEGATIVOS ---------
    for (const auto& entry : fs::directory_iterator(NEG_DIR))
    {
        Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        resize(img, img, WIN_SIZE);

        vector<float> desc;
        hog.compute(img, desc);

        descriptors.push_back(desc);
        labels.push_back(-1);
    }

    cout << "[INFO] Negativos cargados: " << labels.size() - posCount << endl;

    if (descriptors.empty()) {
        cerr << "❌ ERROR: No se cargaron imágenes." << endl;
        return -1;
    }

    // --------- CONVERTIR A MATRIZ ---------
    Mat trainData((int)descriptors.size(), (int)descriptors[0].size(), CV_32F);

    for (size_t i = 0; i < descriptors.size(); i++)
    {
        memcpy(trainData.ptr<float>((int)i),
               descriptors[i].data(),
               descriptors[i].size() * sizeof(float));
    }

    Mat labelsMat(labels);

    // --------- ENTRENAR SVM ---------
    cout << "[INFO] Entrenando SVM..." << endl;

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setC(0.01);           // generaliza bien
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

    svm->train(trainData, ROW_SAMPLE, labelsMat);

    cout << "[INFO] SVM entrenado correctamente." << endl;

    // --------- EXTRAER DETECTOR HOG ---------
    Mat sv = svm->getSupportVectors();
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    vector<float> detector(sv.cols + 1);
    memcpy(detector.data(), sv.ptr(), sv.cols * sizeof(float));
    detector[sv.cols] = (float)-rho;

    // --------- GUARDAR MODELO ---------
    FileStorage fsOut(OUTPUT_MODEL, FileStorage::WRITE);
    fsOut << "detector" << detector;
    fsOut.release();

    cout << "✅ MODELO GUARDADO: " << OUTPUT_MODEL << endl;
    cout << "   Positivos: " << posCount << endl;
    cout << "   Negativos: " << labels.size() - posCount << endl;
    cout << "   Dimensión HOG: " << sv.cols << endl;

    return 0;
}
