#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

vector<float> loadCustomDetector(string path) {
    vector<float> detector;
    float value;
    ifstream file(path);
    if (!file.is_open()) return detector;
    while (file >> value) detector.push_back(value);
    file.close();
    return detector;
}

int main() {
    // 1. Configuración EXACTA del Descriptor (64x128)
    // Estos parámetros deben ser idénticos a los de tu script de Python
    HOGDescriptor hogCustom(
        Size(64, 128), // winSize
        Size(16, 16),  // blockSize
        Size(8, 8),    // blockStride
        Size(8, 8),    // cellSize
        9              // nbins
    );

    string modelPath = "/home/jellz/Documents/VisionPorComputador/ProyectoDeteccion/python/custom_hog_detector.txt";
    vector<float> myDetector = loadCustomDetector(modelPath);

    if (myDetector.empty()) {
        cout << "❌ No se pudo cargar el archivo txt." << endl;
        return -1;
    }

    // Intentar asignar el detector
    hogCustom.setSVMDetector(myDetector);

    VideoCapture cap("/home/jellz/Documents/VisionPorComputador/ProyectoDeteccion/codigo/classes/videomall.mp4");
    Mat frame;

    while (cap.read(frame)) {
        resize(frame, frame, Size(640, 480));

        vector<Rect> poses;
        vector<double> confidences;

        // --- TRUCO DE DIAGNÓSTICO ---
        // Usamos detect() en lugar de detectMultiScale para ver los niveles de confianza reales
        // hitThreshold negativo (-0.5) para obligar al modelo a mostrar TODO lo que sospecha
        hogCustom.detectMultiScale(frame, poses, -0.5, Size(8,8), Size(32,32), 1.05, 2);

        for (const auto& r : poses) {
            rectangle(frame, r, Scalar(255, 0, 0), 2); // Azul para poses raras
            putText(frame, "SOSPECHA DE POSE", Point(r.x, r.y - 5), 
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 0), 1);
        }

        imshow("Diagnostico Modelo Custom", frame);
        if (waitKey(1) == 27) break;
    }
    return 0;
}