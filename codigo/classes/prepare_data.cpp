/**
 * codigo/classes/prepare_data.cpp
 * Genera dataset leyendo IDs desde train.txt para evitar contaminacion de datos.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <random>

namespace fs = std::filesystem;

// --- CONFIGURACIÓN ---
const int MODEL_W = 64;
const int MODEL_H = 128;
const int NUM_NEGATIVES_PER_IMAGE = 2; // Negativos por cada imagen procesada

// Rutas (ajustadas a tu estructura)
const std::string BASE_DIR = "../Dataset"; 
const std::string OUT_DIR = "../generated_data"; 

// Archivo de lista a usar (Usamos train.txt para entrenar)
// Si quieres incluir validación, puedes procesar ambos.
const std::vector<std::string> LIST_FILES = {"train.txt"}; 

struct Box { int x1, y1, x2, y2; };

// IoU para asegurar que el negativo no solapa con un peatón
float computeIoU(const cv::Rect& a, const Box& b) {
    int xx1 = std::max(a.x, b.x1);
    int yy1 = std::max(a.y, b.y1);
    int xx2 = std::min(a.x + a.width, b.x2);
    int yy2 = std::min(a.y + a.height, b.y2);

    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);

    float interArea = (float)(w * h);
    float areaA = (float)(a.width * a.height);
    float areaB = (float)((b.x2 - b.x1) * (b.y2 - b.y1));

    return interArea / (areaA + areaB - interArea);
}

void processDataset() {
    std::string imgDir = BASE_DIR + "/Images";
    std::string annDir = BASE_DIR + "/Annotations";
    
    std::string posDir = OUT_DIR + "/positives";
    std::string negDir = OUT_DIR + "/negatives";

    if (!fs::exists(imgDir) || !fs::exists(annDir)) {
        std::cerr << "ERROR: No se encuentra Dataset en " << fs::absolute(BASE_DIR) << std::endl;
        return;
    }

    fs::create_directories(posDir);
    fs::create_directories(negDir);

    int posCount = 0;
    int negCount = 0;
    int filesProcessed = 0;
    
    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "=== Generando Dataset desde listas de texto ===" << std::endl;

    for (const std::string& listName : LIST_FILES) {
        std::string listPath = BASE_DIR + "/" + listName;
        std::ifstream listFile(listPath);
        
        if (!listFile.is_open()) {
            std::cerr << "Advertencia: No se pudo abrir la lista " << listPath << std::endl;
            continue;
        }

        std::cout << "Procesando lista: " << listName << "..." << std::endl;

        std::string imageID;
        while (listFile >> imageID) {
            // imageID suele ser algo como "000040"
            if (imageID.empty()) continue;

            // Construir rutas
            // WiderPerson: La imagen es ID.jpg, la anotacion es ID.jpg.txt
            std::string imgPath = imgDir + "/" + imageID + ".jpg";
            std::string annPath = annDir + "/" + imageID + ".jpg.txt";

            // Verificar existencia
            if (!fs::exists(imgPath)) {
                // Intento alternativo por si acaso
                continue; 
            }
            if (!fs::exists(annPath)) {
                // A veces la anotación es solo ID.txt
                annPath = annDir + "/" + imageID + ".txt";
                if (!fs::exists(annPath)) continue;
            }

            cv::Mat img = cv::imread(imgPath);
            if (img.empty()) continue;

            // Leer anotaciones
            std::ifstream annFile(annPath);
            std::vector<Box> pedestrians;
            std::string line;
            
            // Leer línea por línea
            while (std::getline(annFile, line)) {
                if (line.empty()) continue;
                std::stringstream ss(line);
                int label, x1, y1, x2, y2;
                
                // Si la línea no tiene 5 enteros (ej: el header con count), falla y salta
                if (!(ss >> label >> x1 >> y1 >> x2 >> y2)) continue;

                // Label 1: Peatones
                if (label == 1) {
                    x1 = std::max(0, x1); y1 = std::max(0, y1);
                    x2 = std::min(img.cols, x2); y2 = std::min(img.rows, y2);
                    
                    if ((x2 - x1) < 20 || (y2 - y1) < 40) continue;

                    // --- GENERAR POSITIVO ---
                    cv::Mat crop = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                    cv::Mat resized;
                    cv::resize(crop, resized, cv::Size(MODEL_W, MODEL_H));
                    
                    std::string outName = posDir + "/pos_" + imageID + "_" + std::to_string(posCount++) + ".jpg";
                    cv::imwrite(outName, resized);

                    pedestrians.push_back({x1, y1, x2, y2});
                }
            }

            // --- GENERAR NEGATIVOS (Fondo) ---
            std::uniform_int_distribution<> disX(0, std::max(0, img.cols - MODEL_W));
            std::uniform_int_distribution<> disY(0, std::max(0, img.rows - MODEL_H));

            int generatedHere = 0;
            int attempts = 0;
            
            // Solo generar negativos si la imagen es lo bastante grande
            if (img.cols >= MODEL_W && img.rows >= MODEL_H) {
                while (generatedHere < NUM_NEGATIVES_PER_IMAGE && attempts < 20) {
                    attempts++;
                    int randX = disX(gen);
                    int randY = disY(gen);
                    cv::Rect proposal(randX, randY, MODEL_W, MODEL_H);

                    bool overlaps = false;
                    for (const auto& ped : pedestrians) {
                        if (computeIoU(proposal, ped) > 0.05) { // Si toca un peatón
                            overlaps = true;
                            break;
                        }
                    }

                    if (!overlaps) {
                        cv::Mat negCrop = img(proposal);
                        std::string outName = negDir + "/neg_" + imageID + "_" + std::to_string(negCount++) + ".jpg";
                        cv::imwrite(outName, negCrop);
                        generatedHere++;
                    }
                }
            }

            filesProcessed++;
            if (filesProcessed % 100 == 0) {
                std::cout << " Archivos: " << filesProcessed << " | Pos: " << posCount << " | Neg: " << negCount << "\r" << std::flush;
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Finalizado." << std::endl;
    std::cout << "Total Archivos procesados: " << filesProcessed << std::endl;
    std::cout << "Total Positivos: " << posCount << std::endl;
    std::cout << "Total Negativos: " << negCount << std::endl;
}

int main() {
    processDataset();
    return 0;
}