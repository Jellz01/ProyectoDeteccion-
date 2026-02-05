#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <chrono>
#include <curl/curl.h>
#include <numeric>
#include <deque>

using namespace cv;
using namespace std;

// --- CONFIGURACIÓN ---
const string API_URL = "http://localhost:8000/detect";
const int CAPTURE_COOLDOWN_MS = 2500;
auto lastCaptureTime = chrono::steady_clock::now();

// --- FUNCIONES DE TELEMETRÍA ---

struct CPUStats {
    long long user, nice, system, idle, iowait, irq, softirq, steal;
};

CPUStats getCPUStats() {
    CPUStats stats;
    ifstream file("/proc/stat");
    string cpu;
    file >> cpu >> stats.user >> stats.nice >> stats.system >> stats.idle >> stats.iowait >> stats.irq >> stats.softirq >> stats.steal;
    return stats;
}

double calculateCPUUsage(CPUStats &prev, CPUStats &curr) {
    long long prevIdle = prev.idle + prev.iowait;
    long long currIdle = curr.idle + curr.iowait;
    long long prevNonIdle = prev.user + prev.nice + prev.system + prev.irq + prev.softirq + prev.steal;
    long long currNonIdle = curr.user + curr.nice + curr.system + curr.irq + curr.softirq + curr.steal;
    long long prevTotal = prevIdle + prevNonIdle;
    long long currTotal = currIdle + currNonIdle;
    double totalDiff = (double)(currTotal - prevTotal);
    double idleDiff = (double)(currIdle - prevIdle);
    return (totalDiff > 0) ? (totalDiff - idleDiff) / totalDiff * 100.0 : 0.0;
}

double getMemoryUsage() {
    string line;
    ifstream status_file("/proc/self/status");
    while (getline(status_file, line)) {
        if (line.find("VmRSS:") != string::npos) {
            size_t start = line.find_first_of("0123456789");
            size_t end = line.find_last_of("0123456789");
            return stod(line.substr(start, end - start + 1)) / 1024.0;
        }
    }
    return 0.0;
}

// --- ANÁLISIS DE ILUMINACIÓN ---
struct LightingAnalysis {
    double meanBrightness;
    double stdDevBrightness;
    bool isBacklit;
    bool isOverexposed;
    bool isUnderexposed;
    bool hasHighContrast;
    double dynamicRange;
};

LightingAnalysis analyzeLighting(const Mat& gray) {
    LightingAnalysis result;
    
    // Calcular estadísticas básicas
    Scalar mean, stddev;
    meanStdDev(gray, mean, stddev);
    result.meanBrightness = mean[0];
    result.stdDevBrightness = stddev[0];
    
    // Calcular histograma
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    // Normalizar histograma
    normalize(hist, hist, 0, gray.rows * gray.cols, NORM_MINMAX);
    
    // Análisis de distribución de luminosidad
    float darkPixels = 0, brightPixels = 0, midPixels = 0;
    for (int i = 0; i < 85; i++) darkPixels += hist.at<float>(i);
    for (int i = 85; i < 170; i++) midPixels += hist.at<float>(i);
    for (int i = 170; i < 256; i++) brightPixels += hist.at<float>(i);
    
    float totalPixels = gray.rows * gray.cols;
    
    // Detección de contraluz (backlight)
    // Hay mucha luminosidad en los bordes y oscuridad en el centro
    Rect centerROI(gray.cols * 0.3, gray.rows * 0.3, gray.cols * 0.4, gray.rows * 0.4);
    Rect edgeROI1(0, 0, gray.cols, gray.rows * 0.2); // Top edge
    Mat centerRegion = gray(centerROI);
    Mat edgeRegion = gray(edgeROI1);
    
    double centerMean = cv::mean(centerRegion)[0];
    double edgeMean = cv::mean(edgeRegion)[0];
    
    result.isBacklit = (edgeMean - centerMean > 50) && (brightPixels / totalPixels > 0.25);
    
    // Detección de sobreexposición
    result.isOverexposed = (result.meanBrightness > 200) || (brightPixels / totalPixels > 0.4);
    
    // Detección de subexposición
    result.isUnderexposed = (result.meanBrightness < 60) || (darkPixels / totalPixels > 0.5);
    
    // Detección de alto contraste
    result.hasHighContrast = result.stdDevBrightness > 60;
    
    // Rango dinámico
    double minVal, maxVal;
    minMaxLoc(gray, &minVal, &maxVal);
    result.dynamicRange = maxVal - minVal;
    
    return result;
}

// --- CORRECCIÓN AUTOMÁTICA DE ILUMINACIÓN ---
Mat correctLighting(const Mat& input, const LightingAnalysis& analysis) {
    Mat corrected = input.clone();
    
    // Si hay contraluz, aplicar ecualización adaptativa más agresiva
    if (analysis.isBacklit) {
        Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
        clahe->apply(corrected, corrected);
        
        // Además, aplicar corrección gamma para levantar sombras
        Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        double gamma = 0.7; // Gamma < 1 levanta sombras
        for(int i = 0; i < 256; ++i) {
            p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        LUT(corrected, lookUpTable, corrected);
    }
    // Si está sobreexpuesto, reducir brillo
    else if (analysis.isOverexposed) {
        Ptr<CLAHE> clahe = createCLAHE(1.0, Size(16, 16));
        clahe->apply(corrected, corrected);
        corrected = corrected * 0.8; // Reducir 20% brillo
    }
    // Si está subexpuesto, aumentar brillo
    else if (analysis.isUnderexposed) {
        Ptr<CLAHE> clahe = createCLAHE(2.5, Size(8, 8));
        clahe->apply(corrected, corrected);
        
        // Gamma para aclarar
        Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        double gamma = 1.3; // Gamma > 1 aclara
        for(int i = 0; i < 256; ++i) {
            p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        LUT(corrected, lookUpTable, corrected);
    }
    // Condiciones normales
    else {
        Ptr<CLAHE> clahe = createCLAHE(1.5, Size(8, 8));
        clahe->apply(corrected, corrected);
    }
    
    return corrected;
}

// --- API SEND ---
size_t write_callback(void *ptr, size_t size, size_t nmemb, void *userdata) { return size * nmemb; }

void sendToAPI(Mat frame) {
    CURL *curl = curl_easy_init();
    if(curl) {
        vector<uchar> buf;
        imencode(".jpg", frame, buf, {IMWRITE_JPEG_QUALITY, 85});
        curl_mime *form = curl_mime_init(curl);
        curl_mimepart *field = curl_mime_addpart(form);
        curl_mime_name(field, "file");
        curl_mime_data(field, (const char*)buf.data(), buf.size());
        curl_mime_filename(field, "detection.jpg");
        curl_mime_type(field, "image/jpeg");
        curl_easy_setopt(curl, CURLOPT_URL, API_URL.c_str());
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 2L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_perform(curl);
        curl_mime_free(form);
        curl_easy_cleanup(curl);
    }
}

// --- FILTRO DE DETECCIONES ---
bool isValidDetection(Rect r, double weight, int frameWidth, int frameHeight) {
    if (weight < 0.8) return false;
    
    int minWidth = frameWidth * 0.08;
    int maxWidth = frameWidth * 0.7;
    int minHeight = frameHeight * 0.15;
    int maxHeight = frameHeight * 0.9;
    
    if (r.width < minWidth || r.width > maxWidth) return false;
    if (r.height < minHeight || r.height > maxHeight) return false;
    
    float aspectRatio = (float)r.height / (float)r.width;
    if (aspectRatio < 1.2 || aspectRatio > 4.0) return false;
    
    int minArea = (frameWidth * frameHeight) * 0.02;
    if (r.area() < minArea) return false;
    
    return true;
}

// --- SUPRESIÓN NO MÁXIMA MEJORADA ---
vector<Rect> improvedNMS(vector<Rect> &boxes, vector<double> &weights, float overlapThresh = 0.3) {
    if (boxes.empty()) return {};
    
    vector<Rect> result;
    vector<int> indices(boxes.size());
    iota(indices.begin(), indices.end(), 0);
    
    sort(indices.begin(), indices.end(), [&weights](int i1, int i2) {
        return weights[i1] > weights[i2];
    });
    
    vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        result.push_back(boxes[idx]);
        
        for (size_t j = i + 1; j < indices.size(); j++) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            Rect intersection = boxes[idx] & boxes[idx2];
            float iou = (float)intersection.area() / 
                       (float)(boxes[idx].area() + boxes[idx2].area() - intersection.area());
            
            if (iou > overlapThresh) {
                suppressed[idx2] = true;
            }
        }
    }
    
    return result;
}

int main() {
    VideoCapture cap;
    
    // --- LÓGICA DE APERTURA ROBUSTA ---
    int backends[] = {CAP_ANY, CAP_V4L2};
    int indices[] = {0, 2, 1};
    bool is_opened = false;

    cout << "🔍 Buscando cámara disponible..." << endl;

    for (int b : backends) {
        for (int i : indices) {
            cout << "Probando Indice: " << i << " con Backend: " << (b == CAP_ANY ? "ANY" : "V4L2") << "..." << endl;
            cap.open(i, b);
            if (cap.isOpened()) {
                is_opened = true;
                break;
            }
        }
        if (is_opened) break;
    }

    if (!is_opened) {
        cerr << "❌ ERROR FATAL: No se detectó ninguna cámara activa." << endl;
        cerr << "💡 Solución rápida en terminal: sudo chmod 666 /dev/video*" << endl;
        return -1;
    }

    // Configuración de imagen
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    
    // Intentar desactivar auto-ajustes que pueden causar problemas
    cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25);  // Auto exposure parcial
    cap.set(CAP_PROP_AUTOFOCUS, 0);          // Desactivar autofocus
    cap.set(CAP_PROP_AUTO_WB, 1);            // Mantener white balance auto

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
    Ptr<SIFT> sift = SIFT::create(100);
    vector<KeyPoint> keypoints;
    
    TickMeter tm;
    Mat frame, gray, blurred, corrected;
    int frameCounter = 0;
    CPUStats prevCPU = getCPUStats();
    double cpuUsage = 0.0;
    
    // Historial de condiciones de luz para suavizar cambios
    deque<double> brightnessHistory;
    const int HISTORY_SIZE = 10;

    curl_global_init(CURL_GLOBAL_ALL);

    cout << "✅ ¡Cámara conectada con éxito!" << endl;
    cout << "🎯 Sistema adaptativo de iluminación activado" << endl;
    cout << "💡 Compensación de contraluz habilitada" << endl;

    while (true) {
        tm.start();
        cap >> frame;
        if (frame.empty()) {
            cerr << "Frame vacío, reintentando..." << endl;
            continue;
        }

        flip(frame, frame, 1);
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // ===== ANÁLISIS DE ILUMINACIÓN =====
        LightingAnalysis lighting = analyzeLighting(gray);
        
        // Mantener historial de brillo para suavizar
        brightnessHistory.push_back(lighting.meanBrightness);
        if (brightnessHistory.size() > HISTORY_SIZE) {
            brightnessHistory.pop_front();
        }
        
        // Calcular brillo promedio suavizado
        double smoothedBrightness = accumulate(brightnessHistory.begin(), 
                                               brightnessHistory.end(), 0.0) / brightnessHistory.size();
        
        // ===== CORRECCIÓN ADAPTATIVA =====
        // Aplicar desenfoque gaussiano para reducir ruido
        GaussianBlur(gray, blurred, Size(5, 5), 0);
        
        // Aplicar corrección según condiciones de luz
        corrected = correctLighting(blurred, lighting);

        // ===== DETECCIÓN SIFT =====
        if (frameCounter % 5 == 0) sift->detect(corrected, keypoints);
        drawKeypoints(frame, keypoints, frame, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);

        // ===== DETECCIÓN HOG =====
        vector<Rect> found;
        vector<double> weights;
        
        hog.detectMultiScale(corrected, found, weights, 0, Size(8,8), Size(32,32), 1.05, 2);

        // ===== FILTRADO ESTRICTO =====
        vector<Rect> validBoxes;
        vector<double> validWeights;
        int rejected = 0;
        
        for (size_t i = 0; i < found.size(); i++) {
            if (isValidDetection(found[i], weights[i], frame.cols, frame.rows)) {
                validBoxes.push_back(found[i]);
                validWeights.push_back(weights[i]);
            } else {
                rejected++;
            }
        }
        
        // Aplicar NMS mejorada
        vector<Rect> finalBoxes = improvedNMS(validBoxes, validWeights, 0.3);

        auto currentTime = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(currentTime - lastCaptureTime).count();

        for (size_t i = 0; i < finalBoxes.size(); i++) {
            Scalar color = Scalar(0, 255, 0);
            rectangle(frame, finalBoxes[i], color, 3);
            
            auto it = find(validBoxes.begin(), validBoxes.end(), finalBoxes[i]);
            if (it != validBoxes.end()) {
                int idx = distance(validBoxes.begin(), it);
                string confText = "Conf: " + to_string(validWeights[idx]).substr(0, 4);
                putText(frame, confText, Point(finalBoxes[i].x, finalBoxes[i].y - 5), 
                        FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
                
                if (elapsed >= CAPTURE_COOLDOWN_MS) {
                    Rect safeROI = finalBoxes[i] & Rect(0, 0, frame.cols, frame.rows);
                    if (safeROI.width > 0 && safeROI.height > 0) {
                        sendToAPI(frame(safeROI)); 
                        lastCaptureTime = currentTime;
                        cout << "📡 API Sent | Conf: " << validWeights[idx] << endl;
                    }
                }
            }
        }

        tm.stop();
        
        if (frameCounter % 10 == 0) {
            CPUStats currCPU = getCPUStats();
            cpuUsage = calculateCPUUsage(prevCPU, currCPU);
            prevCPU = currCPU;
        }

        // ===== PANEL DE TELEMETRÍA EXPANDIDO =====
        rectangle(frame, Rect(5, 5, 290, 210), Scalar(0,0,0), -1);
        
        putText(frame, "FPS: " + to_string((int)tm.getFPS()), Point(15, 25), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        putText(frame, "CPU: " + to_string(cpuUsage).substr(0, 4) + " %", Point(15, 50), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        putText(frame, "RAM: " + to_string(getMemoryUsage()).substr(0, 5) + " MB", Point(15, 75), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        
        // Información de iluminación
        putText(frame, "--- LIGHTING ---", Point(15, 100), 
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(100, 200, 255), 1);
        putText(frame, "Brightness: " + to_string((int)smoothedBrightness), Point(15, 120), 
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(200, 200, 200), 1);
        
        Scalar statusColor = Scalar(0, 255, 0);
        string status = "OK";
        if (lighting.isBacklit) {
            status = "BACKLIT!";
            statusColor = Scalar(0, 165, 255);
        } else if (lighting.isOverexposed) {
            status = "OVEREXP";
            statusColor = Scalar(0, 100, 255);
        } else if (lighting.isUnderexposed) {
            status = "UNDEREXP";
            statusColor = Scalar(255, 100, 0);
        }
        putText(frame, "Status: " + status, Point(15, 140), 
                FONT_HERSHEY_SIMPLEX, 0.4, statusColor, 1);
        
        // Información de detección
        putText(frame, "--- DETECTION ---", Point(15, 165), 
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(100, 200, 255), 1);
        putText(frame, "Valid: " + to_string(finalBoxes.size()), Point(15, 185), 
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
        putText(frame, "Rejected: " + to_string(rejected), Point(15, 205), 
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 100, 255), 1);

        imshow("Webcam Monitor", frame);
        
        frameCounter++;
        tm.reset();

        if (waitKey(1) == 27) break; 
    }

    curl_global_cleanup();
    cap.release();
    destroyAllWindows();
    return 0;
}