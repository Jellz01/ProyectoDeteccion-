#!/bin/bash

# Detener el script si ocurre algún error
set -e

# Colores para que se vea bonito en la terminal
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== PREPARACIÓN DE DATASET (WiderPerson -> ACF) ===${NC}"
echo ""

# 1. Verificación de seguridad: ¿Existe el dataset?
if [ ! -d "Dataset/Images" ] || [ ! -d "Dataset/Annotations" ]; then
    echo -e "${RED}❌ Error: No se encuentra la carpeta 'Dataset'.${NC}"
    echo "   Asegúrate de tener la estructura:"
    echo "   ├── Dataset"
    echo "   │   ├── Images (con .jpg)"
    echo "   │   └── Annotations (con .txt)"
    exit 1
fi

# 2. Configurar directorio de compilación
echo -e "${YELLOW}🔨 Configurando entorno de compilación (CMake)...${NC}"

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Ejecutar CMake (regenerar makefiles basado en tu CMakeLists.txt actual)
cmake ..

# 3. Compilar el ejecutable 'prepare_data'
echo -e "${YELLOW}⚙️  Compilando 'prepare_data'...${NC}"
# -j$(nproc) usa todos los núcleos de tu CPU para compilar rápido
make prepare_data -j$(nproc)

# 4. Ejecutar el programa
echo ""
echo -e "${GREEN}🚀 Ejecutando preprocesamiento...${NC}"
echo "-----------------------------------------------------"

./prepare_data

echo "-----------------------------------------------------"
echo -e "${GREEN}✅ Proceso finalizado.${NC}"
echo "   Revisa la carpeta 'generated_data/positives' para ver los recortes."