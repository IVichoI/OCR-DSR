# Intalación de dependencias:
# Nombre del nuevo entorno
ENV_NAME="prueba_env"

echo -e "Paso 1: Creando nuevo entorno con Python 3.9"
conda create -n $ENV_NAME python=3.9 -y

echo -e "Paso 2: Activando entorno"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo -e "Paso 3: Actualizando conda y pip"
conda update -n base -c defaults conda -y
pip install --upgrade pip

echo -e "Paso 4: Instalando dependencias del sistema básicas"
conda install -c conda-forge numpy pillow matplotlib scipy scikit-learn -y

echo -e "Paso 5: Instalando OpenCV desde conda-forge"
conda install -c conda-forge opencv -y

echo -e "Paso 6: Instalando PyTorch (CPU)"
# Instalamos primero la versión CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo -e "Paso 7: Instalando protobuf específico"
pip install protobuf==3.20.3

echo -e "Paso 8: Instalando transformers"
pip install transformers

echo -e "Paso 9: Instalando DocTR"
pip install python-doctr[torch]

echo -e "Paso 10: Instalando dependencias adicionales"
pip install datasets sentencepiece pytesseract tqdm psutil

echo -e "Paso 11: Instalando dependencias para gráficos"
pip install seaborn plotly

echo -e "¡Instalación completada!"