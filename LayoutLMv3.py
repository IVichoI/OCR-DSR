"""
Detector Simplificado de Estructuras de Documentos con LayoutLMv3 + OpenCV + OCR
Versión reducida y optimizada del código original
"""

import os
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime
import warnings

# Configurar warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

# Importaciones LayoutLMv3
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch

# Importar OCR
from OCR import OCRIndependiente

class DetectorSimple:
    def __init__(self):
        """Inicializa el detector simplificado"""
        print("Inicializando detector")
        
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {self.device}")
        
        # Cargar modelo LayoutLMv3
        try:
            model_name = "microsoft/layoutlmv3-base"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.processor = LayoutLMv3Processor.from_pretrained(model_name)
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return None
        
        # Inicializar OCR
        self.ocr = OCRIndependiente(generar_visualizacion=False, guardar_archivos=False)
        
        # Configuración
        self.output_dir = "resultados/resultadosDsr"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analisis"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "imagenes"), exist_ok=True)
        
        # Etiquetas simplificadas
        self.labels = {0: 'TEXT', 1: 'TEXT', 2: 'TEXT', 3: 'TITLE', 4: 'TITLE', 
                      5: 'LIST', 6: 'LIST', 7: 'TABLE', 8: 'TABLE', 9: 'FIGURE', 10: 'FIGURE'}
        
        # Colores para visualización
        self.colors = {'TEXT': (0, 255, 0), 'TITLE': (255, 0, 0), 'LIST': (0, 0, 255), 
                      'TABLE': (255, 255, 0), 'FIGURE': (255, 0, 255), 'HEADER': (255, 165, 0)}
    
    def procesar_documento(self, ruta_imagen):
        """Procesa un documento completo"""
        print(f"\nProcesando: {ruta_imagen}")
        
        if not os.path.exists(ruta_imagen):
            print("Error: Imagen no encontrada")
            return None
        
        # Cargar imagen
        imagen_pil = Image.open(ruta_imagen).convert('RGB')
        imagen_cv = cv2.imread(ruta_imagen)
        
        try:
            # Paso 1: Detectar bloques con OpenCV
            bloques = self._detectar_bloques(imagen_cv)
            
            # Paso 2: Clasificar con LayoutLMv3
            estructuras = self._clasificar_bloques(imagen_pil, bloques)
            
            # Paso 3: Aplicar OCR
            resultado = self._aplicar_ocr(imagen_cv, estructuras, ruta_imagen)
            
            # Paso 4: Guardar resultados
            self._guardar_resultados(resultado, ruta_imagen)
            
            # Paso 5: Crear visualización de bloques detectados
            self._crear_visualizacion(imagen_cv, estructuras, ruta_imagen)
            
            return resultado
            
        except Exception as e:
            print(f"Error procesando: {e}")
            return None
    
    def _detectar_bloques(self, imagen):
        """Detecta bloques de texto con OpenCV"""
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Umbralización adaptativa
        umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Operaciones morfológicas para conectar texto
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        morfologia = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bloques = []
        altura_img, ancho_img = imagen.shape[:2]
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 500:  # Filtrar áreas muy pequeñas
                x, y, w, h = cv2.boundingRect(contorno)
                
                # Filtros básicos
                if w > 20 and h > 10 and area < altura_img * ancho_img * 0.8:
                    bloque = {
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'centro_y': y + h // 2
                    }
                    bloques.append(bloque)
        
        # Ordenar por posición vertical
        bloques.sort(key=lambda b: b['centro_y'])
        print(f"   Bloques detectados: {len(bloques)}")
        return bloques
    
    def _clasificar_bloques(self, imagen_pil, bloques):
        """Clasifica bloques con LayoutLMv3"""
        estructuras = []
        
        for i, bloque in enumerate(bloques):
            bbox = bloque['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extraer región
            region = imagen_pil.crop((x1, y1, x2, y2))
            
            # Clasificar con LayoutLMv3
            tipo = self._clasificar_region(region)
            
            # Clasificación heurística simple
            tipo_heuristico = self._clasificar_heuristica(bloque, imagen_pil.size, i)
            
            # Combinar resultados
            tipo_final = tipo_heuristico if tipo == 'TEXT' and tipo_heuristico != 'TEXT' else tipo
            
            estructura = {
                'tipo': tipo_final,
                'bbox': bbox,
                'confianza': 0.7
            }
            estructuras.append(estructura)
        
        print(f"   Estructuras clasificadas: {len(estructuras)}")
        return estructuras
    
    def _clasificar_region(self, region):
        """Clasifica una región con LayoutLMv3"""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                encoding = self.processor(region, return_tensors="pt", padding=True, truncation=True)
                
                # Mover al dispositivo
                for key in encoding:
                    if isinstance(encoding[key], torch.Tensor):
                        encoding[key] = encoding[key].to(self.device)
                
                # Inferencia
                with torch.no_grad():
                    outputs = self.model(**encoding)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = predictions.argmax(-1)
                
                # Obtener etiqueta más común
                labels = predicted_class.squeeze().cpu().numpy()
                if labels.ndim > 0:
                    label = np.bincount(labels).argmax()
                else:
                    label = labels.item()
                
                return self.labels.get(label, 'TEXT')
                
        except Exception:
            return 'TEXT'
    
    def _clasificar_heuristica(self, bloque, tamaño_imagen, indice):
        """Clasificación heurística simple"""
        ancho_img, altura_img = tamaño_imagen
        bbox = bloque['bbox']
        
        y_centro = (bbox[1] + bbox[3]) / 2
        ancho = bbox[2] - bbox[0]
        
        # Encabezado: arriba y ancho
        if y_centro < altura_img * 0.15 and ancho > ancho_img * 0.6:
            return 'HEADER'
        
        # Título: primeros elementos, centrados
        if indice < 2 and y_centro < altura_img * 0.3 and ancho > ancho_img * 0.4:
            return 'TITLE'
        
        # Por defecto, texto
        return 'TEXT'
    
    def _aplicar_ocr(self, imagen, estructuras, ruta_imagen):
        """Aplica OCR a las estructuras"""
        resultado = {
            'ruta_imagen': ruta_imagen,
            'timestamp': datetime.now().isoformat(),
            'estructuras': []
        }
        
        for i, estructura in enumerate(estructuras):
            bbox = estructura['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extraer región con padding
            padding = 10
            altura, ancho = imagen.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(ancho, x2 + padding)
            y2 = min(altura, y2 + padding)
            
            region = imagen[y1:y2, x1:x2]
            
            # Guardar región temporal
            temp_path = f"temp_region_{i}.png"
            cv2.imwrite(temp_path, region)
            
            try:
                # Aplicar OCR
                resultado_ocr = self.ocr.procesar_imagen(temp_path)
                texto = resultado_ocr['texto_extraido'] if resultado_ocr else ""
                
                info_estructura = {
                    'tipo': estructura['tipo'],
                    'bbox': bbox,
                    'texto': texto.strip(),
                    'num_palabras': len(texto.split()) if texto else 0
                }
                
                resultado['estructuras'].append(info_estructura)
                print(f"   OCR {i+1}: {info_estructura['num_palabras']} palabras")
                
            except Exception as e:
                print(f"Error OCR región {i}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return resultado
    
    def _guardar_resultados(self, resultado, ruta_imagen):
        """Guarda los resultados"""
        nombre_base = os.path.splitext(os.path.basename(ruta_imagen))[0]
        
        # Guardar JSON
        archivo_json = os.path.join(self.output_dir, "analisis", f"{nombre_base}_analisis.json")
        with open(archivo_json, 'w', encoding='utf-8') as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        
        # Guardar texto para traducción
        archivo_texto = os.path.join(self.output_dir, "analisis", f"{nombre_base}_texto.txt")
        with open(archivo_texto, 'w', encoding='utf-8') as f:
            for estructura in resultado['estructuras']:
                if estructura['texto']:
                    if estructura['tipo'] == 'TITLE':
                        f.write(f"\n{estructura['texto']}\n\n")
                    else:
                        f.write(f"{estructura['texto']} ")
        
        print(f"Resultados guardados: {archivo_json}, {archivo_texto}")
    
    def _crear_visualizacion(self, imagen_cv, estructuras, ruta_imagen):
        """Crea y guarda una imagen con los bloques detectados visualizados"""
        print("Creando visualización de bloques detectados...")
        
        try:
            # Crear copia de la imagen original
            imagen_visual = imagen_cv.copy()
            
            # Dibujar cada estructura detectada
            for i, estructura in enumerate(estructuras):
                bbox = estructura['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                tipo = estructura['tipo']
                
                # Obtener color según el tipo de estructura
                color = self.colors.get(tipo, (128, 128, 128))  # Gris por defecto
                
                # Dibujar rectángulo del bounding box
                cv2.rectangle(imagen_visual, (x1, y1), (x2, y2), color, 2)
                
                # Preparar etiqueta con información
                etiqueta = f"{i+1}. {tipo}"
                
                # Calcular tamaño del texto para el fondo
                (ancho_texto, alto_texto), baseline = cv2.getTextSize(
                    etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                # Dibujar fondo para el texto
                cv2.rectangle(imagen_visual, 
                            (x1, y1 - alto_texto - baseline - 5), 
                            (x1 + ancho_texto + 5, y1), 
                            color, -1)
                
                # Dibujar texto en blanco
                cv2.putText(imagen_visual, etiqueta, 
                          (x1 + 2, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Guardar imagen con visualización
            nombre_base = os.path.splitext(os.path.basename(ruta_imagen))[0]
            archivo_visual = os.path.join(self.output_dir, "imagenes", f"{nombre_base}_bloques.png")
            cv2.imwrite(archivo_visual, imagen_visual)
            
            print(f"Visualización guardada en: {archivo_visual}")
            
            # También crear una leyenda de colores
            self._crear_leyenda_colores(nombre_base)
            
        except Exception as e:
            print(f"Error creando visualización: {e}")
    
    def _crear_leyenda_colores(self, nombre_base):
        """Crea una imagen con la leyenda de colores"""
        try:
            # Crear imagen para leyenda (400x200 píxeles)
            leyenda = np.ones((200, 400, 3), dtype=np.uint8) * 255
            
            # Título
            cv2.putText(leyenda, "LEYENDA DE COLORES", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Lista de tipos y colores
            tipos_mostrar = ['TEXT', 'TITLE', 'HEADER', 'LIST', 'TABLE', 'FIGURE']
            y_pos = 60
            
            for tipo in tipos_mostrar:
                if tipo in self.colors:
                    color = self.colors[tipo]
                    
                    # Dibujar rectángulo de color
                    cv2.rectangle(leyenda, (50, y_pos-10), (80, y_pos+10), color, -1)
                    cv2.rectangle(leyenda, (50, y_pos-10), (80, y_pos+10), (0, 0, 0), 1)
                    
                    # Escribir nombre del tipo
                    cv2.putText(leyenda, tipo, (90, y_pos+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    y_pos += 25
            
            # Guardar leyenda
            archivo_leyenda = os.path.join(self.output_dir, "imagenes", f"{nombre_base}_leyenda.png")
            cv2.imwrite(archivo_leyenda, leyenda)
            
            print(f"Leyenda guardada en: {archivo_leyenda}")
            
        except Exception as e:
            print(f"Error creando leyenda: {e}")

def main():
    """Función principal"""
    print("DETECTOR SIMPLIFICADO DE ESTRUCTURAS")
    print("=" * 40)
    
    detector = DetectorSimple()
    
    if detector is None:
        print("Error inicializando detector")
        return
    
    try:
        # Procesar imagen de ejemplo
        ruta_imagen = "images/Job.jpeg"
        resultado = detector.procesar_documento(ruta_imagen)
        
        if resultado:
            print(f"\nAnálisis completado!")
            print(f"Estructuras: {len(resultado['estructuras'])}")
            
            # Mostrar resumen
            for i, est in enumerate(resultado['estructuras']):
                print(f"  {i+1}. {est['tipo']}: {est['num_palabras']} palabras")
        else:
            print("Error en el análisis")
            
    except KeyboardInterrupt:
        print("\nProceso interrumpido")
    except Exception as e:
        print(f"Error: {e}")

# Alias para compatibilidad
LayoutLMv3Detector = DetectorSimple

if __name__ == "__main__":
    main()
