"""
OCR con DocTR
Detecta y extrae texto de imágenes usando DocTR de forma autónoma
"""

import os
import sys
import cv2
import numpy as np

# Importaciones de DocTR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Configuración de matplotlib para evitar problemas de interfaz gráfica
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class OCRIndependiente:
    def __init__(self, generar_visualizacion=True, guardar_archivos=True):
        """Inicializa el sistema OCR
        
        Args:
            generar_visualizacion (bool): Si True, genera visualizaciones. Si False, solo extrae texto.
            guardar_archivos (bool): Si True, guarda archivos de texto. Si False, solo retorna el resultado.
        """
        print("Inicializando DocTR")
        
        self.generar_visualizacion = generar_visualizacion
        self.guardar_archivos = guardar_archivos
        
        try:
            # Cargar modelo DocTR preentrenado
            self.model = ocr_predictor(pretrained=True)
            print("Modelo DocTR cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo DocTR: {e}")
            sys.exit(1)
        
        # Configurar carpetas de salida
        self.output_dir = "resultados/resultadosOcr"
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Configura el directorio de salida"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "imagenes"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "textos"), exist_ok=True)
        print(f"Directorio de salida: {self.output_dir}")
    
    def procesar_imagen(self, ruta_imagen):
        """
        Procesa una imagen completa con OCR
        
        Args:
            ruta_imagen (str): Ruta a la imagen a procesar
            
        Returns:
            dict: Resultado del procesamiento con texto extraído y metadatos
        """
        print(f"\nProcesando imagen: {ruta_imagen}")
        
        # Verificar que la imagen existe
        if not os.path.exists(ruta_imagen):
            print(f"Error: No se encontró la imagen en {ruta_imagen}")
            return None
        
        # Cargar imagen para obtener información
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen desde {ruta_imagen}")
            return None
        
        altura, ancho = imagen.shape[:2]
        print(f"Dimensiones: {ancho}x{altura} píxeles")
        
        try:
            # Procesar imagen con DocTR
            print("Ejecutando DocTR")
            doc = DocumentFile.from_images(ruta_imagen)
            resultado = self.model(doc)
            
            # Extraer texto
            texto_extraido = self._extraer_texto_completo(resultado)
            
            # Información de resultado
            info_resultado = {
                'ruta_imagen': ruta_imagen,
                'dimensiones': (ancho, altura),
                'texto_extraido': texto_extraido,
                'num_caracteres': len(texto_extraido),
                'num_palabras': len(texto_extraido.split()),
                'num_lineas': len(texto_extraido.split('\n')),
                'resultado_doctr': resultado
            }
            
            print(f"OCR completado:")
            print(f"   Caracteres extraídos: {info_resultado['num_caracteres']}")
            print(f"   Palabras detectadas: {info_resultado['num_palabras']}")
            print(f"   Líneas de texto: {info_resultado['num_lineas']}")
            
            # Guardar resultados solo si está habilitado
            if self.guardar_archivos:
                self._guardar_resultados(info_resultado)
            
            # Crear visualización solo si está habilitada
            if self.generar_visualizacion:
                self._crear_visualizacion(info_resultado)
            
            return info_resultado
            
        except Exception as e:
            print(f"Error durante el procesamiento OCR: {e}")
            return None
    
    def _extraer_texto_completo(self, resultado):
        """Extrae todo el texto del resultado de DocTR"""
        lineas_texto = []
        
        if resultado.pages and len(resultado.pages) > 0:
            # Obtener estructura de texto
            datos_ocr = resultado.export()["pages"][0]["blocks"]
            
            for bloque in datos_ocr:
                for linea in bloque["lines"]:
                    # Unir todas las palabras de la línea
                    texto_linea = " ".join([palabra["value"] for palabra in linea["words"]])
                    if texto_linea.strip():  # Solo agregar líneas no vacías
                        lineas_texto.append(texto_linea.strip())
        
        return '\n'.join(lineas_texto)
    
    def _guardar_resultados(self, info_resultado):
        """Guarda los resultados en archivos"""
        nombre_base = os.path.splitext(os.path.basename(info_resultado['ruta_imagen']))[0]
        
        # Guardar solo texto puro (sin metadatos)
        archivo_texto = os.path.join(self.output_dir, "textos", f"{nombre_base}_texto.txt")
        with open(archivo_texto, 'w', encoding='utf-8') as f:
            f.write(info_resultado['texto_extraido'])
        
        print(f"Texto guardado en: {archivo_texto}")
    
    def _crear_visualizacion(self, info_resultado):
        """Crea una visualización del resultado OCR"""
        print("Creando visualización...")
        
        try:
            # Configurar matplotlib
            plt.ioff()  # Desactivar modo interactivo
            fig = plt.figure(figsize=(15, 10))
            
            # Crear visualización con DocTR
            resultado_doctr = info_resultado['resultado_doctr']
            
            try:
                # Intentar con fuente personalizada
                visualizacion = resultado_doctr.pages[0].synthesize(
                    font_family="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                )
            except:
                # Usar fuente por defecto
                visualizacion = resultado_doctr.pages[0].synthesize()
            
            # Mostrar imagen
            plt.imshow(visualizacion)
            plt.axis('off')
            
            # Añadir título con información
            nombre_imagen = os.path.basename(info_resultado['ruta_imagen'])
            titulo = f"OCR - {nombre_imagen}\n"
            titulo += f"Caracteres: {info_resultado['num_caracteres']} | "
            titulo += f"Palabras: {info_resultado['num_palabras']} | "
            titulo += f"Líneas: {info_resultado['num_lineas']}"
            
            plt.title(titulo, fontsize=12, pad=20)
            
            # Guardar visualización
            nombre_base = os.path.splitext(os.path.basename(info_resultado['ruta_imagen']))[0]
            archivo_viz = os.path.join(self.output_dir, "imagenes", f"{nombre_base}_ocr_visual.png")
            
            plt.savefig(archivo_viz, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            plt.clf()
            
            print(f"Visualización guardada en: {archivo_viz}")
            
        except Exception as e:
            print(f"No se pudo crear visualización: {e}")
        finally:
            plt.close('all')

def main():
    """Función principal para ejecutar el OCR"""
    print("OCR CON DOCTR")
    print("="*50)
    
    # Crear instancia del OCR
    ocr = OCRIndependiente()
    
    try:
        ruta_imagen = "images/Job.jpeg"
        print(f"Usando imagen por defecto: {ruta_imagen}")
        
        resultado = ocr.procesar_imagen(ruta_imagen)
        
        if resultado:
            print(f"\nProcesamiento completado exitosamente!")
            print(f"Texto extraído: {resultado['num_caracteres']} caracteres")
            print(f"Palabras detectadas: {resultado['num_palabras']}")
            print(f"Líneas de texto: {resultado['num_lineas']}")
            print(f"Resultados guardados en: {ocr.output_dir}")
            
            # Mostrar una muestra del texto extraído
            if resultado['texto_extraido']:
                print(f"\nMuestra del texto extraído:")
                print("-" * 40)
                # Mostrar solo las primeras 200 caracteres
                muestra = resultado['texto_extraido'][:200]
                print(muestra)
                if len(resultado['texto_extraido']) > 200:
                    print("...")
                print("-" * 40)
        else:
            print("\nError en el procesamiento")
            
    except KeyboardInterrupt:
        print("\n\nProcesamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")

if __name__ == "__main__":
    main()