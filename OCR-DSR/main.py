"""
Script Principal para Detección de Estructuras + OCR
Permite elegir entre LayoutLMv3 o detección simple
"""

import os
import sys
import argparse
from datetime import datetime

def main():
    """Función principal con opciones de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Detección de estructuras de documentos con LayoutLMv3 + OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python document_analyzer.py --image images/English.jpg
  python document_analyzer.py --image images/VentaAuto.png --verbose
        """
    )
    
    parser.add_argument(
        '--image',
        default='images/Job.jpeg',
        help='Ruta a la imagen a procesar (default: images/English.jpg)'
    )
    
    parser.add_argument(
        '--output',
        default='resultados_layoutlm',
        help='Directorio de salida (default: resultados_layoutlm)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ANALIZADOR HÍBRIDO DE ESTRUCTURAS: LAYOUTLMV3 + OPENCV + OCR")
    print("="*70)
    print(f"Imagen: {args.image}")
    print(f"Salida: {args.output}")
    print("="*60)
    
    # Verificar que la imagen existe
    if not os.path.exists(args.image):
        print(f"Error: No se encontró la imagen en {args.image}")
        sys.exit(1)
    
    # Usar solo LayoutLMv3
    try:
        print("\nInicializando LayoutLMv3...")
        from LayoutLMv3 import LayoutLMv3Detector
        detector = LayoutLMv3Detector()
        resultado = detector.procesar_documento(args.image)
    except Exception as e:
        print(f"Error con LayoutLMv3: {e}")
        sys.exit(1)
    
    # Mostrar resultados
    if resultado:
        print("\n" + "="*60)
        print("RESULTADOS DEL ANÁLISIS")
        print("="*60)
        print("Análisis completado exitosamente")
        print("Método utilizado: LayoutLMv3")
        print(f"Estructuras detectadas: {resultado['num_estructuras']}")
        
        if args.verbose:
            print(f"\nDetalle de estructuras:")
            print("-" * 40)
            total_caracteres = 0
            total_palabras = 0
            
            for i, estructura in enumerate(resultado['estructuras']):
                print(f"{i+1}. {estructura['tipo']}")
                print(f"   Dimensiones: {estructura['dimensiones_region'][0]}x{estructura['dimensiones_region'][1]} px")
                print(f"   Texto: {estructura['num_caracteres']} caracteres, {estructura['num_palabras']} palabras")
                if 'confianza_deteccion' in estructura:
                    print(f"   Confianza: {estructura['confianza_deteccion']:.2f}")
                
                total_caracteres += estructura['num_caracteres']
                total_palabras += estructura['num_palabras']
                
                # Mostrar muestra del texto si no es muy largo
                if estructura['texto_extraido'] and len(estructura['texto_extraido']) > 0:
                    muestra = estructura['texto_extraido'][:100]
                    if len(estructura['texto_extraido']) > 100:
                        muestra += "..."
                    print(f"   Muestra: {muestra}")
                print()
            
            print(f"Total: {total_caracteres} caracteres, {total_palabras} palabras")
        
        print(f"Resultados guardados en: {detector.output_dir}")
        
        # Mostrar archivos generados
        nombre_base = os.path.splitext(os.path.basename(args.image))[0]
        
        archivos_generados = [
            f"analisis/{nombre_base}_analisis.json",
            f"analisis/{nombre_base}_resumen.txt",
            f"analisis/{nombre_base}_traduccion.txt",
            f"imagenes/{nombre_base}_estructuras.png"
        ]
        
        print(f"\nArchivos generados:")
        for archivo in archivos_generados:
            ruta_completa = os.path.join(detector.output_dir, archivo)
            if os.path.exists(ruta_completa):
                print(f"  {archivo}")
            else:
                print(f"  [No encontrado] {archivo}")
    
    else:
        print("\n" + "="*60)
        print("ERROR EN EL ANÁLISIS")
        print("="*60)
        print("No se pudo procesar la imagen")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcesamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        sys.exit(1)
