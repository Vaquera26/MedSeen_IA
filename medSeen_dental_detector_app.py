"""
MedSeen - Sistema de Detección de Instrumentos Dentales con IA

Descripción:
Aplicación web completa desarrollada con Streamlit que implementa un sistema
de detección de instrumentos dentales en tiempo real utilizando YOLO v8.
La aplicación incluye interfaz de usuario profesional, detección por cámara web,
análisis estadístico en tiempo real, múltiples visualizaciones interactivas,
y generación automática de reportes PDF profesionales.

Características principales:
- Detección en tiempo real con cámara web
- Interfaz profesional con diseño MedSeen
- Dashboard interactivo con múltiples gráficas
- Generación automática de reportes PDF
- Sistema de navegación por pestañas
- Control de sesiones y estadísticas
- Análisis avanzado de confianza y distribución temporal

Tecnologías utilizadas:
- Streamlit (Frontend web)
- YOLO v8 (Detección de objetos)
- OpenCV (Procesamiento de video)
- Plotly (Gráficas interactivas)
- ReportLab (Generación de PDF)
- Matplotlib/Seaborn (Análisis estadístico)

Autor(es):
- Juan Fernando Vaquera Sanchez (21130869)
- Miriam Alicia Sanchez Cervantes (21130882)
- Diego Muñoz Rede (21130893)

Fecha de creación: Mayo 2025
Archivo: medSeen_dental_detector_app.py

Requisitos:
- streamlit
- ultralytics (YOLO)
- opencv-python
- plotly
- pandas
- reportlab
- matplotlib
- seaborn
- numpy

Estructura esperada:
img/logo.png (Logo de la aplicación)
runs/detect/instrumentos_dentales_yolo_model5/weights/best.pt (Modelo entrenado)

Ejecución:
streamlit run medSeen_dental_detector_app.py
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import time
from datetime import datetime, timedelta
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Configuración de la página
st.set_page_config(
    page_title="MedSeen - Detector de Instrumentos Dentales",
    page_icon="img/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores MedSeen
COLORS = {
    'primary': "#FFFFFF",      # Azul marino del logo
    'secondary': '#4FC3D7',    # Azul cyan del logo
    'accent': '#1A4A6B',       # Azul más oscuro
    'light': '#E8F4F8',        # Azul muy claro
    'white': '#FFFFFF',
    'text': '#2C3E50',
    'success': '#27AE60',
    'warning': '#F39C12',
    'error': '#E74C3C'
}

# CSS minimalista con colores MedSeen
st.markdown(f"""
<style>
    .main {{
        background-color: {COLORS['white']};
        color: {COLORS['text']};
    }}
    
    .stApp {{
        background-color: {COLORS['white']};
        color: {COLORS['text']};
    }}
    
    /* Asegurar que todo el texto sea visible */
    .stMarkdown, .stText, p, div, span {{
        color: {COLORS['text']} !important;
    }}
    
    /* Sidebar completamente blanco con texto visible */
    .css-1d391kg, .css-1y4p8pa, .stSidebar, .css-1lcbmhc, .css-1cypcdb {{
        background-color: {COLORS['white']} !important;
        color: {COLORS['text']} !important;
    }}
    
    /* Contenido del sidebar */
    .css-1lcbmhc .stMarkdown, .css-1lcbmhc .stSelectbox, .css-1lcbmhc .stSlider {{
        color: {COLORS['text']} !important;
    }}
    
    /* Labels y texto del sidebar */
    .css-1lcbmhc label, .css-1lcbmhc .stMarkdown p {{
        color: {COLORS['text']} !important;
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(43, 91, 132, 0.15);
    }}
    
    .logo-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }}
    
    .metric-card {{
        background: {COLORS['white']};
        border: 2px solid {COLORS['light']};
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(43, 91, 132, 0.08);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        border-color: {COLORS['secondary']};
        transform: translateY(-2px);
    }}
    
    .team-card {{
        background: {COLORS['white']};
        border: 2px solid {COLORS['light']};
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem;
        box-shadow: 0 4px 15px rgba(43, 91, 132, 0.1);
        transition: all 0.3s ease;
        border-left: 5px solid {COLORS['secondary']};
    }}
    
    .team-card:hover {{
        border-color: {COLORS['primary']};
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(43, 91, 132, 0.2);
    }}
    
    .process-step {{
        background: linear-gradient(135deg, {COLORS['light']}, {COLORS['white']});
        border: 2px solid {COLORS['secondary']};
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        position: relative;
    }}
    
    .step-number {{
        background: {COLORS['primary']};
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        position: absolute;
        top: -20px;
        left: 20px;
        font-size: 18px;
    }}
    
    .roboflow-section {{
        background: linear-gradient(135deg, {COLORS['white']}, {COLORS['light']});
        border: 3px solid {COLORS['secondary']};
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
    }}
    
    .stButton > button {{
        border-radius: 8px;
        border: 2px solid {COLORS['primary']};
        background: {COLORS['primary']};
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        width: 100%;
    }}
    
    .stButton > button:hover {{
        background: {COLORS['secondary']};
        border-color: {COLORS['secondary']};
        transform: translateY(-1px);
    }}
    
    .status-active {{
        background: linear-gradient(135deg, {COLORS['success']}, #2ECC71);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }}
    
    .status-inactive {{
        background: {COLORS['light']};
        color: {COLORS['text']};
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }}
    
    .detection-info {{
        background: {COLORS['white']};
        border-left: 4px solid {COLORS['secondary']};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }}
    
    .footer {{
        background: {COLORS['light']};
        color: {COLORS['text']};
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 2rem;
        font-size: 0.9rem;
        border: 1px solid {COLORS['secondary']};
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {COLORS['primary']};
    }}
    
    .metric-label {{
        color: {COLORS['text']};
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }}
    
    .nav-tabs {{
        border-bottom: 2px solid {COLORS['light']};
        margin-bottom: 2rem;
    }}
</style>
""", unsafe_allow_html=True)

class PDFGenerator:
    def __init__(self):
        self.colors_pdf = {
            'primary': colors.Color(43/255, 91/255, 132/255),
            'secondary': colors.Color(79/255, 195/255, 215/255),
            'text': colors.Color(44/255, 62/255, 80/255)
        }
    
    def generar_reporte(self, detector):
        """Genera un reporte PDF completo de la sesión"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
        
        # Estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=self.colors_pdf['primary'],
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],  
            fontSize=16,
            textColor=self.colors_pdf['primary'],
            spaceAfter=12
        )
        
        story = []
        
        # Logo y título
        try:
            if os.path.exists("img/logo.png"):
                logo = Image("img/logo.png", width=2*inch, height=2*inch)
                logo.hAlign = 'CENTER'
                story.append(logo)
                story.append(Spacer(1, 20))
        except:
            pass
        
        # Título principal
        title = Paragraph("REPORTE DE DETECCIÓN DE INSTRUMENTOS DENTALES", title_style)
        story.append(title)
        story.append(Spacer(1, 30))
        
        # Información de la sesión
        story.append(Paragraph("INFORMACIÓN DE LA SESIÓN", heading_style))
        
        session_data = [
            ['Fecha de inicio', detector.inicio_sesion.strftime("%d/%m/%Y") if detector.inicio_sesion else 'N/A'],
            ['Hora de inicio', detector.inicio_sesion.strftime("%H:%M:%S") if detector.inicio_sesion else 'N/A'],
            ['Duración estimada', self._calcular_duracion(detector)],
            ['Total de detecciones', str(len(detector.detecciones_confirmadas))],
            ['Instrumentos únicos', str(len(detector.estadisticas))],
            ['Confianza promedio', f"{self._calcular_confianza_promedio(detector):.1%}"]
        ]
        
        session_table = Table(session_data, colWidths=[3*inch, 2*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors_pdf['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, self.colors_pdf['primary'])
        ]))
        
        story.append(session_table)
        story.append(Spacer(1, 30))
        
        # Estadísticas por instrumento
        if detector.estadisticas:
            story.append(Paragraph("ESTADÍSTICAS POR INSTRUMENTO", heading_style))
            
            stats_data = [['Instrumento', 'Cantidad', 'Porcentaje']]
            total = sum(detector.estadisticas.values())
            
            for instrumento, cantidad in sorted(detector.estadisticas.items(), key=lambda x: x[1], reverse=True):
                porcentaje = (cantidad / total) * 100
                stats_data.append([instrumento, str(cantidad), f"{porcentaje:.1f}%"])
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 1*inch, 1*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors_pdf['primary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, self.colors_pdf['primary']),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 30))
        
        # Log de detecciones
        story.append(Paragraph("REGISTRO CRONOLÓGICO DE DETECCIONES", heading_style))
        
        if detector.detecciones_confirmadas:
            log_data = [['Hora', 'Instrumento', 'Confianza']]
            
            for deteccion in detector.detecciones_confirmadas[-20:]:  # Últimas 20
                log_data.append([
                    deteccion['tiempo'],
                    deteccion['instrumento'],
                    f"{deteccion['confianza']:.1%}"
                ])
            
            log_table = Table(log_data, colWidths=[1.5*inch, 2.5*inch, 1*inch])
            log_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors_pdf['secondary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, self.colors_pdf['primary']),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            story.append(log_table)
        else:
            story.append(Paragraph("No se registraron detecciones en esta sesión.", styles['Normal']))
        
        # Agregar gráficas al PDF
        if detector.estadisticas and detector.detecciones_confirmadas:
            story.append(PageBreak())
            story.append(Paragraph("ANÁLISIS GRÁFICO DETALLADO", heading_style))
            
            # Generar gráficas con matplotlib para el PDF
            self._generar_graficas_pdf(detector, story)
        
        story.append(Spacer(1, 40))
        
        # Footer
        footer_text = f"""
        <para align="center">
        <b>MedSeen - Sistema de Detección de Instrumentos Dentales</b><br/>
        Reporte generado el {datetime.now().strftime("%d/%m/%Y a las %H:%M:%S")}<br/>
        Tecnología: YOLO + OpenCV + Inteligencia Artificial
        </para>
        """
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Construir PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _calcular_duracion(self, detector):
        if not detector.inicio_sesion:
            return "N/A"
        
        # Estimar duración basada en las detecciones
        if detector.detecciones_confirmadas:
            ultima_deteccion = detector.detecciones_confirmadas[-1]['timestamp']
            duracion = ultima_deteccion - detector.inicio_sesion
        else:
            duracion = timedelta(minutes=1)  # Duración mínima estimada
        
        return str(duracion).split('.')[0]  # Remover microsegundos
    
    def _calcular_confianza_promedio(self, detector):
        if not detector.detecciones_confirmadas:
            return 0
        
        total_confianza = sum(d['confianza'] for d in detector.detecciones_confirmadas)
        return total_confianza / len(detector.detecciones_confirmadas)
    
    def _generar_graficas_pdf(self, detector, story):
        """Genera múltiples gráficas para incluir en el PDF"""
        try:
            # Configurar matplotlib para PDF
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Preparar datos
            df_stats = pd.DataFrame(list(detector.estadisticas.items()), columns=['Instrumento', 'Cantidad'])
            df_detecciones = pd.DataFrame(detector.detecciones_confirmadas)
            
            # Crear figura con múltiples subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Análisis Gráfico de Detecciones', fontsize=16, fontweight='bold')
            
            # 1. Gráfica de barras
            colors_bars = ['#2B5B84', '#4FC3D7', '#1A4A6B', '#27AE60', '#F39C12']
            bars = ax1.bar(df_stats['Instrumento'], df_stats['Cantidad'], 
                          color=colors_bars[:len(df_stats)])
            ax1.set_title('Detecciones por Instrumento', fontweight='bold')
            ax1.set_xlabel('Instrumentos')
            ax1.set_ylabel('Cantidad')
            ax1.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            # 2. Gráfica de pie
            ax2.pie(df_stats['Cantidad'], labels=df_stats['Instrumento'], autopct='%1.1f%%',
                   colors=colors_bars[:len(df_stats)], startangle=90)
            ax2.set_title('Distribución Porcentual', fontweight='bold')
            
            # 3. Línea de tiempo (agrupada por minutos)
            if len(df_detecciones) > 1:
                df_detecciones['minuto'] = df_detecciones['timestamp'].dt.floor('min')
                conteo_temporal = df_detecciones.groupby('minuto').size()
                ax3.plot(conteo_temporal.index, conteo_temporal.values, 
                        marker='o', linewidth=2, markersize=6, color='#2B5B84')
                ax3.set_title('Detecciones en el Tiempo', fontweight='bold')
                ax3.set_xlabel('Tiempo')
                ax3.set_ylabel('Detecciones por Minuto')
                ax3.tick_params(axis='x', rotation=45)
            else:
                ax3.text(0.5, 0.5, 'Datos insuficientes\npara gráfica temporal', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Detecciones en el Tiempo')
            
            # 4. Gráfica de confianza
            confianzas = [d['confianza'] for d in detector.detecciones_confirmadas]
            if confianzas:
                ax4.hist(confianzas, bins=10, color='#4FC3D7', alpha=0.7, edgecolor='black')
                ax4.axvline(np.mean(confianzas), color='red', linestyle='--', 
                           label=f'Promedio: {np.mean(confianzas):.2%}')
                ax4.set_title('Distribución de Confianza', fontweight='bold')
                ax4.set_xlabel('Nivel de Confianza')
                ax4.set_ylabel('Frecuencia')
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'Sin datos de confianza', 
                        ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            
            # Guardar gráfica como imagen temporal
            temp_img = io.BytesIO()
            plt.savefig(temp_img, format='png', dpi=150, bbox_inches='tight')
            temp_img.seek(0)
            plt.close()
            
            # Agregar imagen al PDF
            img = Image(temp_img, width=7*inch, height=5.8*inch)
            story.append(img)
            story.append(Spacer(1, 20))
            
        except Exception as e:
            story.append(Paragraph(f"Error generando gráficas: {str(e)}", getSampleStyleSheet()['Normal']))

# Detector mejorado
class MedSeenDentalDetector:
    def __init__(self, model_path, confidence=0.5, umbral_tiempo=3):
        self.model_path = model_path
        self.confidence = confidence
        self.umbral_tiempo = umbral_tiempo
        
        # Variables de control
        self.clase_en_proceso = ""
        self.tiempo_detectando = 0
        self.detecciones_confirmadas = []
        self.estadisticas = {}
        self.log_detecciones = []
        self.inicio_sesion = None
        self.model = None
        self.cap = None
        self.running = False
        self.pdf_generator = PDFGenerator()
        
    def cargar_modelo(self):
        try:
            self.model = YOLO(self.model_path)
            st.success("Modelo YOLO cargado correctamente")
            return True
        except Exception as e:
            st.error(f"Error cargando modelo: {e}")
            return False
    
    def iniciar_camara(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                st.error("No se pudo acceder a la cámara")
                return False
            
            # Configuración optimizada
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            st.success("Cámara iniciada correctamente")
            return True
            
        except Exception as e:
            st.error(f"Error con la cámara: {e}")
            return False
    
    def iniciar_sesion(self):
        if not self.cargar_modelo():
            return False
        
        if not self.iniciar_camara():
            return False
        
        # Reset completo
        self.inicio_sesion = datetime.now()
        self.detecciones_confirmadas = []
        self.estadisticas = {}
        self.log_detecciones = []
        self.clase_en_proceso = ""
        self.tiempo_detectando = 0
        self.running = True
        
        st.success("Sesión iniciada correctamente")
        return True
    
    def capturar_y_procesar(self):
        if not self.running or not self.cap or not self.model:
            return None, None
        
        try:
            # Limpiar buffer de la cámara para evitar lag
            for _ in range(2):  # Descartar algunos frames viejos
                ret, _ = self.cap.read()
                if not ret:
                    break
            
            # Capturar frame actual
            ret, frame = self.cap.read()
            if not ret:
                # Reintentar captura
                self.cap.release()
                time.sleep(0.1)
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = self.cap.read()
                if not ret:
                    return None, None
            
            # Predicción YOLO
            results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
            
            info_actual = {
                'detectando': False,
                'instrumento': '',
                'confianza': 0,
                'progreso': f"0/{self.umbral_tiempo}"
            }
            
            if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
                boxes = results[0].boxes
                clases = results[0].names
                
                if boxes.conf is not None and len(boxes.conf) > 0:
                    scores = boxes.conf.tolist()
                    class_ids = boxes.cls.tolist()
                    
                    # Mejor detección
                    max_idx = scores.index(max(scores))
                    nombre = clases[int(class_ids[max_idx])]
                    confianza = scores[max_idx]
                    
                    info_actual.update({
                        'detectando': True,
                        'instrumento': nombre,
                        'confianza': confianza
                    })
                    
                    # Lógica de confirmación
                    if nombre == self.clase_en_proceso:
                        self.tiempo_detectando += 1
                    else:
                        self.clase_en_proceso = nombre
                        self.tiempo_detectando = 1
                    
                    info_actual['progreso'] = f"{self.tiempo_detectando}/{self.umbral_tiempo}"
                    
                    # Confirmar detección
                    if self.tiempo_detectando >= self.umbral_tiempo:
                        self._confirmar_deteccion(nombre, confianza)
                        # Reducir tiempo pero no resetear completamente para mantener detección activa
                        self.tiempo_detectando = max(1, self.umbral_tiempo - 2)
            else:
                self.clase_en_proceso = ""
                self.tiempo_detectando = 0
            
            annotated = results[0].plot() if results else frame
            return annotated, info_actual
            
        except Exception as e:
            st.error(f"Error procesando frame: {e}")
            # Intentar reiniciar cámara en caso de error
            try:
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass
            return None, None
    
    def _confirmar_deteccion(self, nombre, confianza):
        timestamp = datetime.now()
        
        # Evitar duplicados cercanos
        if (not self.detecciones_confirmadas or 
            (timestamp - self.detecciones_confirmadas[-1]['timestamp']).total_seconds() > 2):
            
            deteccion = {
                'instrumento': nombre,
                'confianza': confianza,
                'tiempo': timestamp.strftime("%H:%M:%S"),
                'timestamp': timestamp
            }
            
            self.detecciones_confirmadas.append(deteccion)
            
            if nombre not in self.estadisticas:
                self.estadisticas[nombre] = 0
            self.estadisticas[nombre] += 1
            
            self.log_detecciones.append(f"{nombre} ({confianza:.1%}) - {timestamp.strftime('%H:%M:%S')}")
        
        # NO resetear la clase en proceso aquí para evitar que se pare
        # self.tiempo_detectando = 0  # REMOVIDO - esto hacía que se parara
    
    def detener_sesion(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Generar PDF automáticamente
        if self.detecciones_confirmadas:
            return self.generar_pdf_sesion()
        
        st.success("Sesión terminada")
        return None
    
    def generar_pdf_sesion(self):
        try:
            pdf_buffer = self.pdf_generator.generar_reporte(self)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reporte_dental_{timestamp}.pdf"
            
            return pdf_buffer, filename
        except Exception as e:
            st.error(f"Error generando PDF: {e}")
            return None, None

def crear_grafica_barras(df_stats):
    """Crea gráfica de barras minimalista"""
    fig = go.Figure(data=[
        go.Bar(
            x=df_stats['Instrumento'],
            y=df_stats['Cantidad'],
            marker_color=COLORS['primary'],
            text=df_stats['Cantidad'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Detecciones por Instrumento',
            'x': 0.5,
            'font': {'color': COLORS['primary'], 'size': 16}
        },
        xaxis_title="Instrumentos",
        yaxis_title="Cantidad",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': COLORS['text']},
        height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['light'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['light'])
    
    return fig

def crear_grafica_pie(df_stats):
    """Crea gráfica de pie minimalista"""
    fig = go.Figure(data=[go.Pie(
        labels=df_stats['Instrumento'],
        values=df_stats['Cantidad'],
        hole=0.3,
        marker_colors=[COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success'], COLORS['warning']]
    )])
    
    fig.update_layout(
        title={
            'text': 'Distribución de Instrumentos',
            'x': 0.5,
            'font': {'color': COLORS['primary'], 'size': 16}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': COLORS['text']},
        height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def crear_grafica_area(detecciones):
    """Crea gráfica de área de detecciones acumuladas"""
    if not detecciones:
        return None
    
    df_tiempo = pd.DataFrame(detecciones)
    df_tiempo['minuto'] = df_tiempo['timestamp'].dt.strftime('%H:%M')
    conteo_por_minuto = df_tiempo.groupby('minuto').size().reset_index(name='detecciones')
    conteo_por_minuto['acumulado'] = conteo_por_minuto['detecciones'].cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=conteo_por_minuto['minuto'],
        y=conteo_por_minuto['acumulado'],
        fill='tozeroy',
        mode='lines',
        line=dict(color=COLORS['secondary'], width=2),
        fillcolor=f"rgba({int(COLORS['secondary'][1:3], 16)}, {int(COLORS['secondary'][3:5], 16)}, {int(COLORS['secondary'][5:7], 16)}, 0.3)"
    ))
    
    fig.update_layout(
        title={
            'text': 'Detecciones Acumuladas',
            'x': 0.5,
            'font': {'color': COLORS['primary'], 'size': 16}
        },
        xaxis_title="Tiempo",
        yaxis_title="Total Acumulado",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': COLORS['text']},
        height=300,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['light'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['light'])
    
    return fig

def crear_mapa_calor(detecciones):
    """Crea mapa de calor de detecciones por hora y instrumento"""
    if not detecciones or len(detecciones) < 3:
        return None
    
    df = pd.DataFrame(detecciones)
    df['hora'] = df['timestamp'].dt.hour
    df['minuto'] = df['timestamp'].dt.minute
    df['periodo'] = df['hora'].astype(str) + ':' + (df['minuto'] // 10 * 10).astype(str).str.zfill(2)
    
    # Crear matriz de calor
    heatmap_data = df.groupby(['instrumento', 'periodo']).size().unstack(fill_value=0)
    
    if heatmap_data.empty:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[
            [0, COLORS['white']],
            [0.5, COLORS['light']],
            [1, COLORS['primary']]
        ],
        showscale=True
    ))
    
    fig.update_layout(
        title={
            'text': 'Mapa de Calor: Detecciones por Tiempo',
            'x': 0.5,
            'font': {'color': COLORS['primary'], 'size': 16}
        },
        xaxis_title="Período (Hora:Minuto)",
        yaxis_title="Instrumento",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': COLORS['text']},
        height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def crear_grafica_radar(df_stats):
    """Crea gráfica de radar para múltiples métricas"""
    if len(df_stats) < 3:
        return None
    
    # Normalizar valores para el radar
    max_val = df_stats['Cantidad'].max()
    df_stats['Normalizado'] = df_stats['Cantidad'] / max_val * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=df_stats['Normalizado'].tolist() + [df_stats['Normalizado'].iloc[0]],
        theta=df_stats['Instrumento'].tolist() + [df_stats['Instrumento'].iloc[0]],
        fill='toself',
        line_color=COLORS['primary'],
        fillcolor=f"rgba({int(COLORS['primary'][1:3], 16)}, {int(COLORS['primary'][3:5], 16)}, {int(COLORS['primary'][5:7], 16)}, 0.3)"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title={
            'text': 'Distribución Radial de Instrumentos',
            'x': 0.5,
            'font': {'color': COLORS['primary'], 'size': 16}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': COLORS['text']},
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def crear_grafica_confianza(detecciones):
    """Crea gráfica de distribución de confianza"""
    if not detecciones:
        return None
    
    confianzas = [d['confianza'] for d in detecciones]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=confianzas,
        nbinsx=10,
        marker_color=COLORS['secondary'],
        opacity=0.7
    ))
    
    # Agregar línea de promedio
    promedio = sum(confianzas) / len(confianzas)
    fig.add_vline(x=promedio, line_dash="dash", line_color=COLORS['error'],
                  annotation_text=f"Promedio: {promedio:.1%}")
    
    fig.update_layout(
        title={
            'text': 'Distribución de Niveles de Confianza',
            'x': 0.5,
            'font': {'color': COLORS['primary'], 'size': 16}
        },
        xaxis_title="Nivel de Confianza",
        yaxis_title="Frecuencia",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': COLORS['text']},
        height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['light'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['light'])
    
    return fig

def crear_dashboard_completo(detector):
    """Crea un dashboard completo con múltiples gráficas"""
    if not detector.estadisticas:
        return None
    
    df_stats = pd.DataFrame(
        list(detector.estadisticas.items()),
        columns=['Instrumento', 'Cantidad']
    )
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Barras', 'Línea Temporal', 'Distribución', 'Confianza'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "pie"}, {"type": "histogram"}]]
    )
    
    # Gráfica de barras
    fig.add_trace(
        go.Bar(x=df_stats['Instrumento'], y=df_stats['Cantidad'], 
               marker_color=COLORS['primary'], name="Detecciones"),
        row=1, col=1
    )
    
    # Línea temporal
    if detector.detecciones_confirmadas:
        df_tiempo = pd.DataFrame(detector.detecciones_confirmadas)
        df_tiempo['minuto'] = df_tiempo['timestamp'].dt.strftime('%H:%M')
        conteo_temporal = df_tiempo.groupby('minuto').size().reset_index(name='count')
        
        fig.add_trace(
            go.Scatter(x=conteo_temporal['minuto'], y=conteo_temporal['count'],
                      mode='lines+markers', line_color=COLORS['secondary'], name="Temporal"),
            row=1, col=2
        )
    
    # Gráfica de pie
    fig.add_trace(
        go.Pie(labels=df_stats['Instrumento'], values=df_stats['Cantidad'], name="Distribución"),
        row=2, col=1
    )
    
    # Histograma de confianza
    if detector.detecciones_confirmadas:
        confianzas = [d['confianza'] for d in detector.detecciones_confirmadas]
        fig.add_trace(
            go.Histogram(x=confianzas, marker_color=COLORS['accent'], name="Confianza"),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text="Dashboard Completo de Análisis",
        title_x=0.5,
        title_font_color=COLORS['primary'],
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': COLORS['text']},
        showlegend=False
    )
    
    return fig

def mostrar_proceso_desarrollo():
    """Muestra la sección de proceso de desarrollo con Roboflow"""
    st.markdown(f"""
    <div class="roboflow-section">
        <h1 style="color: {"#2C3E50"}; text-align: center; margin-bottom: 2rem;">
            🔬 Proceso de Desarrollo con Roboflow
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Pasos del proceso
    pasos_desarrollo = [
        {
            "numero": "1",
            "titulo": "Recolección de Datos",
            "descripcion": "Capturamos imágenes de alta calidad de instrumentos dentales en diferentes ángulos, iluminaciones y condiciones reales de uso.",
            "detalles": [
                "📸 500+ imágenes de instrumentos dentales",
                "🔍 Múltiples ángulos y perspectivas",
                "💡 Diferentes condiciones de iluminación",
                "📱 Imágenes desde cámara web y dispositivos móviles"
            ]
        },
        {
            "numero": "2", 
            "titulo": "Anotación en Roboflow",
            "descripcion": "Utilizamos la plataforma Roboflow para etiquetar manualmente cada instrumento dental en las imágenes.",
            "detalles": [
                "🏷️ Etiquetado manual de bounding boxes",
                "🎯 Clasificación de tipos de instrumentos",
                "✅ Revisión y validación de anotaciones",
                "📊 Control de calidad de las etiquetas"
            ]
        },
        {
            "numero": "3",
            "titulo": "Preprocesamiento de Datos",
            "descripcion": "Aplicamos técnicas de aumento de datos y preprocesamiento para mejorar la robustez del modelo.",
            "detalles": [
                "🔄 Data augmentation (rotación, escala, brillo)",
                "📏 Normalización de tamaños de imagen",
                "⚖️ Balanceo de clases",
                "🎭 Aplicación de filtros y efectos"
            ]
        },
        {
            "numero": "4",
            "titulo": "Entrenamiento YOLO",
            "descripcion": "Entrenamos el modelo YOLOv8 con nuestro dataset personalizado para detección de instrumentos dentales.",
            "detalles": [
                "🧠 Modelo YOLOv8 pre-entrenado",
                "⚡ Transfer learning para optimización",
                "📈 Métricas de precisión y recall",
                "🎯 Fine-tuning de hiperparámetros"
            ]
        },
        {
            "numero": "5",
            "titulo": "Validación y Testing",
            "descripcion": "Probamos el modelo con datos no vistos y evaluamos su rendimiento en condiciones reales.",
            "detalles": [
                "🧪 Test con datos de validación",
                "📊 Análisis de matriz de confusión",
                "🎯 Evaluación de precisión por clase",
                "🔍 Pruebas en tiempo real"
            ]
        },
        {
            "numero": "6",
            "titulo": "Implementación Final",
            "descripcion": "Integramos el modelo entrenado en nuestra aplicación Streamlit con interfaz de usuario completa.",
            "detalles": [
                "🖥️ Interfaz web con Streamlit",
                "📹 Integración con cámara web",
                "📈 Dashboard de análisis en tiempo real",
                "📄 Generación automática de reportes PDF"
            ]
        }
    ]
    
    # Mostrar cada paso
    for paso in pasos_desarrollo:
        st.markdown(f"""
        <div class="process-step">
            <div class="step-number">{paso['numero']}</div>
            <h3 style="color: {"#2C3E50"}; margin-top: 10px;">{paso['titulo']}</h3>
            <p style="color: {COLORS['text']}; font-size: 16px; margin: 15px 0;">
                {paso['descripcion']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detalles del paso
        col1, col2 = st.columns([1, 3])
        with col2:
            for detalle in paso['detalles']:
                st.markdown(f"**{detalle}**")
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Métricas del modelo
   

def mostrar_equipo_desarrollo():
    """Muestra la información del equipo de desarrollo"""
    st.markdown(f"""
    <div class="roboflow-section">
        <h1 style="color: {"#2C3E50"}; text-align: center; margin-bottom: 2rem;">
            👥 Equipo de Desarrollo MedSeen
        </h1>
        <p style="text-align: center; font-size: 18px; color: {COLORS['text']}; margin-bottom: 3rem;">
            Proyecto de Inteligencia Artificial para Detección de Instrumentos Dentales
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Información del equipo
    equipo = [
        {
            "nombre": "Juan Fernando Vaquera Sanchez",
            "control": "21130869",
            "rol": "Líder de Proyecto & Desarrollador IA",
            "especialidad": "Machine Learning, YOLO, Computer Vision",
            "contribuciones": [
                "🧠 Arquitectura del modelo YOLO",
                "📊 Implementación del sistema de detección",
                "🎯 Optimización de hiperparámetros",
                "📈 Desarrollo de métricas y análisis"
            ]
        },
        {
            "nombre": "Miriam Alicia Sanchez Cervantes",
            "control": "21130882", 
            "rol": "Desarrolladora Frontend & UX/UI",
            "especialidad": "Streamlit, Diseño de Interfaces, Experiencia de Usuario",
            "contribuciones": [
                "🎨 Diseño de la interfaz de usuario",
                "📱 Desarrollo de la aplicación web",
                "📊 Implementación de gráficas interactivas",
                "🎭 Diseño visual y paleta de colores"
            ]
        },
        {
            "nombre": "Diego Muñez Rede",
            "control": "21130893",
            "rol": "Especialista en Datos & Testing",
            "especialidad": "Roboflow, Preparación de Datos, Quality Assurance",
            "contribuciones": [
                "📸 Recolección y curación de datos",
                "🏷️ Anotación de imágenes en Roboflow",
                "🧪 Testing y validación del modelo",
                "📋 Documentación y reportes PDF"
            ]
        }
    ]
    
    # Mostrar cada miembro del equipo
    for i, miembro in enumerate(equipo):
        st.markdown(f"""
        <div class="team-card">
            <h2 style="color: {"#2C3E50"}; margin-bottom: 10px;">
                {miembro['nombre']}
            </h2>
            <h4 style="color: {COLORS['secondary']}; margin-bottom: 5px;">
                📋 No. de Control: {miembro['control']}
            </h4>
            
        </div>
        """, unsafe_allow_html=True)
        
       
        
        st.markdown("<br>", unsafe_allow_html=True)
    

    

def main():
    # Header con logo
    st.markdown(f"""
    <div class="main-header">
        <div class="logo-container">
            <img src="data:image/png;base64,{get_base64_image('img/logo.png')}" width="80" style="margin-right: 20px;">
            <div>
                <h1 style="margin: 0;">MedSeen</h1>
                <p style="margin: 5px 0 0 0;">Sistema de Detección de Instrumentos Dentales</p>
            </div>
        </div>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">Inteligencia Artificial • YOLO Detection • Análisis en Tiempo Real</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sistema de navegación con tabs
    tab_deteccion, tab_proceso, tab_equipo = st.tabs([
        "🔍 Sistema de Detección", 
        "🔬 Proceso de Desarrollo", 
        "👥 Equipo de Desarrollo"
    ])
    
    with tab_deteccion:
        # Contenido original del sistema de detección
        # Inicializar session state
        if 'detector' not in st.session_state:
            st.session_state.detector = None
        if 'detection_active' not in st.session_state:
            st.session_state.detection_active = False
        
        # Sidebar
        with st.sidebar:
            # Logo arriba del panel de control
            try:
                if os.path.exists("img/logo.png"):
                    st.image("img/logo.png", width=150)
                    st.markdown("<br>", unsafe_allow_html=True)
            except:
                st.markdown("### MedSeen")
                st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"### <span style='color: {COLORS['text']}'>Configuración</span>", unsafe_allow_html=True)
            
            model_path = "runs/detect/instrumentos_dentales_yolo_model5/weights/best.pt"
            confidence = st.slider("Nivel de Confianza", 0.1, 1.0, 0.5, 0.1)
            umbral_tiempo = st.slider("Frames para Confirmar", 1, 10, 3)
            
            st.markdown("---")
            st.markdown(f"### <span style='color: {COLORS['text']}'>Controles</span>", unsafe_allow_html=True)
            
            if not st.session_state.detection_active:
                if st.button("INICIAR DETECCIÓN", type="primary", use_container_width=True):
                    with st.spinner("Iniciando sistema..."):
                        st.session_state.detector = MedSeenDentalDetector(model_path, confidence, umbral_tiempo)
                        
                        if st.session_state.detector.iniciar_sesion():
                            st.session_state.detection_active = True
                            time.sleep(1)
                            st.rerun()
            else:
                if st.button("FINALIZAR SESIÓN", type="secondary", use_container_width=True):
                    if st.session_state.detector:
                        # Generar PDF antes de detener
                        pdf_data = st.session_state.detector.detener_sesion()
                        
                        if pdf_data:
                            pdf_buffer, filename = pdf_data
                            st.session_state.pdf_data = pdf_buffer.getvalue()
                            st.session_state.pdf_filename = filename
                    
                    st.session_state.detection_active = False
                    st.session_state.detector = None
                    st.rerun()
            
            # Estado del sistema
            st.markdown("---")
            st.markdown(f"### <span style='color: {COLORS['text']}'>Estado del Sistema</span>", unsafe_allow_html=True)
            
            if st.session_state.detection_active and st.session_state.detector:
                st.markdown('<div class="status-active">DETECCIÓN ACTIVA</div>', unsafe_allow_html=True)
                
                detector = st.session_state.detector
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    st.metric("Detecciones", len(detector.detecciones_confirmadas))
                with col_s2:
                    st.metric("Instrumentos", len(detector.estadisticas))
                
                if detector.inicio_sesion:
                    duracion = datetime.now() - detector.inicio_sesion
                    st.metric("Tiempo Activo", str(duracion).split('.')[0])
                
                # Log reciente
                if detector.log_detecciones:
                    st.markdown(f"### <span style='color: {COLORS['text']}'>Actividad Reciente</span>", unsafe_allow_html=True)
                    for log in detector.log_detecciones[-3:]:
                        st.markdown(f"<span style='color: {COLORS['text']}'>{log}</span>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-inactive">Sistema Inactivo</div>', unsafe_allow_html=True)
            
            # Footer
            st.markdown("---")
            st.markdown(f"""
            <div class="footer">
                <b>MedSeen v2.0</b><br/>
                Proyecto Académico<br/>
                Inteligencia Artificial
            </div>
            """, unsafe_allow_html=True)
        
        # Contenido principal
        if st.session_state.detection_active and st.session_state.detector:
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"### <span style='color: {COLORS['text']}'>Video en Tiempo Real</span>", unsafe_allow_html=True)
                
                frame, info = st.session_state.detector.capturar_y_procesar()
                
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if info and info['detectando']:
                        caption = f"DETECTANDO: {info['instrumento']} ({info['confianza']:.1%}) - Progreso: {info['progreso']}"
                    else:
                        caption = "DETECCIÓN ACTIVA - Buscando instrumentos..."
                    
                    st.image(frame_rgb, caption=caption, use_container_width=True)
                    
                    # Info de detección actual
                    if info and info['detectando']:
                        st.markdown(f"""
                        <div class="detection-info">
                            <strong>{info['instrumento']}</strong><br/>
                            Confianza: {info['confianza']:.1%} • Progreso: {info['progreso']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Buscando instrumentos dentales...")
                else:
                    st.error("Error capturando video")
            
            with col2:
                
                detector = st.session_state.detector
                
                # Métricas principales
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(detector.detecciones_confirmadas)}</div>
                        <div class="metric-label">Total</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(detector.estadisticas)}</div>
                        <div class="metric-label">Únicos</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    confianza_prom = 0
                    if detector.detecciones_confirmadas:
                        confianza_prom = sum(d['confianza'] for d in detector.detecciones_confirmadas) / len(detector.detecciones_confirmadas)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{confianza_prom:.0%}</div>
                        <div class="metric-label">Confianza</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Múltiples gráficas
                if detector.estadisticas:
                    df_stats = pd.DataFrame(
                        list(detector.estadisticas.items()),
                        columns=['Instrumento', 'Cantidad']
                    )
                    
                    # Dashboard completo
                    st.markdown("#### Dashboard Completo")
                    fig_dashboard = crear_dashboard_completo(detector)
                    if fig_dashboard:
                        st.plotly_chart(fig_dashboard, use_container_width=True)
                    
                    # Gráficas individuales en tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Barras", "🥧 Pie", "📈 Área", "🔥 Calor", "🎯 Radar"])
                    
                    with tab1:
                        fig_barras = crear_grafica_barras(df_stats)
                        st.plotly_chart(fig_barras, use_container_width=True)
                    
                    with tab2:
                        fig_pie = crear_grafica_pie(df_stats)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with tab3:
                        if len(detector.detecciones_confirmadas) > 2:
                            fig_area = crear_grafica_area(detector.detecciones_confirmadas)
                            if fig_area:
                                st.plotly_chart(fig_area, use_container_width=True)
                        else:
                            st.info("Se necesitan más detecciones para la gráfica de área")
                    
                    with tab4:
                        if len(detector.detecciones_confirmadas) > 3:
                            fig_calor = crear_mapa_calor(detector.detecciones_confirmadas)
                            if fig_calor:
                                st.plotly_chart(fig_calor, use_container_width=True)
                        else:
                            st.info("Se necesitan más detecciones para el mapa de calor")
                    
                    with tab5:
                        if len(df_stats) >= 3:
                            fig_radar = crear_grafica_radar(df_stats)
                            if fig_radar:
                                st.plotly_chart(fig_radar, use_container_width=True)
                        else:
                            st.info("Se necesitan al menos 3 instrumentos para el radar")
                    
                    # Gráfica de confianza adicional
                    if detector.detecciones_confirmadas:
                        st.markdown("#### Análisis de Confianza")
                        fig_confianza = crear_grafica_confianza(detector.detecciones_confirmadas)
                        if fig_confianza:
                            st.plotly_chart(fig_confianza, use_container_width=True)
                else:
                    st.info("Sin datos para graficar")
                
                # Log de actividad
                st.markdown(f"### <span style='color: {COLORS['text']}'>Registro de Actividad</span>", unsafe_allow_html=True)
                if detector.log_detecciones:
                    with st.container():
                        for log in detector.log_detecciones[-5:]:
                            st.markdown(f"<small style='color: {COLORS['text']}'>{log}</small>", unsafe_allow_html=True)
                else:
                    st.info("Sin actividad registrada")
            
            # Auto-refresh con control de estabilidad
            if st.session_state.detection_active:
                # Refresh más lento cuando hay detección activa para mayor estabilidad
                time.sleep(0.5)
            else:
                time.sleep(0.1)
            st.rerun()
        
        else:
            # Pantalla de inicio
            st.markdown(f"## <span style='color: {COLORS['text']}'>Sistema Listo para Detectar</span>", unsafe_allow_html=True)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {"#2C3E50"};">Video Estable</h3>
                    <p>Transmisión fluida sin interrupciones como videollamada profesional</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {"#2C3E50"};">IA Avanzada</h3>
                    <p>Detección YOLO en tiempo real con alta precisión</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_info3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {"#2C3E50"};">Reportes PDF</h3>
                    <p>Generación automática de reportes profesionales</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"### <span style='color: {COLORS['text']}'>Instrucciones de Uso</span>", unsafe_allow_html=True)
            
            instrucciones = [
                "Configura los parámetros de detección en el panel lateral",
                "Presiona INICIAR DETECCIÓN para cargar el modelo y activar la cámara", 
                "Observa el video en tiempo real con detecciones automáticas",
                "Revisa las estadísticas y gráficas que se actualizan en vivo",
                "Presiona FINALIZAR SESIÓN para generar el reporte PDF automático"
            ]
            
            for i, instruccion in enumerate(instrucciones, 1):
                st.markdown(f"<span style='color: {COLORS['text']}'><strong>{i}.</strong> {instruccion}</span>", unsafe_allow_html=True)
        
        # Descarga de PDF si está disponible
        if hasattr(st.session_state, 'pdf_data') and st.session_state.pdf_data:
            st.markdown("---")
            st.markdown("## Reporte de Sesión Generado")
            
            col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
            
            with col_pdf2:
                st.success("Reporte PDF generado exitosamente")
                
                st.download_button(
                    label="DESCARGAR REPORTE PDF",
                    data=st.session_state.pdf_data,
                    file_name=st.session_state.pdf_filename,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
                
                if st.button("Nueva Sesión", use_container_width=True):
                    # Limpiar datos de PDF
                    if hasattr(st.session_state, 'pdf_data'):
                        del st.session_state.pdf_data
                    if hasattr(st.session_state, 'pdf_filename'):
                        del st.session_state.pdf_filename
                    st.rerun()
    
    with tab_proceso:
        mostrar_proceso_desarrollo()
    
    with tab_equipo:
        mostrar_equipo_desarrollo()

def get_base64_image(image_path):
    """Convierte imagen a base64 para embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        # Crear un placeholder simple si no se encuentra la imagen
        return ""

if __name__ == "__main__":
    main()