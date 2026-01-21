# medical_processor_gui.py version ligera

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import psutil

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:
    pydicom = None

# Intentar importar pynvml para monitoreo de GPU NVIDIA
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MedicalImageProcessor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Procesador de Imágenes Médicas")
        self.geometry("1400x900")
        self.original_image = None
        self.processed_image = None
        self.cv_original = None
        self.file_type = ""
        self.dicom_metadata = {}
        self.zoom_factor = 1.0
        self.max_zoom = 4.0
        self.min_zoom = 0.5

        # Parámetros iniciales
        self.clahe_clip_limit = 2.0
        self.clahe_tile_size = 8
        self.guided_radius = 5
        self.guided_eps = 0.01
        self.unsharp_amount = 1.5
        self.unsharp_radius = 2
        self.unsharp_threshold = 0
        
        # Control de hilos activos
        self.active_threads = 0
        self.thread_lock = threading.Lock()
        self.processing_thread = None
        
        # Configuración para CPUs limitadas
        self.detect_cpu_capabilities()
        
        # Variable para cancelar procesamiento pendiente
        self.cancel_processing = False

        self.create_widgets()
        if self.enable_realtime_metrics:
            self.start_metrics_monitor()
        else:
            self.update_static_metrics()

    def detect_cpu_capabilities(self):
        """Detecta capacidades de CPU y ajusta configuración"""
        cpu_count = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        # Si tiene 2 o menos hilos lógicos, optimizar para bajo rendimiento
        if cpu_count <= 2:
            self.low_performance_mode = True
            self.enable_realtime_metrics = False  # Desactivar monitor en tiempo real
            self.metrics_update_interval = 2000  # Actualizar cada 2 segundos
            # Limitar número de hilos de OpenCV
            cv2.setNumThreads(1)
        else:
            self.low_performance_mode = False
            self.enable_realtime_metrics = True
            self.metrics_update_interval = 500
            # Dejar que OpenCV use hilos automáticamente
            cv2.setNumThreads(max(1, cpu_count - 1))
        
        self.cpu_info = {
            'logical': cpu_count,
            'physical': physical_cores,
            'mode': 'Bajo consumo' if self.low_performance_mode else 'Normal'
        }

    def create_widgets(self):
        # Panel izquierdo: controles - con scrollbar
        self.control_frame = ctk.CTkScrollableFrame(self, width=300)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)

        title = ctk.CTkLabel(self.control_frame, text="Procesador de Imágenes Médicas", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        self.upload_btn = ctk.CTkButton(self.control_frame, text="Cargar imagen", command=self.load_image)
        self.upload_btn.pack(pady=5)

        self.file_label = ctk.CTkLabel(self.control_frame, text="", font=("Arial", 10))
        self.file_label.pack()

        # Frame con scroll para metadata DICOM
        self.metadata_frame = ctk.CTkFrame(self.control_frame, height=150)
        self.metadata_frame.pack(pady=5, fill="x", padx=5)
        
        self.metadata_text = ctk.CTkTextbox(
            self.metadata_frame, 
            height=150, 
            font=("Arial", 9),
            wrap="word",
            state="disabled"
        )
        self.metadata_text.pack(fill="both", expand=True)

        self.reset_btn = ctk.CTkButton(self.control_frame, text="Resetear Parámetros", command=self.reset_params)
        self.reset_btn.pack(pady=10)
        self.reset_btn.configure(state="disabled")

        # Métricas de rendimiento
        metrics_height = 100 if self.low_performance_mode else 120
        self.metrics_label = ctk.CTkLabel(
            self.control_frame,
            text=self.get_initial_metrics_text(),
            justify="left",
            font=("Arial", 9),
            height=metrics_height
        )
        self.metrics_label.pack(pady=10, fill="x", padx=5)

        self.create_sliders()

        self.download_btn = ctk.CTkButton(self.control_frame, text="Descargar Imagen", command=self.download_image)
        self.download_btn.pack(pady=10)
        self.download_btn.configure(state="disabled")

        # Botones de zoom
        self.zoom_frame = ctk.CTkFrame(self.control_frame)
        self.zoom_frame.pack(pady=10, fill="x")
        ctk.CTkLabel(self.zoom_frame, text="Zoom:", font=("Arial", 10)).pack(side="left", padx=(0, 5))
        self.zoom_in_btn = ctk.CTkButton(self.zoom_frame, text="+", width=30, command=self.zoom_in)
        self.zoom_in_btn.pack(side="left", padx=2)
        self.zoom_out_btn = ctk.CTkButton(self.zoom_frame, text="-", width=30, command=self.zoom_out)
        self.zoom_out_btn.pack(side="left", padx=2)
        self.zoom_reset_btn = ctk.CTkButton(self.zoom_frame, text="Reset", width=50, command=self.reset_zoom)
        self.zoom_reset_btn.pack(side="left", padx=2)

        # Panel derecho: imágenes
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Contenedor para títulos
        self.titles_frame = ctk.CTkFrame(self.image_frame, fg_color="transparent")
        self.titles_frame.pack(side="top", fill="x", padx=10, pady=(0, 5))

        orig_title = ctk.CTkLabel(self.titles_frame, text="Original", font=("Arial", 12, "bold"))
        orig_title.pack(side="left", expand=True)

        proc_title = ctk.CTkLabel(self.titles_frame, text="Procesada", font=("Arial", 12, "bold"))
        proc_title.pack(side="right", expand=True)

        # Canvas para imágenes
        self.canvas_orig = ctk.CTkCanvas(self.image_frame, bg="#2b2b2b")
        self.canvas_proc = ctk.CTkCanvas(self.image_frame, bg="#2b2b2b")

        self.canvas_orig.pack(side="left", fill="both", expand=True, padx=(0, 5), pady=(0, 10))
        self.canvas_proc.pack(side="right", fill="both", expand=True, padx=(5, 0), pady=(0, 10))

        # Eventos de zoom
        self.canvas_orig.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas_proc.bind("<MouseWheel>", self.on_mouse_wheel)

        self.bind("<Configure>", self.on_resize)

    def get_initial_metrics_text(self):
        """Texto inicial de métricas"""
        base_text = (
            f"Modo: {self.cpu_info['mode']}\n"
            f"CPU: --\n"
            f"Núcleos: {self.cpu_info['physical']}F/{self.cpu_info['logical']}L\n"
        )
        
        if not self.low_performance_mode:
            base_text += (
                f"Hilos procesamiento: --\n"
                f"Hilos totales: --\n"
            )
        
        base_text += f"RAM: -- MB"
        
        if GPU_AVAILABLE:
            base_text += f"\nGPU: -- MB"
        
        return base_text

    def create_sliders(self):
        # CLAHE
        ctk.CTkLabel(self.control_frame, text="CLAHE", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.clahe_clip_slider = ctk.CTkSlider(self.control_frame, from_=1, to=5, number_of_steps=40, command=self.update_clahe_clip)
        self.clahe_clip_slider.set(2.0)
        self.clahe_clip_slider.pack(fill="x", padx=10)
        self.clahe_clip_label = ctk.CTkLabel(self.control_frame, text="Clip Limit: 2.0")
        self.clahe_clip_label.pack(anchor="w", padx=20)

        self.clahe_tile_slider = ctk.CTkSlider(self.control_frame, from_=4, to=16, number_of_steps=6, command=self.update_clahe_tile)
        self.clahe_tile_slider.set(8)
        self.clahe_tile_slider.pack(fill="x", padx=10)
        self.clahe_tile_label = ctk.CTkLabel(self.control_frame, text="Tile Size: 8x8")
        self.clahe_tile_label.pack(anchor="w", padx=20)

        # Filtro Guiado
        ctk.CTkLabel(self.control_frame, text="Filtro Guiado", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.guided_radius_slider = ctk.CTkSlider(self.control_frame, from_=1, to=15, number_of_steps=14, command=self.update_guided_radius)
        self.guided_radius_slider.set(5)
        self.guided_radius_slider.pack(fill="x", padx=10)
        self.guided_radius_label = ctk.CTkLabel(self.control_frame, text="Radio: 5")
        self.guided_radius_label.pack(anchor="w", padx=20)

        self.guided_eps_slider = ctk.CTkSlider(self.control_frame, from_=0.001, to=0.1, number_of_steps=99, command=self.update_guided_eps)
        self.guided_eps_slider.set(0.01)
        self.guided_eps_slider.pack(fill="x", padx=10)
        self.guided_eps_label = ctk.CTkLabel(self.control_frame, text="Epsilon: 0.010")
        self.guided_eps_label.pack(anchor="w", padx=20)

        # Unsharp Mask
        ctk.CTkLabel(self.control_frame, text="Unsharp Mask", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.unsharp_amount_slider = ctk.CTkSlider(self.control_frame, from_=0, to=3, number_of_steps=30, command=self.update_unsharp_amount)
        self.unsharp_amount_slider.set(1.5)
        self.unsharp_amount_slider.pack(fill="x", padx=10)
        self.unsharp_amount_label = ctk.CTkLabel(self.control_frame, text="Amount: 1.5")
        self.unsharp_amount_label.pack(anchor="w", padx=20)

        self.unsharp_radius_slider = ctk.CTkSlider(self.control_frame, from_=1, to=10, number_of_steps=9, command=self.update_unsharp_radius)
        self.unsharp_radius_slider.set(2)
        self.unsharp_radius_slider.pack(fill="x", padx=10)
        self.unsharp_radius_label = ctk.CTkLabel(self.control_frame, text="Radio: 2")
        self.unsharp_radius_label.pack(anchor="w", padx=20)

        self.unsharp_threshold_slider = ctk.CTkSlider(self.control_frame, from_=0, to=50, number_of_steps=50, command=self.update_unsharp_threshold)
        self.unsharp_threshold_slider.set(0)
        self.unsharp_threshold_slider.pack(fill="x", padx=10)
        self.unsharp_threshold_label = ctk.CTkLabel(self.control_frame, text="Threshold: 0")
        self.unsharp_threshold_label.pack(anchor="w", padx=20)

    def update_clahe_clip(self, value):
        self.clahe_clip_limit = float(value)
        self.clahe_clip_label.configure(text=f"Clip Limit: {value}")
        self.schedule_process()

    def update_clahe_tile(self, value):
        val = int(round(float(value) / 2) * 2)
        self.clahe_tile_size = max(4, min(16, val))
        self.clahe_tile_slider.set(self.clahe_tile_size)
        self.clahe_tile_label.configure(text=f"Tile Size: {self.clahe_tile_size}x{self.clahe_tile_size}")
        self.schedule_process()

    def update_guided_radius(self, value):
        self.guided_radius = int(round(float(value)))
        self.guided_radius_label.configure(text=f"Radio: {self.guided_radius}")
        self.schedule_process()

    def update_guided_eps(self, value):
        self.guided_eps = float(value)
        self.guided_eps_label.configure(text=f"Epsilon: {value:.3f}")
        self.schedule_process()

    def update_unsharp_amount(self, value):
        self.unsharp_amount = float(value)
        self.unsharp_amount_label.configure(text=f"Amount: {value}")
        self.schedule_process()

    def update_unsharp_radius(self, value):
        self.unsharp_radius = int(round(float(value)))
        self.unsharp_radius_label.configure(text=f"Radio: {self.unsharp_radius}")
        self.schedule_process()

    def update_unsharp_threshold(self, value):
        self.unsharp_threshold = int(round(float(value)))
        self.unsharp_threshold_label.configure(text=f"Threshold: {self.unsharp_threshold}")
        self.schedule_process()

    def schedule_process(self):
        """Programa procesamiento con debouncing mejorado"""
        # Marcar para cancelar procesamiento anterior
        self.cancel_processing = True
        
        if hasattr(self, '_process_timer'):
            self.after_cancel(self._process_timer)
        
        # En modo bajo rendimiento, esperar más tiempo antes de procesar
        delay = 500 if self.low_performance_mode else 300
        self._process_timer = self.after(delay, self.process_image)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.dcm *.dicom")]
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".dcm", ".dicom"]:
            threading.Thread(target=self.load_dicom, args=(file_path,), daemon=True).start()
        else:
            self.load_standard_image(file_path)

    def load_standard_image(self, path):
        try:
            pil_img = Image.open(path).convert("RGB")
            
            # En modo bajo rendimiento, reducir tamaño si es muy grande
            if self.low_performance_mode:
                max_dimension = 2048
                if max(pil_img.size) > max_dimension:
                    ratio = max_dimension / max(pil_img.size)
                    new_size = tuple(int(dim * ratio) for dim in pil_img.size)
                    pil_img = pil_img.resize(new_size, Image.LANCZOS)
            
            self.original_image = pil_img
            self.cv_original = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            self.file_type = "Imagen"
            self.file_label.configure(text=self.file_type)
            self.dicom_metadata = {}
            self.update_metadata_display()
            self.display_original()
            self.reset_btn.configure(state="normal")
            self.process_image()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")

    def load_dicom(self, path):
        if not pydicom:
            self.after(0, lambda: messagebox.showerror("Error", "pydicom no está instalado."))
            return
        try:
            ds = pydicom.dcmread(path)

            # Extraer metadata con formato más compacto
            self.dicom_metadata = {
                "Paciente": str(getattr(ds, 'PatientName', 'N/D')),
                "Fecha": str(getattr(ds, 'StudyDate', 'N/D')),
                "Modalidad": str(getattr(ds, 'Modality', 'N/D')),
                "Dimensiones": f"{getattr(ds, 'Rows', '?')}x{getattr(ds, 'Columns', '?')}",
                "Window C/W": f"{getattr(ds, 'WindowCenter', 'N/D')}/{getattr(ds, 'WindowWidth', 'N/D')}",
                "Interp. Foto": getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2'),
                "Bits": getattr(ds, 'BitsAllocated', 8),
            }

            # Aplicar VOI LUT (windowing automático)
            pixel_array = apply_voi_lut(ds.pixel_array, ds, index=0)

            # Normalizar a 0-255
            if pixel_array.dtype != np.uint8:
                pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-6)) * 255.0
                pixel_array = pixel_array.astype(np.uint8)

            # Invertir si MONOCHROME1
            if self.dicom_metadata["Interp. Foto"] == 'MONOCHROME1':
                pixel_array = 255 - pixel_array

            # Convertir a RGB
            if len(pixel_array.shape) == 2:
                img_array = np.stack([pixel_array, pixel_array, pixel_array], axis=-1)
            else:
                img_array = pixel_array

            # En modo bajo rendimiento, reducir tamaño si es muy grande
            if self.low_performance_mode:
                max_dimension = 2048
                if max(img_array.shape[:2]) > max_dimension:
                    ratio = max_dimension / max(img_array.shape[:2])
                    new_size = tuple(int(dim * ratio) for dim in img_array.shape[:2][::-1])
                    img_array = cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)

            self.cv_original = img_array
            self.original_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

            # Actualizar UI en hilo principal
            self.after(0, self._update_ui_after_dicom_load)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error DICOM", f"No se pudo cargar:\n{str(e)}"))

    def _update_ui_after_dicom_load(self):
        self.file_type = "DICOM"
        self.file_label.configure(text=self.file_type)
        self.update_metadata_display()
        self.display_original()
        self.reset_btn.configure(state="normal")
        self.process_image()

    def update_metadata_display(self):
        """Actualiza el textbox de metadata"""
        self.metadata_text.configure(state="normal")
        self.metadata_text.delete("1.0", "end")
        
        if self.dicom_metadata:
            meta_lines = [f"{k}: {v}" for k, v in self.dicom_metadata.items()]
            self.metadata_text.insert("1.0", "\n".join(meta_lines))
        
        self.metadata_text.configure(state="disabled")

    def display_image(self, pil_img, canvas):
        canvas.delete("all")
        if not pil_img:
            canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2,
                               text="No hay imagen", fill="white", font=("Arial", 12))
            return

        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 400, 300

        zoom_w = int(pil_img.width * self.zoom_factor)
        zoom_h = int(pil_img.height * self.zoom_factor)

        resized = pil_img.resize((zoom_w, zoom_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        canvas.image = tk_img  # Evita garbage collection

        x = (canvas_w - zoom_w) // 2
        y = (canvas_h - zoom_h) // 2
        canvas.create_image(x, y, image=tk_img, anchor="nw")

    def display_original(self):
        if self.original_image:
            self.display_image(self.original_image, self.canvas_orig)

    def display_processed(self):
        if self.processed_image:
            self.display_image(self.processed_image, self.canvas_proc)

    def on_resize(self, event=None):
        self.display_original()
        self.display_processed()

    def reset_params(self):
        self.clahe_clip_limit = 2.0
        self.clahe_tile_size = 8
        self.guided_radius = 5
        self.guided_eps = 0.01
        self.unsharp_amount = 1.5
        self.unsharp_radius = 2
        self.unsharp_threshold = 0

        self.clahe_clip_slider.set(2.0)
        self.clahe_tile_slider.set(8)
        self.guided_radius_slider.set(5)
        self.guided_eps_slider.set(0.01)
        self.unsharp_amount_slider.set(1.5)
        self.unsharp_radius_slider.set(2)
        self.unsharp_threshold_slider.set(0)

        self.clahe_clip_label.configure(text="Clip Limit: 2.0")
        self.clahe_tile_label.configure(text="Tile Size: 8x8")
        self.guided_radius_label.configure(text="Radio: 5")
        self.guided_eps_label.configure(text="Epsilon: 0.010")
        self.unsharp_amount_label.configure(text="Amount: 1.5")
        self.unsharp_radius_label.configure(text="Radio: 2")
        self.unsharp_threshold_label.configure(text="Threshold: 0")

        self.process_image()

    def get_system_metrics(self):
        """Obtiene métricas del sistema y del proceso específico"""
        cpu = psutil.cpu_percent(interval=0.05 if self.low_performance_mode else 0.1)
        ram = psutil.virtual_memory().used / (1024**2)  # MB
        gpu_mem = 0
        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem = mem_info.used / (1024**2)
            except:
                gpu_mem = 0

        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        
        # Información del proceso actual
        current_process = psutil.Process()
        thread_count = current_process.num_threads()
        
        # Hilos activos del programa de procesamiento
        with self.thread_lock:
            active_processing_threads = self.active_threads

        return cpu, ram, gpu_mem, physical_cores, logical_cores, thread_count, active_processing_threads

    def update_static_metrics(self):
        """Actualización estática de métricas (modo bajo rendimiento)"""
        try:
            cpu, ram, gpu, phys, log, total_threads, proc_threads = self.get_system_metrics()
            
            metrics_text = (
                f"Modo: {self.cpu_info['mode']}\n"
                f"CPU: {cpu:.1f}%\n"
                f"Núcleos: {phys}F/{log}L\n"
                f"RAM: {ram:.1f} MB"
            )
            
            if GPU_AVAILABLE:
                metrics_text += f"\nGPU: {gpu:.1f} MB"
            
            self.metrics_label.configure(text=metrics_text)
        except:
            pass

    def start_metrics_monitor(self):
        """Inicia monitor de métricas en tiempo real"""
        def update_metrics():
            while True:
                try:
                    cpu, ram, gpu, phys, log, total_threads, proc_threads = self.get_system_metrics()
                    
                    metrics_text = (
                        f"Modo: {self.cpu_info['mode']}\n"
                        f"CPU: {cpu:.1f}%\n"
                        f"Núcleos: {phys}F/{log}L\n"
                        f"Hilos procesamiento: {proc_threads}\n"
                        f"Hilos totales: {total_threads}\n"
                        f"RAM: {ram:.1f} MB"
                    )
                    
                    if GPU_AVAILABLE:
                        metrics_text += f"\nGPU: {gpu:.1f} MB"
                    
                    self.after(0, lambda t=metrics_text: self.metrics_label.configure(text=t))
                    time.sleep(self.metrics_update_interval / 1000)
                except:
                    break
        
        monitor_thread = threading.Thread(target=update_metrics, daemon=True)
        monitor_thread.start()

    def process_image(self):
        if self.cv_original is None:
            return
        
        # Cancelar procesamiento anterior si existe
        self.cancel_processing = False
        
        # Evitar múltiples hilos de procesamiento
        if self.processing_thread and self.processing_thread.is_alive():
            return

        self.processing_thread = threading.Thread(target=self._process_image_thread, daemon=True)
        self.processing_thread.start()

    def _process_image_thread(self):
        # Incrementar contador de hilos activos
        with self.thread_lock:
            self.active_threads += 1
        
        start_time = time.time()

        try:
            # Verificar cancelación antes de comenzar
            if self.cancel_processing:
                return
            
            img = self.cv_original.copy()

            # 1. CLAHE
            if self.cancel_processing:
                return
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 2. Filtro guiado
            if self.cancel_processing:
                return
            try:
                img = cv2.ximgproc.guidedFilter(img, img, radius=self.guided_radius, eps=self.guided_eps*1000)
            except AttributeError:
                kernel_size = 2 * self.guided_radius + 1
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

            # 3. Unsharp Mask
            if self.cancel_processing:
                return
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=self.unsharp_radius)
            sharpened = cv2.addWeighted(img, 1 + self.unsharp_amount, blurred, -self.unsharp_amount, 0)
            diff = cv2.absdiff(img, sharpened)
            mask = diff > self.unsharp_threshold
            img = np.where(mask, sharpened, img)

            # Verificación final antes de convertir
            if self.cancel_processing:
                return

            rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            self.processed_image = Image.fromarray(rgb)

            end_time = time.time()
            process_time = end_time - start_time

            self.after(0, lambda: self._update_ui_after_processing(process_time))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Error al procesar:\n{str(e)}"))
        finally:
            # Decrementar contador de hilos activos
            with self.thread_lock:
                self.active_threads -= 1
            
            # Actualizar métricas si está en modo bajo rendimiento
            if self.low_performance_mode:
                self.after(0, self.update_static_metrics)

    def _update_ui_after_processing(self, process_time):
        self.display_processed()
        self.download_btn.configure(state="normal")

    def download_image(self):
        if not self.processed_image:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if file_path:
            self.processed_image.save(file_path)

    def zoom_in(self):
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor += 0.1
            self.zoom_factor = min(self.zoom_factor, self.max_zoom)
            self.display_original()
            self.display_processed()

    def zoom_out(self):
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor -= 0.1
            self.zoom_factor = max(self.zoom_factor, self.min_zoom)
            self.display_original()
            self.display_processed()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.display_original()
        self.display_processed()

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

if __name__ == "__main__":
    app = MedicalImageProcessor()
    app.mainloop()