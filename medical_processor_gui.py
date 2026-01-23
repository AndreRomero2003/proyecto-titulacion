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

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MedicalImageProcessor4Core(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Procesador de Imágenes Médicas - Optimizado para 4 Núcleos")
        self.geometry("1400x900")
        
        # Variables principales
        self.original_image = None
        self.processed_image = None
        self.cv_original = None
        self.file_type = ""
        self.dicom_metadata = {}
        self.zoom_factor = 1.0
        self.max_zoom = 4.0
        self.min_zoom = 0.5
        
        # Grids optimizados para sistemas de 4 núcleos
        self.clip_grid = [1, 2, 3, 4, 6, 8, 12, 16]  # CLAHE clipLimit
        self.tile_grid = [(4,4), (6,6), (8,8), (12,12), (16,16), (24,24), (32,32)]  # CLAHE tileGridSize
        self.radius_grid = [1, 3, 4, 6, 8, 12, 16, 24, 32]  # Filtro Guiado radius
        self.eps_grid = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]  # Filtro Guiado epsilon
        self.amount_grid = [0.0, 0.2, 0.3, 0.6, 1.0, 1.6, 2.0, 2.2]  # Unsharp Mask amount
        self.sigma_grid = [0.5, 0.8, 1.2, 1.8, 2.5, 3.0, 3.5]  # Unsharp Mask sigma
        self.threshold_grid = [0.1, 0.005, 0.01, 0.02, 0.05, 0.1]  # Unsharp Mask threshold
        
        # Parámetros iniciales
        self.clahe_clip_limit = self.clip_grid[1]  # 2
        self.clahe_tile_size = self.tile_grid[2][0]  # 8x8
        self.guided_radius = self.radius_grid[1]  # 3 (más cercano a 5)
        self.guided_eps = self.eps_grid[-1]  # 1e-2
        self.unsharp_amount = self.amount_grid[3]  # 0.6 (más cercano a 1.5)
        self.unsharp_radius = self.sigma_grid[1]  # 0.8 (más cercano a 2.0)
        self.unsharp_threshold = self.threshold_grid[0]  # 0.0
        
        # Variables para el modo de procesamiento
        self.processing = False        
        self.create_widgets()
        self.start_resource_monitoring()
    
    def create_widgets(self):
        # Panel izquierdo: controles con scroll
        self.control_frame = ctk.CTkFrame(self, width=320)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        # Scrollable frame dentro del control_frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self.control_frame, width=300, height=700)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        title = ctk.CTkLabel(self.scrollable_frame, text="Procesador de Imágenes Médicas", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        self.upload_btn = ctk.CTkButton(self.scrollable_frame, text="Cargar imagen", command=self.load_image)
        self.upload_btn.pack(pady=5)
        
        self.file_label = ctk.CTkLabel(self.scrollable_frame, text="", font=("Arial", 10))
        self.file_label.pack()
        
        self.metadata_label = ctk.CTkLabel(self.scrollable_frame, text="", justify="left", font=("Arial", 9))
        self.metadata_label.pack(pady=5)        
        self.reset_btn = ctk.CTkButton(self.scrollable_frame, text="Resetear Parámetros", command=self.reset_params)
        self.reset_btn.pack(pady=10)
        self.reset_btn.configure(state="disabled")
        
        # Label de métricas con tamaño fijo
        self.metrics_label = ctk.CTkLabel(
            self.scrollable_frame,
            text="CPU: --%\nNúcleos: --\nHilos: --\nRAM: -- MB\nGPU: -- MB",
            justify="left",
            font=("Arial", 10),
            width=200,
            height=120,
            wraplength=190
        )
        self.metrics_label.pack(pady=10, padx=10)
        
        self.create_sliders()
        
        self.download_btn = ctk.CTkButton(self.scrollable_frame, text="Descargar Imagen", command=self.download_image)
        self.download_btn.pack(pady=10)
        self.download_btn.configure(state="disabled")
        
        # Botones de zoom
        self.zoom_frame = ctk.CTkFrame(self.scrollable_frame)
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

    def create_sliders(self):
        # CLAHE - clipLimit
        ctk.CTkLabel(self.scrollable_frame, text="CLARIDAD/CONTRASTE", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.clahe_clip_slider = ctk.CTkSlider(
            self.scrollable_frame,
            from_=0, to=len(self.clip_grid)-1,
            number_of_steps=len(self.clip_grid)-1,
            command=self.update_clahe_clip
        )
        self.clahe_clip_slider.set(1)  # índice 1 → valor 2
        self.clahe_clip_slider.pack(fill="x", padx=10)
        self.clahe_clip_label = ctk.CTkLabel(self.scrollable_frame, text=f"Clip Limit: {self.clip_grid[1]}")
        self.clahe_clip_label.pack(anchor="w", padx=20)
        
        # CLAHE - tileGridSize
        self.clahe_tile_slider = ctk.CTkSlider(
            self.scrollable_frame,
            from_=0, to=len(self.tile_grid)-1,
            number_of_steps=len(self.tile_grid)-1,
            command=self.update_clahe_tile
        )
        self.clahe_tile_slider.set(2)  # índice 2 → (8,8)
        self.clahe_tile_slider.pack(fill="x", padx=10)
        self.clahe_tile_label = ctk.CTkLabel(self.scrollable_frame, text=f"Tile Size: {self.tile_grid[2][0]}x{self.tile_grid[2][1]}")
        self.clahe_tile_label.pack(anchor="w", padx=20)
        
        # Filtro Guiado - radius
        ctk.CTkLabel(self.scrollable_frame, text="SUAVIZADO", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.guided_radius_slider = ctk.CTkSlider(
            self.scrollable_frame,
            from_=0, to=len(self.radius_grid)-1,
            number_of_steps=len(self.radius_grid)-1,
            command=self.update_guided_radius
        )
        self.guided_radius_slider.set(1)  # índice 1 → valor 3 (cercano a 5)
        self.guided_radius_slider.pack(fill="x", padx=10)
        self.guided_radius_label = ctk.CTkLabel(self.scrollable_frame, text=f"Radio: {self.radius_grid[1]}")
        self.guided_radius_label.pack(anchor="w", padx=20)
        
        # Filtro Guiado - epsilon
        self.guided_eps_slider = ctk.CTkSlider(
            self.scrollable_frame,
            from_=0, to=len(self.eps_grid)-1,
            number_of_steps=len(self.eps_grid)-1,
            command=self.update_guided_eps
        )
        self.guided_eps_slider.set(len(self.eps_grid)-1)  # último → 1e-2
        self.guided_eps_slider.pack(fill="x", padx=10)
        self.guided_eps_label = ctk.CTkLabel(self.scrollable_frame, text=f"Epsilon: {self.eps_grid[-1]:.1e}")
        self.guided_eps_label.pack(anchor="w", padx=20)
        
        # Unsharp Mask - amount
        ctk.CTkLabel(self.scrollable_frame, text="NITIDEZ/DETALLE", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.unsharp_amount_slider = ctk.CTkSlider(
            self.scrollable_frame,
            from_=0, to=len(self.amount_grid)-1,
            number_of_steps=len(self.amount_grid)-1,
            command=self.update_unsharp_amount
        )
        self.unsharp_amount_slider.set(3)  # índice 3 → 0.6 (cercano a 1.5)
        self.unsharp_amount_slider.pack(fill="x", padx=10)
        self.unsharp_amount_label = ctk.CTkLabel(self.scrollable_frame, text=f"Amount: {self.amount_grid[3]}")
        self.unsharp_amount_label.pack(anchor="w", padx=20)
        
        # Unsharp Mask - sigma
        self.unsharp_radius_slider = ctk.CTkSlider(
            self.scrollable_frame,
            from_=0, to=len(self.sigma_grid)-1,
            number_of_steps=len(self.sigma_grid)-1,
            command=self.update_unsharp_radius
        )
        self.unsharp_radius_slider.set(1)  # índice 1 → 0.8 (cercano a 2.0)
        self.unsharp_radius_slider.pack(fill="x", padx=10)
        self.unsharp_radius_label = ctk.CTkLabel(self.scrollable_frame, text=f"Radio: {self.sigma_grid[1]}")
        self.unsharp_radius_label.pack(anchor="w", padx=20)
        
        # Unsharp Mask - threshold
        self.unsharp_threshold_slider = ctk.CTkSlider(
            self.scrollable_frame,
            from_=0, to=len(self.threshold_grid)-1,
            number_of_steps=len(self.threshold_grid)-1,
            command=self.update_unsharp_threshold
        )
        self.unsharp_threshold_slider.set(0)  # índice 0 → 0.0
        self.unsharp_threshold_slider.pack(fill="x", padx=10)
        self.unsharp_threshold_label = ctk.CTkLabel(self.scrollable_frame, text=f"Threshold: {self.threshold_grid[0]}")
        self.unsharp_threshold_label.pack(anchor="w", padx=20)
    
    def update_clahe_clip(self, value):
        index = int(round(float(value)))
        self.clahe_clip_limit = self.clip_grid[index]
        self.clahe_clip_label.configure(text=f"Clip Limit: {self.clahe_clip_limit}")
        self.schedule_process()
    
    def update_clahe_tile(self, value):
        index = int(round(float(value)))
        size = self.tile_grid[index][0]  # Tomamos solo el ancho (asumimos cuadrado)
        self.clahe_tile_size = size
        self.clahe_tile_label.configure(text=f"Tile Size: {size}x{size}")
        self.schedule_process()
    
    def update_guided_radius(self, value):
        index = int(round(float(value)))
        self.guided_radius = self.radius_grid[index]
        self.guided_radius_label.configure(text=f"Radio: {self.guided_radius}")
        self.schedule_process()
    
    def update_guided_eps(self, value):
        index = int(round(float(value)))
        self.guided_eps = self.eps_grid[index]
        self.guided_eps_label.configure(text=f"Epsilon: {self.guided_eps:.1e}")
        self.schedule_process()
    
    def update_unsharp_amount(self, value):
        index = int(round(float(value)))
        self.unsharp_amount = self.amount_grid[index]
        self.unsharp_amount_label.configure(text=f"Amount: {self.unsharp_amount}")
        self.schedule_process()
    
    def update_unsharp_radius(self, value):
        index = int(round(float(value)))
        self.unsharp_radius = self.sigma_grid[index]
        self.unsharp_radius_label.configure(text=f"Radio: {self.unsharp_radius}")
        self.schedule_process()
    
    def update_unsharp_threshold(self, value):
        index = int(round(float(value)))
        self.unsharp_threshold = self.threshold_grid[index]
        self.unsharp_threshold_label.configure(text=f"Threshold: {self.unsharp_threshold}")
        self.schedule_process()
    
    def schedule_process(self):
        """Programar procesamiento con debounce"""
        if hasattr(self, '_process_timer'):
            self.after_cancel(self._process_timer)
        self._process_timer = self.after(200, self.process_image)
    
    def load_image(self):
        """Cargar imagen con manejo de memoria optimizado para sistemas de 4 núcleos"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.dcm *.dicom")]
        )
        if not file_path:
            return
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".dcm", ".dicom"]:
            threading.Thread(target=self.load_dicom, args=(file_path,), daemon=True).start()
        else:
            threading.Thread(target=self.load_standard_image, args=(file_path,), daemon=True).start()
    
    def load_standard_image(self, path):
        """Cargar imágenes estándar con optimización para sistemas de 4 núcleos"""
        try:
            pil_img = Image.open(path).convert("RGB")
            self.original_image = pil_img
            self.cv_original = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            self.file_type = "Imagen"
            self.file_label.configure(text=self.file_type)
            self.dicom_metadata = {}
            self.metadata_label.configure(text="")
            self.display_original()
            self.reset_btn.configure(state="normal")
            self.process_image()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")
    
    def load_dicom(self, path):
        """Cargar DICOM con manejo de errores robusto para sistemas de 4 núcleos"""
        if not pydicom:
            self.after(0, lambda: messagebox.showerror("Error", "pydicom no está instalado."))
            return
        
        try:
            ds = pydicom.dcmread(path, force=True)
            self.dicom_metadata = {
                "patientName": str(getattr(ds, 'PatientName', 'No disponible')),
                "studyDate": str(getattr(ds, 'StudyDate', 'No disponible')),
                "modality": str(getattr(ds, 'Modality', 'No disponible')),
                "rows": getattr(ds, 'Rows', '?'),
                "cols": getattr(ds, 'Columns', '?'),
                "windowCenter": getattr(ds, 'WindowCenter', None),
                "windowWidth": getattr(ds, 'WindowWidth', None),
                "photometricInterpretation": getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2'),
                "bitsAllocated": getattr(ds, 'BitsAllocated', 8),
                "transferSyntax": getattr(ds, 'TransferSyntaxUID', 'No disponible')
            }
            
            # Obtener pixel data
            try:
                pixel_array = ds.pixel_array
            except Exception as e:
                messagebox.showerror("Error DICOM", f"No se pudo decodificar la imagen:\n{str(e)}")
                return
            
            # Asegurar tipo float
            pixel_array = pixel_array.astype(np.float32)
            
            # Aplicar VOI LUT si es posible
            wc = self.dicom_metadata["windowCenter"]
            ww = self.dicom_metadata["windowWidth"]
            if wc is not None and ww is not None:
                min_val = wc - ww / 2
                max_val = wc + ww / 2
            else:
                min_val = np.min(pixel_array)
                max_val = np.max(pixel_array)
            
            # Clip y normalizar
            pixel_array = np.clip(pixel_array, min_val, max_val)
            pixel_array = ((pixel_array - min_val) / (max_val - min_val + 1e-6)) * 255.0
            pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
            
            # Invertir si es MONOCHROME1
            if self.dicom_metadata["photometricInterpretation"] == 'MONOCHROME1':
                pixel_array = 255 - pixel_array
            
            # Convertir a RGB
            if len(pixel_array.shape) == 2:
                img_array = np.stack([pixel_array, pixel_array, pixel_array], axis=-1)
            else:
                img_array = pixel_array
            
            self.cv_original = img_array
            self.original_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            self.after(0, self._update_ui_after_dicom_load)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error DICOM", f"No se pudo cargar:\n{str(e)}"))
    
    def _update_ui_after_dicom_load(self):
        self.file_type = "DICOM"
        self.file_label.configure(text=self.file_type)
        meta_text = "\n".join([f"{k}: {v}" for k, v in self.dicom_metadata.items()])
        self.metadata_label.configure(text=meta_text)
        self.display_original()
        self.reset_btn.configure(state="normal")
        self.process_image()
    
    def display_image(self, pil_img, canvas):
        """Mostrar imagen con optimización para sistemas de 4 núcleos"""
        canvas.delete("all")
        
        if not pil_img:
            canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2,
                             text="No hay imagen", fill="white", font=("Arial", 12))
            return
        
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 400, 300
        
        # Calcular nuevo tamaño con zoom
        zoom_w = int(pil_img.width * self.zoom_factor)
        zoom_h = int(pil_img.height * self.zoom_factor)
        
        # Limitar tamaño máximo para evitar problemas de memoria 
        max_size = 5000
        if zoom_w > max_size or zoom_h > max_size:
            ratio = max_size / max(zoom_w, zoom_h)
            zoom_w = int(zoom_w * ratio)
            zoom_h = int(zoom_h * ratio)
        
        # Redimensionar con calidad adecuada
        try:
            # Usar LANCZOS para mejor calidad (sistema de 4 núcleos puede manejarlo)
            resized = pil_img.resize((zoom_w, zoom_h), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized)
            
            # Guardar referencia para evitar que se elimine
            canvas.image = tk_img
            
            x = (canvas_w - zoom_w) // 2
            y = (canvas_h - zoom_h) // 2
            canvas.create_image(x, y, image=tk_img, anchor="nw")
        except MemoryError:
            # Reducir zoom automáticamente si hay error de memoria
            self.zoom_factor = max(0.5, self.zoom_factor * 0.7)
            messagebox.showwarning("Memoria limitada", 
                "No hay suficiente memoria para mostrar la imagen a este zoom.\n"
                "Se reduce el nivel de zoom automáticamente.")
            self.display_image(pil_img, canvas)
        except Exception as e:
            messagebox.showerror("Error de Visualización", f"No se pudo mostrar la imagen:\n{str(e)}")
    
    def display_original(self):
        """Mostrar imagen original"""
        if self.original_image:
            self.display_image(self.original_image, self.canvas_orig)
    
    def display_processed(self):
        """Mostrar imagen procesada"""
        if self.processed_image:
            self.display_image(self.processed_image, self.canvas_proc)
    
    def on_resize(self, event=None):
        """Manejar redimensionamiento con debounce"""
        if hasattr(self, '_resize_timer'):
            self.after_cancel(self._resize_timer)
        self._resize_timer = self.after(100, self._perform_resize)
    
    def _perform_resize(self):
        self.display_original()
        self.display_processed()
    
    def reset_params(self):
        """Resetear parámetros a valores originales"""
        self.clahe_clip_limit = self.clip_grid[1]  # 2
        self.clahe_tile_size = self.tile_grid[2][0]  # 8x8
        self.guided_radius = self.radius_grid[1]  # 3
        self.guided_eps = self.eps_grid[-1]  # 1e-2
        self.unsharp_amount = self.amount_grid[3]  # 0.6
        self.unsharp_radius = self.sigma_grid[1]  # 0.8
        self.unsharp_threshold = self.threshold_grid[0]  # 0.0
        
        # Actualizar sliders
        self.clahe_clip_slider.set(1)
        self.clahe_tile_slider.set(2)
        self.guided_radius_slider.set(1)
        self.guided_eps_slider.set(len(self.eps_grid)-1)
        self.unsharp_amount_slider.set(3)
        self.unsharp_radius_slider.set(1)
        self.unsharp_threshold_slider.set(0)
        
        # Actualizar labels
        self.clahe_clip_label.configure(text=f"Clip Limit: {self.clip_grid[1]}")
        self.clahe_tile_label.configure(text=f"Tile Size: {self.tile_grid[2][0]}x{self.tile_grid[2][1]}")
        self.guided_radius_label.configure(text=f"Radio: {self.radius_grid[1]}")
        self.guided_eps_label.configure(text=f"Epsilon: {self.eps_grid[-1]:.1e}")
        self.unsharp_amount_label.configure(text=f"Amount: {self.amount_grid[3]}")
        self.unsharp_radius_label.configure(text=f"Radio: {self.sigma_grid[1]}")
        self.unsharp_threshold_label.configure(text=f"Threshold: {self.threshold_grid[0]}")
        
        self.process_image()
    
    def get_system_metrics(self):
        """Obtener métricas de sistema optimizadas para 4 núcleos"""
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().used / (1024**2)
        gpu_mem = 0
        
        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem = mem_info.used / (1024**2)
            except:
                gpu_mem = 0
        
        physical_cores = psutil.cpu_count(logical=False) or 4  
        logical_cores = psutil.cpu_count(logical=True) or 4
        
        try:
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
        except:
            thread_count = 0
        
        return cpu, ram, gpu_mem, physical_cores, logical_cores, thread_count
    
    def start_resource_monitoring(self):
        """Monitorear recursos en segundo plano optimizado para 4 núcleos"""
        def monitor():
            while True:
                try:
                    cpu = psutil.cpu_percent(interval=1)
                    ram_percent = psutil.virtual_memory().percent
                    
                    # Ajustar umbrales 
                    if cpu > 95 or ram_percent > 95:
                        self.after(0, lambda: messagebox.showwarning("Recursos Críticos", 
                            f"Uso alto de recursos:\nCPU: {cpu}%\nRAM: {ram_percent}%\n"
                            "Considere reducir la resolución de la imagen."))
                    
                    time.sleep(3)  
                except:
                    break
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def process_image(self):
        """Procesar imagen optimizada para sistemas de 4 núcleos"""
        if self.cv_original is None or self.processing:
            return
        
        self.processing = True
        self.metrics_label.configure(text="Procesando...\nPor favor espere")
        self.update_idletasks()  # Forzar actualización de UI
        
        threading.Thread(target=self._process_image_thread, daemon=True).start()
    
    def _process_image_thread(self):
        """Hilo de procesamiento optimizado para 4 núcleos"""
        cpu_before, ram_before, gpu_before, _, _, _ = self.get_system_metrics()
        start_time = time.time()
        
        try:
            if self.cv_original is None:
                return
            
            img = self.cv_original.copy()
            
            # 1. CLAHE - Ajustar para evitar borrosidad en valores mínimos
            if self.clahe_clip_limit > 1 or self.clahe_tile_size > 4:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 2. Filtro guiado - Ajustar para evitar borrosidad en valores mínimos
            if self.guided_radius > 2 or self.guided_eps < 1e-4:
                try:
                    img = cv2.ximgproc.guidedFilter(img, img, radius=self.guided_radius, eps=self.guided_eps*1000)
                except AttributeError:
                    kernel_size = 2 * self.guided_radius + 1
                    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            # 3. Unsharp Mask - Ajustar para evitar borrosidad en valores mínimos
            if self.unsharp_amount > 0.1 or self.unsharp_radius > 0.5:
                blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=self.unsharp_radius)
                sharpened = cv2.addWeighted(img, 1 + self.unsharp_amount, blurred, -self.unsharp_amount, 0)
                diff = cv2.absdiff(img, sharpened)
                mask = diff > self.unsharp_threshold
                img = np.where(mask, sharpened, img)
            
            # 4. Convertir a RGB
            rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            self.processed_image = Image.fromarray(rgb)
            
            cpu_after, ram_after, gpu_after, phys_cores, log_cores, threads = self.get_system_metrics()
            end_time = time.time()
            
            self.after(0, self._update_ui_after_processing, 
                      cpu_before, cpu_after, ram_before, ram_after, gpu_before, gpu_after,
                      phys_cores, log_cores, threads, end_time - start_time)
            
        except MemoryError:
            self.after(0, lambda: messagebox.showerror("Error de Memoria", 
                "Memoria insuficiente para procesar la imagen.\n"
                "Intente reducir la resolución de la imagen."))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error de Procesamiento", 
                f"Error al procesar la imagen:\n{str(e)}"))
        finally:
            self.processing = False
    
    def _update_ui_after_processing(self, cpu_before, cpu_after, ram_before, ram_after, 
                                   gpu_before, gpu_after, phys_cores, log_cores, threads, processing_time):
        """Actualizar UI después de procesamiento"""
        cpu_avg = (cpu_before + cpu_after) / 2
        ram_used = ram_after - ram_before
        gpu_used = gpu_after - gpu_before
        
        self.metrics_label.configure(
            text=f"Tiempo: {processing_time:.2f}s\n"
            f"CPU: {cpu_avg:.1f}%\n"
            f"Núcleos: {phys_cores}F/{log_cores}L\n"
            f"Hilos (proceso): {threads}\n"
            f"RAM usada: {ram_used:.1f} MB\n"
            f"GPU usada: {gpu_used:.1f} MB"
        )
        
        self.display_processed()
        self.download_btn.configure(state="normal")
    
    def download_image(self):
        """Descargar imagen procesada"""
        if not self.processed_image:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        
        if file_path:
            try:
                self.processed_image.save(file_path, optimize=True)
                messagebox.showinfo("Éxito", f"Imagen guardada exitosamente en:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la imagen:\n{str(e)}")
    
    def zoom_in(self):
        """Aumentar zoom"""
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor += 0.1
            self.zoom_factor = min(self.zoom_factor, self.max_zoom)
            self.display_original()
            self.display_processed()
    
    def zoom_out(self):
        """Reducir zoom"""
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor -= 0.1
            self.zoom_factor = max(self.zoom_factor, self.min_zoom)
            self.display_original()
            self.display_processed()
    
    def reset_zoom(self):
        """Resetear zoom"""
        self.zoom_factor = 1.0
        self.display_original()
        self.display_processed()
    
    def on_mouse_wheel(self, event):
        """Manejar rueda del mouse para zoom"""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

if __name__ == "__main__":
    # Verificar requisitos mínimos
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy', 
        'PIL': 'Pillow',
        'psutil': 'psutil'
    }
    
    missing_packages = []
    for import_name, install_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(install_name)
    
    if missing_packages:
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("Paquetes faltantes", 
            f"Los siguientes paquetes son necesarios:\n{', '.join(missing_packages)}\n\n"
            f"Instale con: pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    app = MedicalImageProcessor4Core()
    app.mainloop()


