import psutil
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.animation import FuncAnimation
import numpy as np

# Intentar importar GPUtil para GPU
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil no disponible. Instala con: pip install gputil")

class SystemMonitor:
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#f8f9fa')
        
        # Crear subplots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Círculos superiores
        self.ax_cpu = self.fig.add_subplot(gs[0, 0])
        self.ax_gpu = self.fig.add_subplot(gs[0, 1])
        self.ax_ram = self.fig.add_subplot(gs[0, 2])
        
        # Barras inferiores
        self.ax_bars = self.fig.add_subplot(gs[1, :])
        
        self.setup_circular_plots()
        self.setup_bar_plot()
        
    def setup_circular_plots(self):
        """Configura los gráficos circulares"""
        for ax in [self.ax_cpu, self.ax_gpu, self.ax_ram]:
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.axis('off')
    
    def setup_bar_plot(self):
        """Configura el gráfico de barras"""
        self.ax_bars.set_xlim(0, 100)
        self.ax_bars.set_ylim(-0.5, 2.5)
        self.ax_bars.set_xlabel('Uso (%)', fontsize=12, fontweight='bold')
        self.ax_bars.grid(axis='x', alpha=0.3, linestyle='--')
        self.ax_bars.set_facecolor('#ffffff')
    
    def draw_circular_gauge(self, ax, value, label, color):
        """Dibuja un medidor circular"""
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Círculo de fondo (gris)
        bg_circle = plt.Circle((0, 0), 1, color='#e5e7eb', fill=False, linewidth=12)
        ax.add_patch(bg_circle)
        
        # Arco de progreso
        theta = np.linspace(0, 2 * np.pi * (value / 100), 100)
        x = np.cos(theta - np.pi/2)
        y = np.sin(theta - np.pi/2)
        ax.plot(x, y, color=color, linewidth=12, solid_capstyle='round')
        
        # Texto central
        ax.text(0, 0.1, f'{value:.1f}%', 
                ha='center', va='center', 
                fontsize=32, fontweight='bold', color='#1f2937')
        
        # Etiqueta
        ax.text(0, -1.3, label, 
                ha='center', va='center', 
                fontsize=16, fontweight='bold', color='#4b5563')
    
    def get_gpu_usage(self):
        """Obtiene el uso de GPU"""
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].load * 100
            except:
                pass
        return 0
    
    def update(self, frame):
        """Actualiza los gráficos"""
        # Obtener datos del sistema
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        gpu = self.get_gpu_usage()
        
        # Actualizar círculos
        self.draw_circular_gauge(self.ax_cpu, cpu, 'CPU', '#ef4444')
        self.draw_circular_gauge(self.ax_gpu, gpu, 'GPU', '#3b82f6')
        self.draw_circular_gauge(self.ax_ram, ram, 'RAM', '#10b981')
        
        # Actualizar barras
        self.ax_bars.clear()
        self.ax_bars.set_xlim(0, 100)
        self.ax_bars.set_ylim(-0.5, 2.5)
        self.ax_bars.set_xlabel('Uso (%)', fontsize=12, fontweight='bold')
        self.ax_bars.grid(axis='x', alpha=0.3, linestyle='--')
        self.ax_bars.set_facecolor('#ffffff')
        
        labels = ['RAM', 'GPU', 'CPU']
        values = [ram, gpu, cpu]
        colors = ['#10b981', '#3b82f6', '#ef4444']
        positions = [0, 1, 2]
        
        bars = self.ax_bars.barh(positions, values, color=colors, height=0.6, alpha=0.8)
        
        # Añadir etiquetas a las barras
        for i, (bar, val) in enumerate(zip(bars, values)):
            self.ax_bars.text(val + 2, i, f'{val:.1f}%', 
                            va='center', fontweight='bold', fontsize=11)
            self.ax_bars.text(-2, i, labels[i], 
                            va='center', ha='right', fontweight='bold', fontsize=11)
        
        self.ax_bars.set_yticks([])
        self.ax_bars.spines['left'].set_visible(False)
        self.ax_bars.spines['top'].set_visible(False)
        self.ax_bars.spines['right'].set_visible(False)
        
        return []
    
    def start(self):
        """Inicia la animación"""
        ani = FuncAnimation(self.fig, self.update, interval=1000, blit=True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    print("Iniciando monitor de sistema...")
    print("Presiona Ctrl+C para detener")
    
    monitor = SystemMonitor()
    monitor.start()