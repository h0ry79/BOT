"""Módulo de interfaz de usuario para el bot de trading.

Este módulo maneja la visualización en tiempo real del estado del bot,
posiciones, indicadores y notificaciones en la terminal.
"""

from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich import box
import time
import os
import logging

logger = logging.getLogger(__name__)

class TerminalUI:
    def __init__(self):
        """Inicializa la interfaz de terminal."""
        self.running = True
        try:
            self.clear_screen()
        except Exception as e:
            logger.error(f"Error al inicializar la interfaz de terminal: {e}")
            # Fallback if clear screen fails
            print("\n" * 100)

    def clear_screen(self):
        """Limpia la pantalla de la terminal."""
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except Exception as e:
            logger.error(f"Error al limpiar la pantalla: {e}")
            # Fallback to printing newlines
            print("\n" * 100)

    def stop(self):
        """Detiene la interfaz."""
        self.running = False

    def start(self):
        """Inicia la interfaz."""
        self.running = True

    def format_number(self, number, decimals=4):
        """Formatea números con separadores de miles y decimales específicos."""
        if number is None:
            return "N/A"
        return f"{number:,.{decimals}f}"

    def create_separator(self, width=80):
        """Crea una línea separadora."""
        return "─" * width

    def update(self, bot, balance, websocket_status, notifications, symbol_info, config):
        """Actualiza la interfaz con la información más reciente."""
        if not self.running:
            return

        self.clear_screen()
        
        # Encabezado
        print(f"\n{self.create_separator()}")
        print(f"│ {'GESTOR TRAE - Panel de Control':^78} │")
        print(self.create_separator())

        # Estado de la conexión y balance
        print(f"\n{'Estado':15} │ {websocket_status:^20} {'Balance USDT':>20} │ {self.format_number(balance, 2):>15}")
        print(self.create_separator())

        # Información de la posición
        print(f"\n{'INFORMACIÓN DE POSICIÓN':^80}")
        print(self.create_separator())
        
        if bot.in_position:
            position_info = [
                ("Lado", f"{bot.position_side:^15}"),
                ("Cantidad", f"{self.format_number(bot.trade_quantity):>15}"),
                ("Precio Entrada", f"{self.format_number(bot.entry_price):>15}"),
                ("Precio Actual", f"{self.format_number(bot.current_price):>15}"),
                ("Beneficio", f"{self.format_number(bot.floating_profit, 2):>15}")
            ]
            
            for label, value in position_info:
                print(f"{label:15} │ {value}")
        else:
            print(f"{'Sin posición activa':^80}")

        # Indicadores técnicos
        print(f"\n{'INDICADORES TÉCNICOS':^80}")
        print(self.create_separator())
        if hasattr(bot, 'long_atr') and hasattr(bot, 'short_atr'):
            print(f"{'ATR Largo':15} │ {self.format_number(bot.long_atr):>15}")
            print(f"{'ATR Corto':15} │ {self.format_number(bot.short_atr):>15}")
        else:
            print(f"{'ATR Largo':15} │ {'N/A':>15}")
            print(f"{'ATR Corto':15} │ {'N/A':>15}")

        # Últimas notificaciones
        print(f"\n{'ÚLTIMAS NOTIFICACIONES':^80}")
        print(self.create_separator())
        if notifications:
            for notification in notifications[-5:]:
                print(f"│ {notification}")
        else:
            print(f"{'Sin notificaciones':^80}")

        print(f"\n{self.create_separator()}")
        print(f"Presione Ctrl+C para detener el bot")