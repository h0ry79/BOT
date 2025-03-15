"""Módulo para la gestión de take profits.

Este módulo maneja la configuración y ejecución de los niveles de take profit
para las posiciones de trading.
"""

from binance.exceptions import BinanceAPIException
import time
import asyncio

from config.config import CONFIG
from utils.helpers import logger

class TakeProfit:
    def __init__(self):
        """Inicializa el gestor de take profits."""
        self.take_profit_levels = []
        self.current_tp_index = 0
        self.initial_quantity = 0.0

    def configure_take_profit(self, entry_price, atr_long, position_side):
        """Configura los niveles de take profit basados en el ATR."""
        if atr_long <= 0 or entry_price <= 0:
            logger.error("Error: ATR o precio de entrada inválidos")
            return False

        if position_side not in ['LONG', 'SHORT']:
            logger.error(f"Error: Posición inválida {position_side}")
            return False

        try:
            self.take_profit_levels = [
                entry_price + (multiplier * atr_long) if position_side == 'LONG'
                else entry_price - (multiplier * atr_long)
                for multiplier in CONFIG['TP_MULTIPLIERS']
            ]
            logger.info(f"TPs configurados: {[f'{tp:.4f}' for tp in self.take_profit_levels]}")
            return True
        except Exception as e:
            logger.error(f"Error al configurar take profits: {e}")
            return False

    def get_next_tp_target(self):
        """Obtiene el siguiente nivel de take profit."""
        if not self.take_profit_levels or self.current_tp_index >= len(self.take_profit_levels):
            return None
        return self.take_profit_levels[self.current_tp_index]

    def get_tp_quantity(self):
        """Calcula la cantidad para el siguiente take profit."""
        if not self.take_profit_levels or self.current_tp_index >= len(CONFIG['TAKE_PROFIT_LEVELS']):
            return 0.0
        
        tp_quantity = self.initial_quantity * CONFIG['TAKE_PROFIT_LEVELS'][self.current_tp_index]
        return max(tp_quantity, CONFIG.get('MIN_QUANTITY', 0.001))

    async def execute_take_profit(self, client, position_side, current_price):
        """Ejecuta el take profit si se alcanza el nivel objetivo."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return False

        if position_side not in ['LONG', 'SHORT']:
            logger.error(f"Error: Posición inválida {position_side}")
            return False

        if current_price <= 0 or not isinstance(current_price, (int, float)):
            logger.error("Error: Precio actual inválido o tipo de dato incorrecto")
            return False

        tp_target = self.get_next_tp_target()
        if not tp_target:
            return False

        if ((position_side == 'LONG' and current_price >= tp_target) or
            (position_side == 'SHORT' and current_price <= tp_target)):
            
            tp_quantity = self.get_tp_quantity()
            if tp_quantity <= CONFIG.get('MIN_QUANTITY', 0.001):
                logger.error(f"Error: Cantidad de TP {tp_quantity} menor que el mínimo permitido")
                return False

            try:
                order = await client.futures_create_order(
                    symbol=CONFIG['SYMBOL'],
                    side='SELL' if position_side == 'LONG' else 'BUY',
                    type="MARKET",
                    quantity=tp_quantity,
                    timestamp=int(time.time() * 1000),
                    recvWindow=CONFIG['RECV_WINDOW']
                )

                if not order or 'orderId' not in order:
                    logger.error("Error: La orden de TP no se creó correctamente")
                    return False

                logger.info(f"Take Profit ejecutado: {tp_target:.4f} | Cantidad: {tp_quantity:.4f} | Orden ID: {order['orderId']}")
                self.current_tp_index += 1
                return True
            except BinanceAPIException as e:
                if 'Account has insufficient balance' in str(e):
                    logger.error("Error: Balance insuficiente para ejecutar take profit")
                elif 'Invalid API-key' in str(e):
                    logger.error("Error: Verifique sus credenciales de API")
                elif 'IP has been auto-banned' in str(e):
                    logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                    await asyncio.sleep(60)
                else:
                    logger.error(f"Error al ejecutar take profit: {e}")
                return False
            except Exception as e:
                logger.error(f"Error inesperado al ejecutar take profit: {e}")
                return False

    def reset(self):
        """Reinicia los valores del gestor de take profits."""
        self.take_profit_levels = []
        self.current_tp_index = 0
        self.initial_quantity = 0.0
        logger.info("Take Profit reseteado")