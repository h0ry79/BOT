# Módulo para la gestión del trailing stop loss dinámico.
#
# Este módulo maneja la configuración y gestión del trailing stop loss dinámico para las posiciones de trading.

from binance.exceptions import BinanceAPIException
from tenacity import retry, wait_exponential, stop_after_attempt
import time
import asyncio

from config.config import CONFIG
from utils.helpers import get_safe_stop_price, logger

class TrailingStopLoss:
    def __init__(self):
        """Inicializa el gestor de trailing stop loss."""
        self.trailing_stop = 0.0
        self.stop_loss_order_id = None
        self.trailing_update_count = 0

    async def validate_order_params(self, side, price, position_side, current_price):
        """Valida los parámetros de la orden de trailing stop."""
        if price <= 0 or not isinstance(price, (int, float)):
            logger.error("Error: Precio de trailing stop inválido o tipo de dato incorrecto")
            return False

        if not current_price or current_price <= 0 or not isinstance(current_price, (int, float)):
            logger.error("Error: Precio actual inválido o tipo de dato incorrecto")
            return False

        if position_side not in ['LONG', 'SHORT']:
            logger.error(f"Error: Posición inválida {position_side}")
            return False

        if ((position_side == 'LONG' and side == 'SELL' and price >= current_price) or
            (position_side == 'SHORT' and side == 'BUY' and price <= current_price)):
            logger.error("Error: Precio de trailing stop inválido para la dirección de la posición")
            return False

        return True

    @retry(wait=wait_exponential(multiplier=2, min=4, max=20), stop=stop_after_attempt(10))
    async def update_stop_loss_order(self, client, position_side, current_price):
        """Actualiza la orden de trailing stop loss."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        if not await self.validate_order_params(
            'SELL' if position_side == 'LONG' else 'BUY',
            self.trailing_stop,
            position_side,
            current_price
        ):
            return

        timestamp = int(time.time() * 1000)
        if self.stop_loss_order_id:
            try:
                await client.futures_cancel_order(
                    symbol=CONFIG['SYMBOL'],
                    orderId=self.stop_loss_order_id,
                    timestamp=timestamp,
                    recvWindow=CONFIG['RECV_WINDOW']
                )
            except BinanceAPIException as e:
                if 'Unknown order sent' not in str(e):
                    logger.error(f"Error al cancelar trailing stop anterior: {e}")
                    if 'IP has been auto-banned' in str(e):
                        await asyncio.sleep(60)
                    raise

        sl_side = 'SELL' if position_side == 'LONG' else 'BUY'
        safe_sl_price = get_safe_stop_price(
            position_side,
            self.trailing_stop,
            current_price,
            0.0001,  # tick_size
            4  # price_precision
        )

        try:
            sl_order = await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=sl_side,
                type="STOP_MARKET",
                stopPrice=safe_sl_price,
                closePosition=True,
                timestamp=timestamp,
                recvWindow=CONFIG['RECV_WINDOW']
            )

            if not sl_order or 'orderId' not in sl_order:
                logger.error("Error: La orden de trailing stop no se creó correctamente")
                return

            self.stop_loss_order_id = sl_order['orderId']
            self.trailing_update_count += 1
            logger.info(f"Trailing Stop actualizado: {safe_sl_price:.4f} | Orden ID: {self.stop_loss_order_id}")
        except BinanceAPIException as e:
            if 'Account has insufficient balance' in str(e):
                logger.error("Error: Saldo insuficiente para crear la orden de trailing stop")
            elif 'Price less than' in str(e) or 'Price greater than' in str(e):
                logger.error(f"Error: Precio de trailing stop fuera de rango: {safe_sl_price}")
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            else:
                logger.error(f"Error al actualizar trailing stop: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al actualizar trailing stop: {e}")
            raise

    def calculate_new_stop(self, current_price, atr_short, position_side):
        """Calcula el nuevo nivel de trailing stop basado en el ATR."""
        if current_price <= 0 or atr_short <= 0:
            logger.error("Error: Precio actual o ATR inválidos")
            return None

        if position_side not in ['LONG', 'SHORT']:
            logger.error(f"Error: Posición inválida {position_side}")
            return None

        fluctuation_margin = CONFIG['STOP_LOSS_FLUCTUATION_MARGIN']
        multiplier = CONFIG['TRAILING_STOP_MULTIPLIER']
        
        new_stop = (
            current_price - (atr_short * multiplier) - fluctuation_margin
            if position_side == 'LONG' else
            current_price + (atr_short * multiplier) + fluctuation_margin
        )
        
        return new_stop

    async def update_trailing_stop(self, client, position_side, current_price, atr_short):
        """Actualiza el trailing stop de la posición."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        if atr_short <= 0 or current_price <= 0:
            logger.error("Error: ATR o precio actual inválidos")
            return

        new_stop = self.calculate_new_stop(current_price, atr_short, position_side)
        if new_stop is None:
            return

        should_update = (
            (position_side == 'LONG' and new_stop > self.trailing_stop) or
            (position_side == 'SHORT' and new_stop < self.trailing_stop)
        )

        if should_update:
            self.trailing_stop = new_stop
            await self.update_stop_loss_order(client, position_side, current_price)