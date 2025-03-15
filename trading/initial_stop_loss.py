# Módulo para la gestión del stop loss inicial.
#
# Este módulo maneja la configuración y gestión del stop loss inicial para las posiciones de trading.

from binance.exceptions import BinanceAPIException
from tenacity import retry, wait_exponential, stop_after_attempt
import time
import asyncio

from config.config import CONFIG
from utils.helpers import get_safe_stop_price, logger

class InitialStopLoss:
    def __init__(self):
        """Inicializa el gestor de stop loss inicial."""
        self.stop_loss_order_id = None
        self.sl_price = 0.0

    async def validate_order_params(self, side, price, position_side, current_price):
        """Valida los parámetros de la orden de stop loss."""
        if price <= 0 or not isinstance(price, (int, float)):
            logger.error("Error: Precio de stop loss inválido o tipo de dato incorrecto")
            return False

        if not current_price or current_price <= 0 or not isinstance(current_price, (int, float)):
            logger.error("Error: Precio actual inválido o tipo de dato incorrecto")
            return False

        if position_side not in ['LONG', 'SHORT']:
            logger.error(f"Error: Posición inválida {position_side}")
            return False

        if ((position_side == 'LONG' and side == 'SELL' and price >= current_price) or
            (position_side == 'SHORT' and side == 'BUY' and price <= current_price)):
            logger.error("Error: Precio de stop loss inválido para la dirección de la posición")
            return False

        return True

    @retry(wait=wait_exponential(multiplier=2, min=4, max=20), stop=stop_after_attempt(10))
    async def configure_stop_loss(self, client, position_side, entry_price, current_price, atr_long):
        """Configura el stop loss inicial para la posición."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return None

        if entry_price <= 0:
            logger.error("Error: Precio de entrada inválido")
            return None

        sl_distance = max(atr_long * 5.0 if atr_long > 0 else 0.002, 0.0001 * 10)
        self.sl_price = (
            entry_price - sl_distance - CONFIG['STOP_LOSS_FLUCTUATION_MARGIN']
            if position_side == 'LONG' else
            entry_price + sl_distance + CONFIG['STOP_LOSS_FLUCTUATION_MARGIN']
        )

        if not await self.validate_order_params(
            'SELL' if position_side == 'LONG' else 'BUY',
            self.sl_price,
            position_side,
            current_price
        ):
            return None

        timestamp = int(time.time() * 1000)
        sl_side = 'SELL' if position_side == 'LONG' else 'BUY'
        safe_sl_price = get_safe_stop_price(
            position_side,
            self.sl_price,
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
                logger.error("Error: La orden de stop loss no se creó correctamente")
                return None

            self.stop_loss_order_id = sl_order['orderId']
            logger.info(f"Stop-Loss configurado: {safe_sl_price:.4f}")
            return self.stop_loss_order_id
        except BinanceAPIException as e:
            if 'Account has insufficient balance' in str(e):
                logger.error("Error: Balance insuficiente para configurar stop loss")
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            else:
                logger.error(f"Error al configurar SL: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al configurar SL: {e}")
            raise

    async def cancel_stop_loss(self, client):
        """Cancela la orden de stop loss actual."""
        if not self.stop_loss_order_id:
            return

        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            await client.futures_cancel_order(
                symbol=CONFIG['SYMBOL'],
                orderId=self.stop_loss_order_id,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )
            logger.info(f"Stop Loss cancelado: OrderID {self.stop_loss_order_id}")
            self.stop_loss_order_id = None
        except BinanceAPIException as e:
            if 'Unknown order sent' not in str(e):
                logger.error(f"Error al cancelar SL: {e}")
                if 'IP has been auto-banned' in str(e):
                    await asyncio.sleep(60)
            self.stop_loss_order_id = None
        except Exception as e:
            logger.error(f"Error inesperado al cancelar SL: {e}")
            self.stop_loss_order_id = None