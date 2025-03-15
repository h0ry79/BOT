# Módulo de gestión de riesgos para el bot de trading.
#
# Este módulo maneja las operaciones de trading, incluyendo la gestión de posiciones,
# stop-loss, take-profit y trailing stops.

from collections import deque
import time
import asyncio
import math
import statistics
import pandas as pd
import numpy as np
from typing import Optional
from binance.enums import ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException
from tenacity import retry, wait_exponential, stop_after_attempt

from config.config import CONFIG
from utils.helpers import (
    get_usdt_balance,
    get_current_price,
    get_open_positions,
    logger
)
from trading.initial_stop_loss import InitialStopLoss
from trading.trailing_stop_loss import TrailingStopLoss
from trading.take_profit import TakeProfit

class RiskManagementBot:
    def __init__(self):
        """Inicializa el bot de gestión de riesgos."""
        self.bid = self.ask = self.current_price = 0.0
        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.total_profit = self.floating_profit = 0.0
        self.prices_long = deque(maxlen=CONFIG['ATR_PERIOD_LONG'])
        self.prices_short = deque(maxlen=CONFIG['ATR_PERIOD_SHORT'])
        self.atr_long = self.atr_short = 0.0
        self.long_atr: Optional[float] = None
        self.short_atr: Optional[float] = None
        self.last_price_update = self.last_reset = time.time()
        self.last_update_time = 0
        self.update_interval = 60  # Actualizar cada minuto
        self.position_lock = asyncio.Lock()
        self.closed_trades = 0
        self.high_short = 0.0
        self.low_short = float('inf')
        self.notifications = []
        
        # Parámetros de gestión de riesgo
        self.risk_per_trade = CONFIG.get('RISK_PER_TRADE', 0.02)  # 2% risk per trade
        self.max_position_size = CONFIG.get('MAX_POSITION_SIZE', 0.1)  # 10% of portfolio
        
        # Inicializar componentes de gestión de riesgo
        self.initial_stop_loss = InitialStopLoss()
        self.trailing_stop_loss = TrailingStopLoss()
        self.take_profit = TakeProfit()
        logger.info("Bot de gestión de riesgos inicializado correctamente")
    
    async def initialize_indicators(self, client) -> bool:
        """Inicializa los indicadores ATR con datos históricos."""
        try:
            if 'LONG_TIMEFRAME' not in CONFIG or 'SHORT_TIMEFRAME' not in CONFIG:
                logger.error("Error: LONG_TIMEFRAME o SHORT_TIMEFRAME no están definidos en la configuración")
                return False

            # Obtener datos históricos para el cálculo inicial
            try:
                # Aumentar el número de velas para asegurar suficientes datos
                min_candles = max(CONFIG['ATR_PERIOD_LONG'] * 2, CONFIG['ATR_PERIOD_SHORT'] * 2)
                limit = min(500, min_candles + 50)  # Añadir margen extra pero limitar a 500
                
                long_klines = await client.futures_klines(
                    symbol=CONFIG['SYMBOL'],
                    interval=CONFIG['LONG_TIMEFRAME'],
                    limit=limit
                )
                
                short_klines = await client.futures_klines(
                    symbol=CONFIG['SYMBOL'],
                    interval=CONFIG['SHORT_TIMEFRAME'],
                    limit=limit
                )
            except Exception as e:
                logger.error(f"Error al obtener datos históricos: {e}")
                return False

            # Verificar que tenemos suficientes datos para el cálculo
            if not long_klines or not short_klines:
                logger.error("No se pudieron obtener datos históricos")
                return False
                
            if len(long_klines) < CONFIG['ATR_PERIOD_LONG'] or len(short_klines) < CONFIG['ATR_PERIOD_SHORT']:
                logger.warning("Insufficient price data for ATR calculation")
                return False

            # Convertir datos a DataFrame
            try:
                long_df = self._prepare_dataframe(long_klines)
                short_df = self._prepare_dataframe(short_klines)
            except Exception as e:
                logger.error(f"Error al preparar DataFrame: {e}")
                return False

            # Calcular ATR inicial
            try:
                self.long_atr = self._calculate_atr(long_df, CONFIG['ATR_PERIOD_LONG'])
                self.short_atr = self._calculate_atr(short_df, CONFIG['ATR_PERIOD_SHORT'])

                if not self.long_atr or not self.short_atr:
                    logger.error("Error: Cálculo de ATR resultó en valores nulos")
                    return False

                logger.info(f"ATR inicializado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")
                return True

            except Exception as e:
                logger.error(f"Error al calcular ATR: {e}")
                return False

        except Exception as e:
            logger.error(f"Error en initialize_indicators: {e}")
            return False
    
    def _prepare_dataframe(self, klines) -> pd.DataFrame:
        """Prepara el DataFrame para el cálculo de ATR."""
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                         'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcula el ATR usando los datos proporcionados."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return float(atr.iloc[-1])

    def add_notification(self, message):
        """Añade una notificación al registro del bot."""
        self.notifications.append(f"{time.strftime('%H:%M:%S')} - {message}")
        logger.info(f"Notificación: {message}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                if not is_noise:
                    # ...existing code...
                else:
                    # ...existing code...
                    logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                    self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                    return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error
        finally:
            # Add a finally block to ensure any necessary cleanup or final steps
            pass

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        if not exit_price:
            self.add_notification("Error: No se pudo obtener el precio de salida")
            return

        if self.trade_quantity < CONFIG.get('MIN_QUANTITY', 0.001):
            self.add_notification(f"Error: Cantidad {self.trade_quantity} menor que el mínimo permitido")
            return

        try:
            order = await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            if not order or 'orderId' not in order:
                self.add_notification("Error: La orden de cierre no se creó correctamente")
                return

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        if not exit_price:
            self.add_notification("Error: No se pudo obtener el precio de salida")
            return

        if self.trade_quantity < CONFIG.get('MIN_QUANTITY', 0.001):
            self.add_notification(f"Error: Cantidad {self.trade_quantity} menor que el mínimo permitido")
            return

        try:
            order = await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            if not order or 'orderId' not in order:
                self.add_notification("Error: La orden de cierre no se creó correctamente")
                return

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        if not exit_price:
            self.add_notification("Error: No se pudo obtener el precio de salida")
            return

        if self.trade_quantity < CONFIG.get('MIN_QUANTITY', 0.001):
            self.add_notification(f"Error: Cantidad {self.trade_quantity} menor que el mínimo permitido")
            return

        try:
            order = await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            if not order or 'orderId' not in order:
                self.add_notification("Error: La orden de cierre no se creó correctamente")
                return

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        try:
            await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        if not exit_price:
            self.add_notification("Error: No se pudo obtener el precio de salida")
            return

        if self.trade_quantity < CONFIG.get('MIN_QUANTITY', 0.001):
            self.add_notification(f"Error: Cantidad {self.trade_quantity} menor que el mínimo permitido")
            return

        try:
            order = await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            if not order or 'orderId' not in order:
                self.add_notification("Error: La orden de cierre no se creó correctamente")
                return

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        if not exit_price:
            self.add_notification("Error: No se pudo obtener el precio de salida")
            return

        if self.trade_quantity < CONFIG.get('MIN_QUANTITY', 0.001):
            self.add_notification(f"Error: Cantidad {self.trade_quantity} menor que el mínimo permitido")
            return

        try:
            order = await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            if not order or 'orderId' not in order:
                self.add_notification("Error: La orden de cierre no se creó correctamente")
                return

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        try:
            await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        try:
            await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        if not exit_price:
            self.add_notification("Error: No se pudo obtener el precio de salida")
            return

        if self.trade_quantity < CONFIG.get('MIN_QUANTITY', 0.001):
            self.add_notification(f"Error: Cantidad {self.trade_quantity} menor que el mínimo permitido")
            return

        try:
            order = await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            if not order or 'orderId' not in order:
                self.add_notification("Error: La orden de cierre no se creó correctamente")
                return

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        try:
            await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                # Mejorar la detección de posiciones con reintentos
                retry_count = 0
                max_retries = 3
                position = None
                
                while retry_count < max_retries:
                    try:
                        position = await get_open_positions(client)
                        if position is not None:  # Verificación explícita
                            break
                        retry_count += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intento {retry_count + 1} fallido al obtener posiciones: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                # Verificar el estado actual de la posición
                current_position_amount = float(position['positionAmt']) if position else 0.0
                position_changed = (
                    (not self.in_position and current_position_amount != 0) or
                    (self.in_position and current_position_amount == 0) or
                    (self.in_position and self.position_side == 'LONG' and current_position_amount < 0) or
                    (self.in_position and self.position_side == 'SHORT' and current_position_amount > 0)
                )

                if position_changed:
                    logger.info(f"Cambio detectado en la posición - Anterior: {self.position_side if self.in_position else 'None'}, "
                               f"Nueva: {'LONG' if current_position_amount > 0 else 'SHORT' if current_position_amount < 0 else 'None'}")

                # Obtener y validar el precio actual
                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                # Obtener y validar el balance
                balance = await get_usdt_balance(client)
                if balance < CONFIG['MIN_BALANCE']:
                    logger.warning(f"Balance insuficiente: {balance} USDT")
                    return

                try:
                    position_amount = float(position['positionAmt'])
                    if abs(position_amount) > 0:
                        if not self.in_position:
                            # Validar el estado de la posición
                            position_info = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
                            if not position_info:
                                logger.error("Error: No se pudo obtener información detallada de la posición")
                                return

                            # Verificar el modo de posición y validar
                            position_mode = await client.futures_get_position_mode()
                            if not isinstance(position_mode, dict) or 'dualSidePosition' not in position_mode:
                                logger.error("Error: No se pudo verificar el modo de posición")
                                return

                            # Calculate maximum position size based on balance and risk with improved validation
                            max_position_value = balance * CONFIG['MAX_POSITION_SIZE_RATIO']
                            
                            # Enhanced volatility check for position sizing with validation
                            if self.atr_long <= 0:
                                logger.warning("ATR no disponible, usando valores predeterminados para el cálculo de volatilidad")
                                volatility_ratio = CONFIG.get('DEFAULT_VOLATILITY_RATIO', 0.02)
                            else:
                                volatility_ratio = self.atr_long / current_price

                            if volatility_ratio > CONFIG.get('MAX_VOLATILITY_RATIO', 0.03):
                                self.add_notification(f"High volatility detected ({volatility_ratio:.2%}). Reducing position size.")
                                max_position_value *= 0.5  # Reduce position size by 50% during high volatility
                        
                        # Calculate risk-adjusted size with minimum position validation
                        min_position_value = CONFIG.get('MIN_POSITION_VALUE', 10.0)  # Minimum position value in USDT
                        if max_position_value < min_position_value:
                            self.add_notification(f"Balance too low for safe trading. Minimum required: {min_position_value} USDT")
                            return
                            
                        risk_adjusted_size = (balance * CONFIG['RISK_PER_TRADE']) / (self.atr_long * CONFIG['STOP_LOSS_ATR_FACTOR'])
                        
                        # Apply dynamic volatility adjustment
                        volatility_factor = min(1.0, CONFIG['VOLATILITY_FACTOR'] / volatility_ratio)
                        risk_adjusted_size *= volatility_factor

                        # Ensure position size doesn't exceed limits with improved safety margins
                        max_quantity = max_position_value / current_price
                        min_quantity = min_position_value / current_price
                        safe_quantity = max(min(risk_adjusted_size, max_quantity), min_quantity)
                        
                        # Enhanced position size validation with liquidation prevention
                        maintenance_margin = CONFIG.get('MAINTENANCE_MARGIN', 0.01)  # 1% maintenance margin
                        max_safe_leverage = (1 / maintenance_margin) * 0.8  # 20% safety buffer
                        if (safe_quantity * current_price) > (balance * max_safe_leverage):
                            safe_quantity = (balance * max_safe_leverage) / current_price
                            self.add_notification("Position size adjusted to prevent liquidation risk")
                        
                        # Validate position size
                        if abs(position_amount) > safe_quantity * (1 + CONFIG['MAX_SIZE_DIFFERENCE']):
                            logger.warning(f"Posición demasiado grande: {abs(position_amount)} > {safe_quantity}")
                            self.add_notification(f"Advertencia: Tamaño de posición excede el límite recomendado")

                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = abs(position_amount)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)
            elif 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
                await asyncio.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en manage_manual_position: {e}")
            await asyncio.sleep(5)
            raise

    async def close_position(self, client, reason="Manual"):
        """Cierra la posición actual."""
        if not self.in_position or self.trade_quantity <= 0:
            return

        close_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
        exit_price = await get_current_price(client)
        try:
            await client.futures_create_order(
                symbol=CONFIG['SYMBOL'],
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity,
                timestamp=int(time.time() * 1000),
                recvWindow=CONFIG['RECV_WINDOW']
            )

            profit = (exit_price - self.entry_price if self.position_side == 'LONG' else
                     self.entry_price - exit_price) * self.trade_quantity
            fees = self.trade_quantity * (self.entry_price + exit_price) * CONFIG['FEE_RATE_MAKER']
            self.total_profit += profit - fees
            self.closed_trades += 1
            self.add_notification(
                f"Posición cerrada @ {exit_price:.4f} | {reason} | P/L: {profit - fees:.4f}"
            )
            await self.reset_position(client)
        except BinanceAPIException as e:
            self.add_notification(f"Error al cerrar posición: {e}")
            logger.error(f"Error detallado al cerrar posición: {str(e)}")

    async def reset_position(self, client=None):
        """Resetea los valores de la posición actual."""
        if client:
            await self.initial_stop_loss.cancel_stop_loss(client)
            # Reiniciar trailing stop loss
            self.trailing_stop_loss.trailing_stop = 0.0
            self.trailing_stop_loss.stop_loss_order_id = None
            self.trailing_stop_loss.trailing_update_count = 0

        self.in_position = False
        self.position_side = None
        self.trade_quantity = self.initial_quantity = self.entry_price = 0.0
        self.floating_profit = 0.0
        self.take_profit.reset()
        self.add_notification("Posición reseteada")

    async def update_indicators(self, client=None):
        """Actualiza los indicadores técnicos del bot."""
        try:
            if not client:
                return

            current_market_price = await get_current_price(client)
            if not current_market_price or current_market_price <= 0:
                logger.warning("Invalid market price received")
                return

            # Validate price data
            if not self.prices_long or not self.prices_short:
                logger.warning("Insufficient price data for ATR calculation")
                return

            # Store previous price before updating
            previous_price = self.current_price

            # Enhanced price staleness validation with dynamic threshold
            current_time = time.time()
            price_age = current_time - self.last_price_update
            volatility_based_threshold = self.atr_long / self.current_price if self.atr_long > 0 else CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            max_price_age = CONFIG.get('MAX_PRICE_AGE', 3) * (1 + min(volatility_based_threshold, 0.1))
            
            if price_age > max_price_age:
                logger.warning(f"Price data is stale: {price_age:.2f} seconds old (max: {max_price_age:.2f}s)")
                return

            # Enhanced price validation with exponential moving average
            if self.current_price > 0:
                ema_factor = CONFIG.get('EMA_FACTOR', 0.1)
                expected_price_range = [self.current_price * (1 - CONFIG.get('MAX_PRICE_DEVIATION', 0.1)),
                                      self.current_price * (1 + CONFIG.get('MAX_PRICE_DEVIATION', 0.1))]
                if (current_market_price < expected_price_range[0] or 
                    current_market_price > expected_price_range[1]):
                    logger.warning(f"Price deviation detected: {current_market_price} vs expected range {expected_price_range}")
                    return
                
                # Smooth price updates with EMA
                self.current_price = current_market_price * ema_factor + self.current_price * (1 - ema_factor)
            else:
                self.current_price = current_market_price

            self.bid = self.ask = self.current_price
            self.last_price_update = current_time

            # Advanced volatility spike detection with dynamic thresholds and market regime detection
            price_change_percent = abs(self.current_price - previous_price) / previous_price if 'previous_price' in locals() else 0
            base_volatility_threshold = CONFIG.get('MAX_PRICE_CHANGE', 0.03)
            historical_volatility = self.atr_long / self.current_price if self.atr_long > 0 else base_volatility_threshold
            
            # Market regime detection using volatility ratios
            volatility_ratio = self.atr_short / self.atr_long if self.atr_long > 0 else 1
            regime_multiplier = math.exp(-volatility_ratio) # Lower multiplier in high volatility regimes
            
            # Adaptive threshold based on market regime
            dynamic_threshold = base_volatility_threshold * regime_multiplier
            
            # Exponential weighting for recent price changes with decay factor
            decay_factor = math.exp(-price_age / CONFIG.get('PRICE_DECAY_FACTOR', 30))
            weighted_change = price_change_percent * decay_factor
            
            # Dynamic threshold based on market conditions
            market_stress_factor = 1 + (self.atr_short / self.atr_long if self.atr_long > 0 else 1)
            dynamic_threshold = base_volatility_threshold * market_stress_factor
            
            if weighted_change > dynamic_threshold:
                logger.warning(f"Abnormal volatility detected: {weighted_change:.2%} change (threshold: {dynamic_threshold:.2%})")
                self.add_notification(f"High volatility alert: {weighted_change:.2%} price change exceeds {dynamic_threshold:.2%} threshold")
                return

            # Enhanced high/low tracking with adaptive reset interval
            reset_interval = CONFIG.get('PRICE_RESET_INTERVAL', 60) * (1 + historical_volatility)
            if current_time - self.last_reset >= reset_interval:
                self.high_short = self.low_short = self.current_price
                self.last_reset = current_time
            else:
                self.high_short = max(self.high_short, self.current_price)
                self.low_short = min(self.low_short, self.current_price)

            # Improved True Range calculation with Garman-Klass volatility estimator
            high_low = self.high_short - self.low_short
            high_close = abs(self.high_short - previous_price) if 'previous_price' in locals() else 0
            low_close = abs(self.low_short - previous_price) if 'previous_price' in locals() else 0
            
            # Enhanced True Range calculation with Yang-Zhang volatility estimator
            open_close = abs(self.current_price - previous_price) if 'previous_price' in locals() else 0
            tr = math.sqrt(
                0.5 * high_low ** 2 +
                0.25 * (open_close ** 2) +  # Overnight jump component
                0.25 * (high_close * low_close)  # Trading period volatility
            ) if high_low > 0 else max(high_close, low_close, open_close)

            # Extreme market condition detection with circuit breaker
            volatility_spike = False
            if self.atr_short > 0:
                current_volatility = tr / self.atr_short
                if current_volatility > CONFIG.get('VOLATILITY_THRESHOLD', 3.0):
                    volatility_spike = True
                    logger.warning(f"Critical: Extreme volatility detected: {current_volatility:.2f}x normal levels")
                    self.add_notification(f"CRITICAL: Market volatility warning: {current_volatility:.2f}x above normal")
                    # Circuit breaker: Force position check on extreme volatility
                    if self.in_position:
                        await self.check_position_risk(client)
                    return

            # Dynamic minimum TR threshold with enhanced market condition adjustment
            base_min_tr = self.current_price * CONFIG.get('MIN_TR_THRESHOLD', 0.0001)
            market_condition_factor = 1 + (historical_volatility * CONFIG.get('VOLATILITY_IMPACT', 0.5))
            min_tr = base_min_tr * market_condition_factor * (1 + volatility_ratio)
            tr = max(tr, min_tr)

            # Apply Kalman filter for noise reduction
            if self.atr_short > 0:
                innovation = tr - self.atr_short
                measurement_noise = 0.1
                process_noise = 0.01
                kalman_gain = process_noise / (process_noise + measurement_noise)
                tr = self.atr_short + kalman_gain * innovation

            # Update ATR with weighted moving average
            alpha_long = 2.0 / (CONFIG['ATR_PERIOD_LONG'] + 1)
            alpha_short = 2.0 / (CONFIG['ATR_PERIOD_SHORT'] + 1)

            if self.atr_long == 0:
                self.atr_long = tr
            else:
                self.atr_long = (tr * alpha_long) + (self.atr_long * (1 - alpha_long))

            if self.atr_short == 0:
                self.atr_short = tr
            else:
                self.atr_short = (tr * alpha_short) + (self.atr_short * (1 - alpha_short))
                    
                # Market microstructure noise filtering with enhanced validation
                noise_threshold = max(
                    self.current_price * CONFIG.get('NOISE_THRESHOLD', 0.0001),
                    min_tr * 0.1  # Minimum noise threshold based on TR
                )
                is_noise = (tr < noise_threshold and 
                          abs(self.current_price - previous_price) < noise_threshold and
                          len(self.prices_long) >= 3)  # Require minimum price history
                
                # Define default values for modified_zscore and z_score_threshold
                modified_zscore = 0
                z_score_threshold = 3.0
                
                if not is_noise:
                    # Apply volume-weighted adjustment with enhanced smoothing
                    volume_factor = 1.0  # Can be enhanced with actual volume data
                    adjusted_tr = tr * volume_factor
                    
                    # Enhanced Kalman filter with adaptive gain
                    if self.atr_short > 0:
                        innovation = adjusted_tr - self.atr_short
                        # Adaptive Kalman gain based on volatility regime
                        base_gain = min(abs(innovation) / self.atr_short, 0.1)
                        regime_adjusted_gain = base_gain * regime_multiplier
                        kalman_gain = min(max(regime_adjusted_gain, 0.01), 0.2)
                        adjusted_tr = self.atr_short + kalman_gain * innovation
                        
                        # Ensure minimum TR value and prevent extreme jumps
                        adjusted_tr = max(adjusted_tr, min_tr)
                        if self.atr_short > 0:
                            max_tr_change = self.atr_short * CONFIG.get('MAX_TR_CHANGE', 3.0)
                            adjusted_tr = min(max(adjusted_tr, self.atr_short / max_tr_change),
                                             self.atr_short * max_tr_change)
                        
                        self.prices_long.append(adjusted_tr)
                        self.prices_short.append(adjusted_tr)
                else:
                    # Initialize ATR values with proper validation
                    if tr > min_tr:
                        self.prices_long.append(tr)
                        self.prices_short.append(tr)
                        # Set initial ATR values
                        if self.atr_long == 0:
                            self.atr_long = tr
                        if self.atr_short == 0:
                            self.atr_short = tr

            # Improved ATR calculation with adaptive alpha and volatility regime detection
            if len(self.prices_long) >= 2:
                window_size_long = min(len(self.prices_long), CONFIG['ATR_PERIOD_LONG'])
                # Enhanced regime detection with volatility clustering
                volatility_cluster = sum(1 for p in list(self.prices_long)[-5:] if p > self.atr_long) / 5
                regime_factor = math.exp(volatility_cluster - 0.5)  # Exponential scaling based on volatility clustering
                
                # Dynamic alpha adjustment with volatility feedback
                base_alpha_long = 2 / (window_size_long + 1)
                volatility_feedback = min(tr / self.atr_long, 2.0) if self.atr_long > 0 else 1.0
                alpha_long = base_alpha_long * regime_factor * volatility_feedback
                alpha_long = min(max(alpha_long, 0.01), 0.5)  # Bound alpha between 1% and 50%
                
                # Volatility regime transition smoothing
                if self.atr_long > 0:
                    regime_transition = abs(tr - self.atr_long) / self.atr_long
                    smoothing_factor = math.exp(-regime_transition)  # Smoother transitions in stable regimes
                    self.atr_long = tr * (alpha_long * smoothing_factor) + self.atr_long * (1 - (alpha_long * smoothing_factor))
                else:
                    self.atr_long = tr

            if len(self.prices_short) >= 2:
                window_size_short = min(len(self.prices_short), CONFIG['ATR_PERIOD_SHORT'])
                # More responsive alpha for short-term with volatility feedback
                base_alpha_short = 2 / (window_size_short + 1)
                volatility_feedback = min(tr / self.atr_short, 2.5) if self.atr_short > 0 else 1.0
                alpha_short = base_alpha_short * regime_factor * volatility_feedback
                alpha_short = min(max(alpha_short, 0.02), 0.6)  # More responsive bounds
                
                # Enhanced short-term volatility detection
                if self.atr_short > 0:
                    short_regime_transition = abs(tr - self.atr_short) / self.atr_short
                    short_smoothing = math.exp(-short_regime_transition * 0.8)  # Less smoothing for faster response
                    self.atr_short = tr * (alpha_short * short_smoothing) + self.atr_short * (1 - (alpha_short * short_smoothing))
                else:
                    self.atr_short = tr

            # Apply dynamic volatility adjustments
            volatility_factor = 1.0
            if len(self.prices_long) >= 5:
                recent_volatility = statistics.stdev([float(p) for p in list(self.prices_long)[-5:]])
                if recent_volatility > self.atr_long:
                    volatility_factor = min(recent_volatility / self.atr_long, 1.5)

            self.atr_long *= volatility_factor 
            self.atr_short *= volatility_factor

            # Ensure minimum ATR values
            min_atr = self.current_price * CONFIG.get('MIN_ATR_THRESHOLD', 0.0005)
            self.atr_long = max(self.atr_long, min_atr)
            self.atr_short = max(self.atr_short, min_atr)
            
            # Ensure short-term ATR doesn't deviate too much from long-term ATR
            if self.atr_long > 0:
                atr_ratio = self.atr_short / self.atr_long
                if atr_ratio > max_atr_ratio:
                    self.atr_short = self.atr_long * max_atr_ratio
                    logger.warning(f"Short-term ATR capped due to excessive deviation (ratio: {atr_ratio:.2f})")
            
            # Update the long_atr and short_atr attributes for UI display
            self.long_atr = self.atr_long
            self.short_atr = self.atr_short
            logger.debug(f"ATR actualizado - Largo: {self.long_atr:.8f}, Corto: {self.short_atr:.8f}")

            # Enhanced profit calculation with improved fee consideration and slippage estimation
            if self.in_position and self.entry_price:
                position_value = self.trade_quantity * self.current_price
                maker_fee = CONFIG.get('FEE_RATE_MAKER', 0.0002)
                taker_fee = CONFIG.get('FEE_RATE_TAKER', 0.0004)
                
                # Dynamic slippage estimation based on market conditions
                base_slippage = CONFIG.get('BASE_SLIPPAGE', 0.0005)
                volatility_impact = (self.atr_short / self.current_price) if self.atr_short > 0 else base_slippage
                market_impact = (self.trade_quantity * current_market_price) / CONFIG.get('MARKET_DEPTH', 100000)
                slippage_estimate = max(base_slippage, volatility_impact * market_impact)
                
                # Total transaction costs including spread
                spread = (self.ask - self.bid) / self.current_price if self.ask > self.bid else taker_fee
                total_cost = position_value * (maker_fee + taker_fee + slippage_estimate + spread)

                # Calculate base profit with price normalization
                price_ratio = self.current_price / self.entry_price
                base_profit = (price_ratio - 1 if self.position_side == 'LONG' else 1 - price_ratio) * position_value
                
                # Apply dynamic profit adjustment based on holding time
                holding_time = time.time() - self.entry_time if hasattr(self, 'entry_time') else 0
                time_decay = math.exp(-holding_time / CONFIG.get('PROFIT_DECAY_FACTOR', 86400))
                adjusted_profit = base_profit * time_decay
                
                self.floating_profit = adjusted_profit - total_cost

                # Enhanced trailing stop logic with dynamic adjustments
                if self.atr_short > 0 and self.floating_profit > 0:
                    # Calculate volatility-adjusted stop distance
                    volatility_ratio = self.atr_short / self.current_price
                    base_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
                    volatility_factor = min(1.0, base_factor / max(volatility_ratio, 0.001))
                    
                    # Apply market regime adjustments
                    trend_strength = abs(self.current_price - self.entry_price) / (self.atr_long if self.atr_long > 0 else self.atr_short)
                    regime_factor = math.tanh(trend_strength)  # Smooth transition between regimes
                    regime_adjustment = 1 - (weighted_change / dynamic_threshold if 'weighted_change' in locals() else 0)
                    
                    # Combine adjustments with profit-based scaling
                    profit_factor = min(1.0, self.floating_profit / (position_value * CONFIG.get('TARGET_PROFIT', 0.02)))
                    final_adjustment = volatility_factor * regime_adjustment * (1 + profit_factor)
                    
                    # Update trailing stop with combined adjustments
                    await self.trailing_stop_loss.update_trailing_stop(
                        client=client,
                        position_side=self.position_side,
                        current_price=self.current_price,
                        atr_short=self.atr_short * final_adjustment
                    )

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
            # Preserve existing values on error

    async def calculate_position_size(self, client, entry_price):
        """Calcula el tamaño de la posición basado en el balance y el riesgo."""
        try:
            balance = await get_usdt_balance(client)
            if not balance or balance <= 0:
                self.add_notification("Error: No se pudo obtener el balance o balance insuficiente")
                return 0.0

            # Validar el balance mínimo requerido con margen de seguridad
            min_balance = CONFIG.get('MIN_BALANCE', 10.0) * 1.1  # 10% extra de margen
            if balance < min_balance:
                self.add_notification(f"Error: Balance {balance:.2f} USDT menor que el mínimo requerido {min_balance} USDT")
                return 0.0

            # Obtener y validar el apalancamiento
            try:
                leverage_info = await client.futures_leverage_bracket(symbol=CONFIG['SYMBOL'])
                max_leverage = float(leverage_info[0]['brackets'][0]['initialLeverage'])
                current_leverage = min(CONFIG.get('MAX_LEVERAGE', 20), max_leverage)
                
                # Validar el apalancamiento mínimo seguro
                if current_leverage < 2:
                    self.add_notification("Error: Apalancamiento demasiado bajo para operar de forma segura")
                    return 0.0
            except Exception as e:
                logger.error(f"Error al obtener información de apalancamiento: {e}")
                return 0.0

            # Calcular el riesgo por operación con límites dinámicos basados en el balance
            base_risk = CONFIG.get('RISK_PER_TRADE', 0.01)
            balance_factor = min(1.0, balance / CONFIG.get('OPTIMAL_BALANCE', 1000.0))
            adjusted_risk = base_risk * balance_factor
            max_risk_percent = CONFIG.get('MAX_RISK_PER_TRADE', 0.02)
            min_risk_percent = CONFIG.get('MIN_RISK_PER_TRADE', 0.005)
            risk_percent = min(max(adjusted_risk, min_risk_percent), max_risk_percent)
            risk_amount = balance * risk_percent
            
            # Usar ATR para calcular la distancia del stop loss con validación mejorada
            if self.atr_long <= 0:
                self.add_notification("Error: ATR no disponible para calcular el tamaño de la posición")
                return 0.0
                
            # Calcular la distancia del stop loss basada en ATR y volatilidad con ajuste dinámico
            volatility_factor = CONFIG.get('VOLATILITY_FACTOR', 1.5)
            market_volatility = self.atr_long / self.current_price
            if market_volatility > 0.03:  # Alta volatilidad
                volatility_factor *= 1.2
            stop_distance = self.atr_long * volatility_factor
            
            # Validar la distancia mínima del stop loss
            min_stop_distance = entry_price * CONFIG.get('MIN_STOP_DISTANCE', 0.003)
            stop_distance = max(stop_distance, min_stop_distance)
            
            # Calcular el tamaño de la posición considerando el apalancamiento y la volatilidad
            position_size = (risk_amount * current_leverage) / stop_distance
            
            # Ajustar por el precio de entrada con validación
            if entry_price <= 0:
                self.add_notification("Error: Precio de entrada inválido")
                return 0.0
            position_size = position_size / entry_price
            
            # Validar límites de la posición con margen de seguridad
            min_qty = CONFIG.get('MIN_QUANTITY', 0.001) * 1.1  # 10% extra de margen
            max_position_value = balance * current_leverage * CONFIG.get('MAX_POSITION_SIZE_RATIO', 0.95)
            max_qty = (max_position_value / entry_price) * 0.95  # 5% de margen de seguridad

            # Aplicar límites con notificaciones detalladas
            if position_size < min_qty:
                self.add_notification(f"Advertencia: Tamaño de posición {position_size:.4f} demasiado pequeño, ajustando al mínimo {min_qty}")
                position_size = min_qty
            elif position_size > max_qty:
                self.add_notification(f"Advertencia: Tamaño de posición {position_size:.4f} demasiado grande, ajustando al máximo {max_qty:.4f}")
                position_size = max_qty

            # Redondear al número de decimales permitido
            decimals = CONFIG.get('QUANTITY_PRECISION', 3)
            position_size = round(position_size, decimals)
            
            # Validación final de la posición con margen de liquidación
            position_value = position_size * entry_price
            max_allowed_value = balance * current_leverage * 0.95  # 5% margen de liquidación
            if position_value > max_allowed_value:
                adjusted_size = (max_allowed_value / entry_price) * 0.95
                self.add_notification(f"Ajustando tamaño para prevenir liquidación: {adjusted_size:.4f}")
                return round(adjusted_size, decimals)
                
            return position_size
            
        except Exception as e:
            logger.error(f"Error al calcular el tamaño de la posición: {e}")
            return 0.0

    async def manage_manual_position(self, client):
        """Gestiona las posiciones manuales abiertas."""
        if not client:
            logger.error("Error: Cliente no inicializado")
            return

        try:
            async with self.position_lock:
                position = await get_open_positions(client)
                if not position:
                    if self.in_position:
                        await self.reset_position(client)
                    return

                current_price = await get_current_price(client)
                if not current_price:
                    logger.error("Error: No se pudo obtener el precio actual")
                    return

                position_amount = float(position['positionAmt'])
                if abs(position_amount) > 0:
                    if not self.in_position:
                        # Calcular el tamaño de posición adecuado
                        calculated_size = await self.calculate_position_size(client, current_price)
                        if calculated_size <= 0:
                            logger.error("Error: No se pudo calcular el tamaño de la posición")
                            return
                            
                        self.in_position = True
                        self.position_side = 'LONG' if position_amount > 0 else 'SHORT'
                        self.entry_price = float(position['entryPrice'])
                        self.trade_quantity = min(abs(position_amount), calculated_size)
                        self.initial_quantity = self.trade_quantity
                        self.add_notification(f"Posición manual detectada: {self.position_side} | Cantidad: {self.trade_quantity:.4f} | Entrada: {self.entry_price:.4f}")
                        
                        # Configurar stop loss inicial y take profit
                        if self.atr_long > 0:
                            await self.initial_stop_loss.configure_stop_loss(
                                client,
                                self.position_side,
                                self.entry_price,
                                current_price,
                                self.atr_long
                            )
                            self.take_profit.configure_take_profit(
                                self.entry_price,
                                self.atr_long,
                                self.position_side
                            )
                        else:
                            logger.warning("ATR no disponible para configurar stop loss y take profit")
                else:
                    if self.in_position:
                        await self.reset_position(client)

        except BinanceAPIException as e:
            logger.error(f"Error de Binance API en manage_manual_position: {e}")
            if 'Connection refused' in str(e):
                logger.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)