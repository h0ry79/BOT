"""Módulo de utilidades y funciones auxiliares.

Este módulo contiene funciones de utilidad y helpers que son utilizados
por otros módulos del bot de trading.
"""

import time
import asyncio
import random
from binance.async_client import AsyncClient
from binance.exceptions import BinanceAPIException
import logging
import logging.handlers

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.handlers.RotatingFileHandler(
    "risk_management_bot.log",
    maxBytes=5*1024*1024,
    backupCount=3
)
stream_handler = logging.StreamHandler()
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [file_handler, stream_handler]

# Variables globales
server_time_offset = 0
last_sync_time = 0

def get_safe_stop_price(position_side, stop_price, current_price, tick_size, price_precision):
    """Calcula un precio de stop seguro basado en el lado de la posición y el tick size."""
    safe_price = round(stop_price / tick_size) * tick_size
    safe_price = round(safe_price, price_precision)
    
    if position_side == 'LONG' and safe_price >= current_price:
        safe_price = current_price - tick_size
    elif position_side == 'SHORT' and safe_price <= current_price:
        safe_price = current_price + tick_size
        
    return safe_price

async def get_usdt_balance(client):
    """Obtiene el balance de USDT disponible."""
    try:
        account = await client.futures_account()
        for asset in account['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
        return 0.0
    except BinanceAPIException as e:
        logger.error(f"Error al obtener balance: {e}")
        return 0.0

async def get_current_price(client):
    """Obtiene el precio actual del símbolo configurado."""
    from config.config import CONFIG
    try:
        if not client:
            logger.error("Error: Cliente no inicializado para obtener precio")
            return 0.0
            
        ticker = await client.futures_symbol_ticker(symbol=CONFIG['SYMBOL'])
        price = float(ticker['price'])
        
        if price <= 0:
            logger.error(f"Error: Precio inválido recibido: {price}")
            return 0.0
            
        return price
    except BinanceAPIException as e:
        logger.error(f"Error al obtener precio actual: {e}")
        if 'IP has been auto-banned' in str(e):
            logger.error("Error: IP bloqueada temporalmente. Esperando antes de reintentar...")
            await asyncio.sleep(60)
        return 0.0
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error al procesar precio: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error inesperado al obtener precio: {e}")
        return 0.0

async def get_open_positions(client):
    """Obtiene las posiciones abiertas para el símbolo configurado."""
    from config.config import CONFIG
    try:
        # Verificar el estado de la conexión antes de consultar posiciones
        await client.futures_ping()
        
        # Obtener información de la posición con validación mejorada
        positions = await client.futures_position_information(symbol=CONFIG['SYMBOL'])
        if not positions:
            logger.warning("No se recibió información de posiciones")
            return None
            
        # Validar la respuesta de la API
        if not isinstance(positions, list):
            logger.error(f"Formato de respuesta inesperado: {type(positions)}")
            return None
            
        # Buscar la posición para el símbolo configurado
        for position in positions:
            if position.get('symbol') == CONFIG['SYMBOL']:
                # Validar campos críticos
                required_fields = ['positionAmt', 'entryPrice', 'symbol', 'leverage']
                if all(field in position for field in required_fields):
                    try:
                        # Validar valores numéricos
                        float(position['positionAmt'])
                        float(position['entryPrice'])
                        return position
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error al convertir valores de posición: {e}")
                        return None
                else:
                    logger.error(f"Campos faltantes en la información de posición: {position}")
                    return None
        
        return None
    except BinanceAPIException as e:
        if 'Invalid API-key' in str(e):
            logger.error("Error de autenticación: Verifique sus credenciales de API")
        elif 'IP has been auto-banned' in str(e):
            logger.warning("IP bloqueada temporalmente, esperando antes de reintentar...")
            await asyncio.sleep(60)
        else:
            logger.error(f"Error al obtener posiciones: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado al obtener posiciones: {e}")
        return None

async def synchronize_time(client, max_retries=3):
    """Sincroniza el tiempo local con el servidor de Binance.
    
    Args:
        client: Cliente de Binance inicializado
        max_retries: Número máximo de reintentos en caso de error
        
    Returns:
        int: Diferencia de tiempo en milisegundos
        
    Raises:
        ValueError: Si no se puede obtener el tiempo del servidor
        BinanceAPIException: Si hay un error en la API de Binance
    """
    retry_count = 0
    base_delay = 2
    last_error = None
    
    while retry_count < max_retries:
        try:
            response = await client.get_server_time()
            server_time = response if isinstance(response, int) else response.get('serverTime')
            
            if not server_time:
                raise ValueError("No se pudo obtener el tiempo del servidor")
                
            time_diff = server_time - int(time.time() * 1000)
            if abs(time_diff) > 5000:  # Advertir si la diferencia es mayor a 5 segundos
                logger.warning(f"Diferencia de tiempo significativa detectada: {time_diff}ms")
            else:
                logger.info(f"Diferencia de tiempo con el servidor: {time_diff}ms")
            return time_diff
            
        except BinanceAPIException as e:
            last_error = e
            if 'Invalid API-key' in str(e):
                logger.error("Error: Verifique sus credenciales de API")
                raise
            elif 'IP has been auto-banned' in str(e):
                logger.warning("IP bloqueada temporalmente, esperando antes de reintentar...")
                await asyncio.sleep(60)
            else:
                delay = min(base_delay ** retry_count, 30)
                logger.warning(f"Error al sincronizar tiempo, reintento {retry_count + 1}/{max_retries} en {delay}s")
                await asyncio.sleep(delay)
            retry_count += 1
            
        except Exception as e:
            last_error = e
            delay = min(base_delay ** retry_count, 30)
            logger.error(f"Error inesperado al sincronizar tiempo: {e}")
            logger.warning(f"Reintentando {retry_count + 1}/{max_retries} en {delay}s")
            await asyncio.sleep(delay)
            retry_count += 1
    
    if last_error:
        logger.error(f"Fallo en la sincronización de tiempo después de {max_retries} intentos")
        raise last_error

async def verify_connection(client):
    """Verifica el estado de la conexión con Binance usando futures_ping y sincronización de tiempo.
    
    Args:
        client: Cliente de Binance inicializado
        
    Returns:
        tuple: (bool, AsyncClient) - Estado de la conexión y cliente actualizado si hubo reconexión
    """
    if not client:
        logger.error("Cliente no inicializado para verificar conexión")
        return False, None
        
    try:
        # Verificar ping y balance como pruebas de conectividad
        await client.futures_ping()
        balance = await get_usdt_balance(client)
        
        if balance is None:
            logger.warning("No se pudo obtener el balance, posible problema de conexión")
            await cleanup_client(client)
            return False, None
            
        # Verificar tiempo de sincronización con umbrales adaptativos y backoff exponencial
        time_diff = await synchronize_time(client)
        warning_threshold = 700  # Aumentado para evitar reconexiones innecesarias (700ms)
        critical_threshold = 1000  # Aumentado para evitar reconexiones innecesarias (1000ms)
        severe_threshold = 1500  # Aumentado para evitar reconexiones innecesarias (1500ms)
        max_time_diff = 50  # Reducido para validación más estricta (50ms)
        max_sync_attempts = 3  # Optimizado para respuesta más rápida
        sync_attempt = 0
        last_sync_check = time.time()
        global last_sync_time, server_time_offset
        current_time = time.time()
        server_time_offset = time_diff  # Actualizar offset global

        # Ajuste dinámico de umbrales basado en la historia de desviación
        dynamic_threshold_factor = 1.0

        # Monitoreo de desviación de tiempo acumulada y ajuste dinámico de umbrales
        if not hasattr(verify_connection, '_time_drift_history'):
            verify_connection._time_drift_history = []
        verify_connection._time_drift_history.append(abs(time_diff))
        if len(verify_connection._time_drift_history) > 10:
            verify_connection._time_drift_history.pop(0)
        avg_drift = sum(verify_connection._time_drift_history) / len(verify_connection._time_drift_history)

        # Ajustar umbrales dinámicamente basado en la historia de desviación
        if avg_drift > warning_threshold:
            dynamic_threshold_factor = max(0.8, min(1.2, avg_drift / warning_threshold))
            warning_threshold *= dynamic_threshold_factor
            critical_threshold *= dynamic_threshold_factor
            logger.info(f"Umbrales ajustados dinámicamente. Factor: {dynamic_threshold_factor:.2f}")

        # Forzar reconexión si la desviación promedio es persistentemente alta
        if avg_drift > critical_threshold * 1.5:  # Umbral aumentado para evitar reconexiones innecesarias
            logger.error(f"Desviación promedio persistentemente alta: {avg_drift:.2f}ms. Forzando reconexión...")
            await cleanup_client(client)
            return False, None
        
        # Verificar si es la primera sincronización
        is_initial_sync = last_sync_time == 0
        
        # Forzar reconexión inmediata si la desincronización es severa
        if abs(time_diff) > 3000:  # Reducido a 3 segundos para mayor seguridad
            logger.error(f"Desincronización crítica detectada: {time_diff}ms > 3000ms. Forzando reconexión inmediata...")
            await cleanup_client(client)
            # Esperar antes de intentar reconexión para evitar sobrecarga
            await asyncio.sleep(1.5)  # Reducido para respuesta más rápida
            return False, None
        
        # Forzar reconexión si la desincronización es severa pero menor a 5 segundos
        if abs(time_diff) > severe_threshold:
            logger.error(f"Desincronización severa detectada: {time_diff}ms > {severe_threshold}ms. Forzando reconexión...")
            await cleanup_client(client)
            return False, None
            
        # Forzar sincronización más frecuente y agresiva
        sync_interval = 15 if is_initial_sync else 30  # Sincronización más frecuente al inicio
        if current_time - last_sync_time > sync_interval or abs(time_diff) > critical_threshold:
            logger.info(f"Iniciando sincronización {'inicial' if is_initial_sync else 'programada'} de tiempo")
            time_diff = await synchronize_time(client)
            server_time_offset = time_diff  # Actualizar offset global
            last_sync_time = current_time
        
        while abs(time_diff) > warning_threshold and sync_attempt < max_sync_attempts:
            logger.warning(f"Intento de sincronización {sync_attempt + 1}/{max_sync_attempts}")
            if abs(time_diff) > critical_threshold * 1.2:  # Aumentado para evitar reconexiones innecesarias
                logger.error(f"Desincronización crítica: {time_diff}ms > {critical_threshold * 1.2}ms")
                await cleanup_client(client)
                time_diff = await synchronize_time(client, max_retries=3)
                if abs(time_diff) > warning_threshold * 1.2:  # Aumentado para evitar reconexiones innecesarias
                    logger.error("Sincronización fallida, forzando reconexión")
                    return False, None
            else:
                time_diff = await synchronize_time(client, max_retries=2)
                if abs(time_diff) <= warning_threshold:
                    logger.info(f"Sincronización exitosa: {time_diff}ms")
                    break
                elif abs(time_diff) > critical_threshold:
                    logger.error("Desincronización empeoró, reconectando")
                    return False, None
            sync_attempt += 1
                
        if abs(time_diff) > critical_threshold:
            logger.error(f"Persistencia de desincronización crítica después de {max_sync_attempts} intentos. Iniciando reconexión...")
            await cleanup_client(client)
            
            # Intentar reconexión con backoff exponencial optimizado y límites de tiempo ajustados
            max_attempts = 3  # Reducido para respuesta más rápida
            base_delay = 2.0  # Aumentado para mejor estabilidad inicial
            max_delay = 15  # Extendido para casos de alta latencia
            jitter = 0.1  # Reducido para mayor predictibilidad
            new_client = None
            
            # Registro detallado de intentos de reconexión
            logger.info(f"Iniciando proceso de reconexión con {max_attempts} intentos máximos")
            logger.info(f"Parámetros de backoff: base_delay={base_delay}s, max_delay={max_delay}s, jitter={jitter}")
            
            # Limpiar historial de desviación al reconectar
            verify_connection._time_drift_history = []
            
            # Monitoreo de intentos de reconexión
            if not hasattr(verify_connection, '_reconnect_attempts'):
                verify_connection._reconnect_attempts = []
            verify_connection._reconnect_attempts.append(time.time())
            # Limpiar intentos antiguos (más de 1 hora)
            verify_connection._reconnect_attempts = [t for t in verify_connection._reconnect_attempts if time.time() - t <= 3600]
            
            for attempt in range(max_attempts):
                try:
                    # Crear nuevo cliente con credenciales de configuración
                    from config.config import CONFIG
                    new_client = await AsyncClient.create(
                        api_key=CONFIG['API_KEY'],
                        api_secret=CONFIG['API_SECRET'],
                        testnet=CONFIG.get('TESTNET', True)
                    )
                    
                    # Verificar conectividad básica y sincronización con reintentos optimizados
                    retry_count = 0
                    max_ping_retries = 4  # Optimizado para balance entre confiabilidad y velocidad
                    while retry_count < max_ping_retries:
                        try:
                            await new_client.futures_ping()
                            break
                        except BinanceAPIException as e:
                            retry_count += 1
                            if retry_count == max_ping_retries:
                                raise
                            if 'IP has been auto-banned' in str(e):
                                logger.warning("IP bloqueada, esperando 60 segundos...")
                                await asyncio.sleep(60)
                                continue
                            # Backoff exponencial con jitter adaptativo
                            jitter = random.uniform(0.1, 0.3) * (retry_count + 1)
                            delay = min(1.5 ** retry_count + jitter, 8)
                            logger.info(f"Reintento {retry_count}/{max_ping_retries} en {delay:.1f}s")
                            await asyncio.sleep(delay)
                    
                    new_balance = await get_usdt_balance(new_client)
                    if new_balance is None:
                        raise ValueError("No se pudo obtener balance con el nuevo cliente")
                    
                    new_time_diff = await synchronize_time(new_client, max_retries=3)
                    if abs(new_time_diff) <= max_time_diff:
                        logger.info(f"Reconexión exitosa. Nueva diferencia de tiempo: {new_time_diff}ms")
                        return True, new_client
                    
                    # Limpiar recursos si la sincronización no fue exitosa
                    await cleanup_client(new_client)
                    new_client = None
                    
                    delay = min(base_delay ** attempt, max_delay)
                    logger.warning(f"Intento {attempt + 1}/{max_attempts}: Persiste desincronización ({new_time_diff}ms). Esperando {delay}s")
                    await asyncio.sleep(delay)
                    
                except BinanceAPIException as e:
                    if new_client:
                        await cleanup_client(new_client)
                        new_client = None
                        
                    if 'IP has been auto-banned' in str(e):
                        logger.warning("IP bloqueada temporalmente, esperando antes de reintentar...")
                        await asyncio.sleep(60)
                        continue
                        
                    logger.error(f"Error de API en intento de reconexión {attempt + 1}/{max_attempts}: {e}")
                    if attempt < max_attempts - 1:
                        delay = min(base_delay ** attempt, max_delay)
                        await asyncio.sleep(delay)
                        
                except Exception as e:
                    if new_client:
                        await cleanup_client(new_client)
                        new_client = None
                        
                    logger.error(f"Error inesperado en intento de reconexión {attempt + 1}/{max_attempts}: {e}")
                    if attempt < max_attempts - 1:
                        delay = min(base_delay ** attempt, max_delay)
                        await asyncio.sleep(delay)
                
            logger.error("No se pudo restablecer la sincronización después de múltiples intentos")
            return False, None
            
        if warning_threshold < abs(time_diff) <= critical_threshold:
            logger.warning(f"Desincronización detectada: {time_diff}ms > {warning_threshold}ms. Monitoreando...")
            return True, client
            
        return True, client
        
    except BinanceAPIException as e:
        logger.error(f"Error de API al verificar conexión: {e}")
        if 'IP has been auto-banned' in str(e):
            logger.warning("IP bloqueada temporalmente, esperando antes de reintentar...")
            await asyncio.sleep(60)
        return False, None
        
    except Exception as e:
        logger.error(f"Error inesperado al verificar conexión: {e}")
        return False, None

async def cleanup_client(client):
    """Limpia y cierra el cliente de Binance de forma segura.
    
    Args:
        client: Cliente de Binance a cerrar
    """
    if client:
        try:
            await client.close_connection()
            logger.info("Conexión con Binance cerrada correctamente")
        except Exception as e:
            logger.error(f"Error al cerrar la conexión con Binance: {e}")
    return None

async def handle_shutdown(client):
    """Maneja el cierre ordenado del bot."""
    logger.info("Iniciando proceso de cierre...")
    try:
        if client:
            await cleanup_client(client)
    except Exception as e:
        logger.error(f"Error durante el cierre: {e}")
    finally:
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        logger.info(f"Cancelando {len(tasks)} tareas pendientes")
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Proceso de cierre completado")