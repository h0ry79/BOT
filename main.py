"""Script principal para iniciar el bot de trading.

Este script sirve como punto de entrada principal para el bot de trading,
encargándose de la configuración inicial y el manejo de excepciones básicas.
"""

import asyncio
import logging
import time
import aiohttp
from binance.client import Client
from binance.async_client import AsyncClient
from binance.exceptions import BinanceAPIException
from tenacity import retry, wait_exponential, stop_after_attempt

from config.config import CONFIG, validate_config
from trading.risk_management import RiskManagementBot
from utils.helpers import synchronize_time, get_usdt_balance, verify_connection, cleanup_client, logger
from ui.terminal_ui import TerminalUI

# Get logger instance
logger = logging.getLogger(__name__)

async def initialize_binance_client():
    """Inicializa y verifica la conexión con el cliente de Binance."""
    try:
        client = await AsyncClient.create(
            api_key=CONFIG['API_KEY'],
            api_secret=CONFIG['API_SECRET'],
            testnet=CONFIG['TESTNET']
            # session parameter removed as it's not supported
        )
        # Sincronizar tiempo con el servidor
        time_diff = await synchronize_time(client)
        if time_diff:
            logger.info("Tiempo sincronizado correctamente con el servidor de Binance")
        return client
    except BinanceAPIException as e:
        if 'API-key format invalid' in str(e):
            logger.error("Error: La clave API no tiene el formato correcto. Verifique sus credenciales.")
        elif 'Invalid API-key' in str(e):
            logger.error("Error: Clave API inválida. Verifique sus credenciales.")
        elif 'API-key' in str(e):
            logger.error("Error: Problema con la autenticación de la API. Verifique sus credenciales.")
        else:
            logger.error(f"Error al conectar con Binance: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al inicializar el cliente de Binance: {e}")
        raise

async def synchronize_time(client):
    """Sincroniza el tiempo con el servidor de Binance y ajusta el tiempo local."""
    try:
        server_time = await client.get_server_time()
        local_time = int(time.time() * 1000)
        time_diff = server_time['serverTime'] - local_time
        logger.info(f"Diferencia de tiempo con el servidor: {time_diff}ms")
        return time_diff
    except Exception as e:
        logger.error(f"Error al sincronizar el tiempo con el servidor: {e}")
        return None

async def shutdown(client=None):
    """Maneja el cierre ordenado del bot."""
    logger.info("Iniciando proceso de cierre ordenado")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    for task in tasks:
        task.cancel()
    
    logger.info(f"Cancelando {len(tasks)} tareas pendientes")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    if client:
        try:
            await client.close_connection()
            logger.info("Conexión con Binance liberada correctamente")
        except Exception as e:
            logger.error(f"Error al liberar la conexión con Binance: {e}")


async def main():
    """Función principal que inicia el bot de trading."""
    client = None
    ui = None
    try:
        # Validar la configuración antes de iniciar
        validate_config()
        logger.info("Configuración validada correctamente")
        
        # Removed session creation as it's not needed
        
        retry_count = 0
        max_retries = CONFIG.get('WEBSOCKET_MAX_RETRIES', 10)
        websocket_status = "Disconnected"
        last_error_time = 0
        error_count = 0
        last_ping_time = 0
        ping_interval = 30
        indicators_initialized = False
        time_deviation_threshold = 1000  # Aumentado para evitar reconexiones innecesarias (1000ms)
        
        # Inicializar el cliente de Binance con reintentos
        while retry_count < max_retries:
            try:
                client = await initialize_binance_client()
                connection_status, new_client = await verify_connection(client)
                if connection_status:
                    if new_client and new_client != client:
                        await cleanup_client(client)
                        client = new_client
                    websocket_status = "Connected"
                    error_count = 0
                    break
                else:
                    raise Exception("Failed initial connection verification")
            except Exception as e:
                retry_count += 1
                wait_time = min(30, 2 ** retry_count)
                if retry_count >= max_retries:
                    logger.error("Se alcanzó el número máximo de reintentos al conectar con Binance")
                    raise
                logger.warning(f"Reintento {retry_count}/{max_retries} de conexión con Binance. Esperando {wait_time} segundos...")
                await asyncio.sleep(wait_time)
        if not client:
            raise Exception("No se pudo establecer conexión con Binance")
                
        logger.info("Conexión con Binance establecida correctamente")
        
        # Inicializar el bot de gestión de riesgos
        try:
            bot = RiskManagementBot()
            logger.info("Bot de trading iniciado correctamente")
            
            # Inicializar la interfaz de usuario
            ui = TerminalUI()
            ui.start()
            
            # Obtener información del símbolo
            symbol_info = await client.futures_exchange_info()
            symbol_info = next((s for s in symbol_info['symbols'] if s['symbol'] == CONFIG['SYMBOL']), {})
            
            # Inicializar indicadores ATR
            logger.info("Inicializando indicadores ATR...")
            if not await bot.initialize_indicators(client):
                raise Exception("Error initializing ATR indicators")
            logger.info("Indicadores ATR inicializados correctamente")
            
        except AttributeError as e:
            logger.error(f"Error al inicializar el bot: {e}")
            raise Exception(f"Error al inicializar el bot: {e}")
        except Exception as e:
            logger.error(f"Error al inicializar el bot: {e}")
            raise
        
        # Bucle principal del bot
        while True:
            try:
                # Verificar estado de la conexión usando la función de helpers
                current_time = time.time()
                if current_time - last_ping_time >= ping_interval:
                    connection_status, new_client = await verify_connection(client)
                    if not connection_status:
                        websocket_status = "Disconnected"
                        logger.warning("Conexión perdida con Binance, intentando reconectar...")
                        # Si verify_connection ya limpió el cliente, no necesitamos hacerlo de nuevo
                        if client and not new_client:
                            await cleanup_client(client)
                        # Si verify_connection ya creó un nuevo cliente, usarlo
                        if new_client:
                            client = new_client
                        else:
                            client = await initialize_binance_client()
                            connection_status, new_client = await verify_connection(client)
                            if not connection_status:
                                raise Exception("Failed to reconnect after connection loss")
                        websocket_status = "Connected"
                        logger.info("Reconexión exitosa con Binance")
                    # Si verify_connection devolvió un nuevo cliente, actualizarlo
                    elif new_client and new_client != client:
                        await cleanup_client(client)
                        client = new_client
                        logger.info("Cliente actualizado después de verificación de conexión")
                    last_ping_time = current_time

                # Obtener balance
                balance = await get_usdt_balance(client)
                if balance is None:
                    raise Exception("Error al obtener balance")
                
                # Actualizar indicadores ATR
                await bot.update_indicators(client)
                
                # Inicializar indicadores si aún no se ha hecho
                if not indicators_initialized:
                    logger.info("Inicializando indicadores ATR...")
                    try:
                        success = await bot.initialize_indicators(client)
                        if not success:
                            logger.error("Error al inicializar los indicadores ATR")
                            raise Exception("Failed to initialize ATR indicators")
                        indicators_initialized = True
                        logger.info("Indicadores ATR inicializados correctamente")
                    except AttributeError as e:
                        logger.error(f"Error en initialize_indicators: {e}")
                        raise Exception("Failed to initialize ATR indicators")
                    except Exception as e:
                        logger.error(f"Error al inicializar los indicadores ATR: {e}")
                        raise Exception("Failed to initialize ATR indicators")

                # Gestionar posiciones manuales
                await bot.manage_manual_position(client)
                # Actualizar indicadores con manejo de errores
                try:
                    await bot.update_indicators(client)
                except Exception as e:
                    logger.error(f"Error al actualizar indicadores: {e}")
                    indicators_initialized = False  # Forzar reinicialización en la siguiente iteración
                
                # Actualizar la interfaz
                if ui:
                    ui.update(bot, balance, websocket_status, bot.notifications, symbol_info, CONFIG)
                
                # Resetear contadores de error si todo está bien
                error_count = 0
                last_error_time = 0
                
                # Esperar antes de la siguiente iteración
                await asyncio.sleep(CONFIG['POSITION_CHECK_INTERVAL'])
                
            except asyncio.CancelledError:
                logger.info("Tarea principal cancelada, iniciando cierre ordenado")
                break
            except BinanceAPIException as e:
                current_time = time.time()
                error_count += 1
                
                # Manejar errores específicos
                if 'IP has been auto-banned' in str(e):
                    logger.warning("IP bloqueada temporalmente, esperando 60 segundos...")
                    websocket_status = "Disconnected"
                    await asyncio.sleep(60)
                    # Intentar reconectar después del tiempo de espera
                    client = await initialize_binance_client()
                    connection_status, new_client = await verify_connection(client)
                    if not connection_status:
                        raise Exception("Failed to reconnect after IP ban")
                    if new_client and new_client != client:
                        await cleanup_client(client)
                        client = new_client
                    websocket_status = "Connected"
                elif 'Invalid API-key' in str(e):
                    logger.error("Error: Verifique sus credenciales de API")
                    break
                else:
                    wait_time = min(30, 2 ** error_count)
                    logger.error(f"Error de API de Binance: {e}. Esperando {wait_time} segundos...")
                    websocket_status = "Disconnected"
                    await asyncio.sleep(wait_time)
                    # Intentar reconectar después del backoff
                    client = await initialize_binance_client()
                    connection_status, new_client = await verify_connection(client)
                    if not connection_status:
                        logger.error("Fallo la reconexión tras error de API. Forzando reconexión...")
                        await cleanup_client(client)
                        client = await initialize_binance_client()
                        connection_status, new_client = await verify_connection(client)
                        if not connection_status:
                            raise Exception("Failed to reconnect after forced reconnection attempt")
                    # Check average time deviation and force reconexión si es demasiado alta
                    time_diff = await synchronize_time(client)
                    if abs(time_diff) > 1000:  # Umbral de desviación de tiempo aumentado a 1000ms
                        logger.error(f"Desviación promedio persistentemente alta: {abs(time_diff):.2f}ms. Forzando reconexión...")
                        await cleanup_client(client)
                        client = await initialize_binance_client()
                        connection_status, new_client = await verify_connection(client)
                        if not connection_status:
                            raise Exception("Failed to reconnect after forced reconnection attempt due to high time deviation")
                    if new_client and new_client != client:
                        await cleanup_client(client)
                        client = new_client
                    websocket_status = "Connected"
                
                # Verificar frecuencia de errores
                if current_time - last_error_time < 60 and error_count > 5:
                    logger.error("Demasiados errores en poco tiempo, deteniendo el bot...")
                    break
                
                last_error_time = current_time
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error en el bucle principal: {e}")
                
                # Implementar backoff exponencial para errores generales
                wait_time = min(30, 2 ** error_count)
                await asyncio.sleep(wait_time)
                
                # Verificar si hay demasiados errores consecutivos
                if error_count > 10:
                    logger.error("Demasiados errores consecutivos, deteniendo el bot...")
                    break
                    
    except Exception as e:
        logger.error(f"Error crítico que requiere atención: {str(e)}")
        raise
    finally:
        if ui:
            ui.stop()
        if client:
            await shutdown(client)
        # Removed session closing as it's no longer created

if __name__ == "__main__":
    # Configurar el SelectorEventLoop para Windows
    if asyncio.get_event_loop_policy()._loop_factory is asyncio.ProactorEventLoop:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot detenido por el usuario")
    except Exception as e:
        logger.error(f"Error fatal: {str(e)}")
        raise