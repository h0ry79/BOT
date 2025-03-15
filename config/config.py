"""Módulo de configuración para el bot de trading.

Este módulo contiene todas las configuraciones necesarias para el funcionamiento del bot,
incluidos los parámetros de trading, conexión y gestión de riesgos.
"""

import os
import logging

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.info("python-dotenv no está instalado. Se usarán solo variables de entorno del sistema.")

# Intentar obtener las credenciales de las variables de entorno del sistema primero
api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
api_secret = os.environ.get('BINANCE_TESTNET_API_SECRET')

# Si no se encuentran en las variables del sistema, intentar obtener del archivo .env
if not api_key:
    api_key = os.getenv('BINANCE_TESTNET_API_KEY')
if not api_secret:
    api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')

# Verificar variables de entorno críticas
if not api_key or api_key == 'your_testnet_api_key_here':
    raise ValueError(
        "Error: BINANCE_TESTNET_API_KEY no está configurada correctamente.\n"
        "Configure la variable de entorno BINANCE_TESTNET_API_KEY o añádala al archivo .env"
    )

if not api_secret or api_secret == 'your_testnet_api_secret_here':
    raise ValueError(
        "Error: BINANCE_TESTNET_API_SECRET no está configurada correctamente.\n"
        "Configure la variable de entorno BINANCE_TESTNET_API_SECRET o añádala al archivo .env"
    )

# Configuration settings for the trading bot

CONFIG = {
    # API Credentials
    'API_KEY': api_key,
    'API_SECRET': api_secret,
    
    # Binance Client Settings
    'USE_TESTNET': True,  # Enable testnet mode
    'BASE_URL': 'https://testnet.binance.vision',  # Base URL for REST API
    'TESTNET_API_URL': 'https://testnet.binance.vision/api',  # Updated Testnet API URL
    'TESTNET_STREAM_URL': 'wss://testnet.binance.vision/ws',  # Testnet WebSocket URL
    'TESTNET_FUTURES_URL': 'https://testnet.binancefuture.com',  # Testnet Futures API URL
    'TESTNET_FUTURES_STREAM': 'wss://stream.binancefuture.com/ws',  # Testnet Futures WebSocket URL
    'TESTNET': True,  # Additional flag required by python-binance
    
    # Trading Parameters
    'SYMBOL': 'BTCUSDT',  # Trading pair
    'QUANTITY_PRECISION': 3,  # Decimal precision for quantity
    'FEE_RATE_MAKER': 0.0002,  # Maker fee rate (0.02%)
    'FEE_RATE_TAKER': 0.0004,  # Taker fee rate (0.04%)
    
    # Risk Management Parameters
    'MIN_BALANCE': 10.0,  # Minimum balance required to trade
    'OPTIMAL_BALANCE': 1000.0,  # Balance threshold for risk scaling
    'MAX_LEVERAGE': 20,  # Maximum allowed leverage
    'RISK_PER_TRADE': 0.02,  # Maximum risk per trade (2% of balance)
    'MIN_RISK_PER_TRADE': 0.005,  # Minimum risk per trade (0.5%)
    'MAX_RISK_PER_TRADE': 0.02,  # Maximum risk per trade (2%)
    'MIN_STOP_DISTANCE': 0.003,  # Minimum stop distance (0.3%)
    'STOP_LOSS_ATR_FACTOR': 2.0,  # Stop loss distance in ATR units
    'MAX_POSITION_SIZE_RATIO': 0.95,  # Maximum position size as a ratio of account balance
    'MIN_POSITION_VALUE': 10.0,  # Minimum position value in USDT
    'VOLATILITY_FACTOR': 1.5,  # Base volatility adjustment factor
    'MAX_VOLATILITY_RATIO': 0.03,  # Maximum acceptable volatility ratio
    'POSITION_CHECK_INTERVAL': 5.0,  # Interval for checking positions in seconds
    'RECV_WINDOW': 5000,  # Receive window for API requests in milliseconds
    
    # Technical Indicators
    'ATR_PERIOD_LONG': 14,  # Period for long-term ATR
    'ATR_PERIOD_SHORT': 5,  # Period for short-term ATR
    'MAX_PRICE_AGE': 5,  # Maximum age of price data in seconds
    'LONG_TIMEFRAME': '1h',  # Timeframe for long-term indicators
    'SHORT_TIMEFRAME': '5m',  # Timeframe for short-term indicators
    'TIME_SYNC_THRESHOLD': 1000,  # Maximum allowed time difference in milliseconds
    'SYNC_CHECK_INTERVAL': 60,  # Time between sync checks in seconds
    'MAX_SYNC_RETRIES': 3,  # Maximum number of sync attempts
    
    # Take Profit Configuration
    'TAKE_PROFIT_LEVELS': [1.5, 2.0, 2.5],  # Multiple take profit levels (in ATR units)
    'TP_MULTIPLIERS': [0.4, 0.3, 0.3],  # Position size distribution for each TP level (must sum to 1.0)
    
    # Logging and Notifications
    'LOG_LEVEL': 'INFO',  # Logging level
    'MAX_NOTIFICATIONS': 100,  # Maximum number of notifications to keep
}

# Validación de configuración
def validate_config():
    """Valida la configuración del bot."""
    try:
        required_fields = [
            'API_KEY', 
            'API_SECRET',
            'SYMBOL',
            'TAKE_PROFIT_LEVELS',
            'TP_MULTIPLIERS',
            'TESTNET_API_URL',
            'USE_TESTNET',
            'LONG_TIMEFRAME',  # Required for ATR calculation
            'SHORT_TIMEFRAME'  # Required for ATR calculation
        ]
        
        # Verificar campos requeridos
        for field in required_fields:
            if field not in CONFIG:
                raise ValueError(f"Campo requerido faltante en la configuración: {field}")

        # Validar TAKE_PROFIT_LEVELS
        if not isinstance(CONFIG['TAKE_PROFIT_LEVELS'], list):
            raise ValueError("TAKE_PROFIT_LEVELS debe ser una lista")
        if not all(isinstance(level, (int, float)) and level > 0 for level in CONFIG['TAKE_PROFIT_LEVELS']):
            raise ValueError("TAKE_PROFIT_LEVELS debe contener solo números positivos")

        # Validar TP_MULTIPLIERS
        if not isinstance(CONFIG['TP_MULTIPLIERS'], list):
            raise ValueError("TP_MULTIPLIERS debe ser una lista")
        if not all(isinstance(mult, (int, float)) and mult > 0 for mult in CONFIG['TP_MULTIPLIERS']):
            raise ValueError("TP_MULTIPLIERS debe contener solo números positivos")
        
        # Nueva validación: suma de TP_MULTIPLIERS debe ser 1.0
        if abs(sum(CONFIG['TP_MULTIPLIERS']) - 1.0) > 0.0001:
            raise ValueError("La suma de TP_MULTIPLIERS debe ser igual a 1.0")
            
        # Validar que TP_MULTIPLIERS y TAKE_PROFIT_LEVELS tengan la misma longitud
        if len(CONFIG['TP_MULTIPLIERS']) != len(CONFIG['TAKE_PROFIT_LEVELS']):
            raise ValueError("TP_MULTIPLIERS y TAKE_PROFIT_LEVELS deben tener la misma cantidad de elementos")

        if CONFIG['RISK_PER_TRADE'] <= 0 or CONFIG['RISK_PER_TRADE'] > 1:
            raise ValueError("Error: RISK_PER_TRADE debe estar entre 0 y 1")

        if not all(level > 0 for level in CONFIG['TAKE_PROFIT_LEVELS']):
            raise ValueError("Error: Todos los niveles de take profit deben ser mayores que 0")

        if CONFIG['POSITION_CHECK_INTERVAL'] < 1.0:
            raise ValueError("Error: POSITION_CHECK_INTERVAL debe ser mayor o igual a 1 segundo")
            
        if CONFIG['RECV_WINDOW'] < 5000 or CONFIG['RECV_WINDOW'] > 60000:
            raise ValueError("Error: RECV_WINDOW debe estar entre 5000 y 60000 milisegundos")
            
        if CONFIG['MAX_LEVERAGE'] <= 0 or CONFIG['MAX_LEVERAGE'] > 125:
            raise ValueError("Error: MAX_LEVERAGE debe estar entre 1 y 125")
            
        if not isinstance(CONFIG['SYMBOL'], str) or len(CONFIG['SYMBOL']) < 5:
            raise ValueError("Error: SYMBOL debe ser una cadena válida de al menos 5 caracteres")
            
        # Add testnet validation
        if CONFIG['USE_TESTNET']:
            testnet_fields = ['TESTNET_API_URL', 'TESTNET_STREAM_URL', 'TESTNET_FUTURES_URL', 'TESTNET_FUTURES_STREAM']
            for field in testnet_fields:
                if not CONFIG.get(field):
                    raise ValueError(f"Campo requerido faltante para testnet: {field}")
                if not isinstance(CONFIG[field], str):
                    raise ValueError(f"{field} debe ser una URL válida")
                if not CONFIG[field].startswith(('http://', 'https://', 'ws://', 'wss://')):
                    raise ValueError(f"{field} debe comenzar con http://, https://, ws:// o wss://")

        logger.info("Configuración validada correctamente")
        return True
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado durante la validación: {e}")
        raise