import pyautogui
import time

# Texto a pegar
texto = "Conduct a comprehensive and detailed analysis of the risk management bot's codebase, simulating its operational flow across a wide variety of market conditions and operational scenarios. Following this deep analysis, select and execute a single, high-impact modification: either rectify a critical error that directly threatens the bot's stability, security, or profitability, or implement an indispensable optimization that demonstrably enhances its performance and efficiency. This modification must preserve the bot's core functionality and avoid introducing unintended side effects. Rigorously test the implemented change using the Binance testnet, ensuring its compatibility and effectiveness in a real market environment. Provide detailed documentation of the modification, including its rationale, implementation process, test results, and any additional considerations relevant to its correct operation and maintenance"

# Esperar 5 segundos para que abras la ventana del programa
print("Tienes 5 segundos para abrir la ventana...")
time.sleep(5)

# Bucle infinito para repetir el proceso
while True:
    # Enfocar el campo de texto y pegar el texto
    pyautogui.click(x=1672, y=964)  # Coordenadas del campo de texto
    time.sleep(5)
    pyautogui.write(texto)
    time.sleep(7)  # Esperar 5 segundos
    pyautogui.press('enter')  # Presionar ENTER

    # Esperar 300 segundos
    time.sleep(500)

    # Hacer 1 clic cada 5 segundos durante 120 segundos (24 clics)
    for _ in range(50):  # 120 segundos / 5 segundos por clic = 50 iteraciones
        pyautogui.click(x=1841, y=915)  # Coordenadas del botón
        time.sleep(5)  # Esperar 5 segundos entre clics

    # Esperar 20 segundos antes de repetir
    time.sleep(20)

print("Automatización completada.")  # No se alcanza con bucle infinito