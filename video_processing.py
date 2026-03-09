# video_processing.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import asyncio
import base64
import os
import logging
import time
import signal
import sys
from threading import Lock
from typing import Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Processing API")

# Константы
CAMERA_INDEX = 0  # Индекс камеры
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
FRAME_DELAY = 1.0 / FPS  # ~0.033 секунды

# Глобальные переменные для управления камерой и потоками
camera_instance = None
camera_lock = Lock()
streaming_active = False
active_connections = []

def get_camera():
    """Получить или создать экземпляр камеры"""
    global camera_instance
    with camera_lock:
        if camera_instance is None:
            camera_instance = open_camera_safely()
        return camera_instance

def release_camera():
    """Освободить камеру"""
    global camera_instance
    with camera_lock:
        if camera_instance is not None:
            camera_instance.release()
            camera_instance = None
            logger.info("Камера остановлена и освобождена")

def open_camera_safely(index=CAMERA_INDEX):
    """Безопасное открытие камеры с проверкой и выбором бэкенда"""
    
    # Проверяем существование устройства на Linux
    if os.name == 'posix':  # Linux/Mac
        if not os.path.exists(f'/dev/video{index}'):
            logger.error(f"Устройство /dev/video{index} не найдено!")
            return None
    
    # Пробуем разные бэкенды
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if os.name == 'posix' else [cv2.CAP_ANY]
    
    for backend in backends:
        logger.info(f"Попытка открыть камеру {index} с бэкендом {backend}")
        cap = cv2.VideoCapture(index, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(index)
        
        if cap.isOpened():
            # Устанавливаем параметры
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FPS)
            
            # Проверяем, что камера действительно отдаёт кадры
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info(f"✓ Камера {index} успешно открыта")
                return cap
            else:
                logger.warning(f"Камера открыта, но кадры не читаются")
                cap.release()
    
    logger.error(f"Не удалось открыть камеру {index}")
    return None

def generate_frames():
    """Генератор кадров с веб-камеры (синхронная функция)"""
    global streaming_active
    camera = None
    face_cascade = None
    
    try:
        # Открываем камеру
        camera = get_camera()
        if camera is None:
            logger.error("Не удалось открыть камеру")
            return
        
        streaming_active = True
        
        # Загружаем классификатор лиц
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                logger.warning("Не удалось загрузить классификатор лиц")
                face_cascade = None
        else:
            logger.warning(f"Файл классификатора не найден: {cascade_path}")
            face_cascade = None
        
        while streaming_active:
            success, frame = camera.read()
            if not success:
                logger.warning("Не удалось прочитать кадр")
                time.sleep(FRAME_DELAY)
                continue
            
            # Применяем обработку (детекцию лиц)
            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Face', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Добавляем информацию на кадр
            faces_count = len(faces) if face_cascade else 0
            cv2.putText(frame, f'Faces: {faces_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Size: {frame.shape[1]}x{frame.shape[0]}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Кодируем кадр в JPEG с оптимальным качеством
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(FRAME_DELAY)
            
    except Exception as e:
        logger.error(f"Ошибка в generate_frames: {e}")
    finally:
        streaming_active = False
        logger.info("Генерация кадров остановлена")

@app.get("/video-feed")
async def video_feed():
    """
    Streaming видео с веб-камеры с обработкой OpenCV
    """
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/camera-info")
async def camera_info():
    """
    Информация о доступных камерах
    """
    available_cameras = []
    
    for i in range(5):  # Проверяем первые 5 индексов
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            available_cameras.append({
                "index": i,
                "width": int(width),
                "height": int(height),
                "fps": float(fps) if fps > 0 else 30,
                "device": f"/dev/video{i}" if os.name == 'posix' else f"Camera {i}"
            })
            cap.release()
    
    return {
        "available_cameras": available_cameras,
        "os": os.name,
        "opencv_version": cv2.__version__
    }

@app.get("/stop-camera")
async def stop_camera():
    """
    Остановить камеру и освободить ресурсы
    """
    global streaming_active
    streaming_active = False
    release_camera()
    return {
        "status": "success",
        "message": "Камера остановлена и ресурсы освобождены"
    }

@app.get("/camera-status")
async def camera_status():
    """
    Проверить статус камеры
    """
    global camera_instance, streaming_active
    
    status = {
        "camera_active": camera_instance is not None,
        "streaming_active": streaming_active,
        "camera_index": CAMERA_INDEX
    }
    
    if camera_instance is not None:
        status["is_opened"] = camera_instance.isOpened()
    
    return status

@app.post("/shutdown")
async def shutdown_server():
    """
    Завершить работу сервера и освободить все ресурсы
    """
    logger.info("Получен запрос на завершение работы")
    
    # Останавливаем камеру
    global streaming_active
    streaming_active = False
    release_camera()
    
    # Закрываем все WebSocket соединения
    for connection in active_connections:
        try:
            await connection.close()
        except:
            pass
    
    # Планируем завершение работы через 1 секунду
    async def shutdown_delay():
        await asyncio.sleep(1)
        logger.info("Завершение работы сервера...")
        os.kill(os.getpid(), signal.SIGINT)
    
    asyncio.create_task(shutdown_delay())
    
    return {
        "status": "success",
        "message": "Сервер завершает работу. Ресурсы освобождены."
    }

@app.get("/health")
async def health_check():
    """
    Проверка здоровья сервера
    """
    return {
        "status": "healthy",
        "camera_initialized": camera_instance is not None,
        "timestamp": time.time()
    }

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """
    WebSocket для потоковой передачи видео
    """
    await websocket.accept()
    logger.info("WebSocket соединение установлено")
    
    # Добавляем соединение в список активных
    active_connections.append(websocket)
    
    camera = None
    
    try:
        # Открываем камеру
        camera = get_camera()
        if camera is None:
            await websocket.send_json({"error": "Cannot open camera"})
            return
        
        while True:
            success, frame = camera.read()
            if not success:
                logger.warning("Не удалось прочитать кадр")
                await asyncio.sleep(FRAME_DELAY)
                continue
            
            # Применяем фильтры
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Конвертируем в base64 для отправки через WebSocket
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            _, buffer_edges = cv2.imencode('.jpg', edges_bgr, encode_params)
            edges_base64 = base64.b64encode(buffer_edges).decode('utf-8')
            
            await websocket.send_json({
                "original": f"data:image/jpeg;base64,{frame_base64}",
                "processed": f"data:image/jpeg;base64,{edges_base64}",
                "timestamp": cv2.getTickCount(),
                "fps": FPS
            })
            
            await asyncio.sleep(FRAME_DELAY)
            
    except WebSocketDisconnect:
        logger.info("WebSocket клиент отключился")
    except Exception as e:
        logger.error(f"Ошибка в websocket_video: {e}")
    finally:
        # Удаляем соединение из списка активных
        if websocket in active_connections:
            active_connections.remove(websocket)
        # Не освобождаем камеру здесь, так как она может использоваться другими подключениями

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка ресурсов при завершении"""
    global streaming_active
    streaming_active = False
    release_camera()
    logger.info("Приложение завершает работу. Все ресурсы освобождены.")


    