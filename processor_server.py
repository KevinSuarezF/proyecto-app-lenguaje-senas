"""
Servidor FastAPI independiente para procesamiento de frames ASL.
Se ejecuta automáticamente desde asl_app.py
"""
import sys
import time
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent))

# Importar después de configurar path
from asl_app.asl_app import process_frame_data

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    # Crear app FastAPI
    app = FastAPI()
    
    # Habilitar CORS de forma más explícita
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost",
            "http://127.0.0.1",
            "*"
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )
    
    # Endpoint CORS preflight
    @app.options("/{path:path}")
    async def preflight_handler(path: str):
        """Manejar solicitudes preflight CORS"""
        return {"message": "OK"}
    
    @app.post("/_process_frame")
    async def process_frame(request: Request):
        """Endpoint para procesar frames de video."""
        try:
            data = await request.json()
            frame_data = data.get("frame_data", "")
            result = process_frame_data(frame_data)
            return result
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    @app.on_event("startup")
    async def startup_event():
        """Se ejecuta cuando el servidor inicia."""
        print("\n" + "="*60)
        print("[OK] SERVIDOR FASTAPI INICIADO CORRECTAMENTE")
        print("="*60)
        print("[INFO] URL: http://127.0.0.1:8002")
        print("[INFO] Endpoint: POST http://127.0.0.1:8002/_process_frame")
        print("[OK] CORS habilitado para http://localhost:3000")
        print("[OK] Procesamiento de frames disponible")
        print("="*60 + "\n")
    
    if __name__ == "__main__":
        # Pequeño delay para que se estabilice
        time.sleep(1)
        
        print("Iniciando servidor FastAPI...")
        
        # Iniciar servidor
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8002,
            log_level="warning"
        )

except ImportError as e:
    print(f"Error: Falta instalar dependencias: {e}")
    print("Instala con: pip install fastapi uvicorn")
    sys.exit(1)
