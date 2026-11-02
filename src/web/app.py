"""FastAPI приложение для веб-интерфейса медицинской онтологии."""

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.config import logger
from src.web.routes import router as api_router

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(
    title="Медицинская Онтология | Web UI",
    description="Веб-интерфейс для просмотра анкет, их JSON структуры и интерактивного графа онтологии признаков.",
    version="1.0.0",
)

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение статических файлов (CSS, JS, Vis.js)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Подключение шаблонизатора Jinja2
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def index_view(request: Request):
    """Главная страница приложения."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Медицинская онтология и анкеты",
        },
    )


# Подключение всех API маршрутов
app.include_router(api_router)


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Запуск веб-сервера через uvicorn."""
    import uvicorn

    print(f"\n=======================================================")
    print(f"🚀 Веб-интерфейс запущен и доступен по адресу:")
    print(f"   👉 http://{host}:{port}")
    print(f"   👉 Swagger API документация: http://{host}:{port}/docs")
    print(f"=======================================================\n")
    uvicorn.run("src.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
