#!/usr/bin/env python3
"""Точка входа для запуска веб-интерфейса проекта Медицинской Онтологии."""

import argparse
import sys
from src.web.app import run_server

def main():
    parser = argparse.ArgumentParser(
        description="Запуск простого веб-интерфейса для просмотра анкет и онтологии признаков."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Хост для привязки веб-сервера (по умолчанию: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для веб-сервера (по умолчанию: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Автоматический перезапуск сервера при изменении исходных файлов (для разработки)",
    )

    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
