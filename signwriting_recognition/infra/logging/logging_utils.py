from logging.handlers import RotatingFileHandler
from datetime import datetime
import logging, os, sys


def setup_logging(
    experiment_path: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """
    Cria/atualiza root-logger com:
      • StreamHandler (console)
      • RotatingFileHandler (<experiment_path>/logs/<timestamp>.log) se caminho fornecido
    """
    
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(console)

    if experiment_path:
        log_dir = os.path.join(experiment_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path  = os.path.join(log_dir, f"{timestamp}.log")

        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(file_handler)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("grpc").setLevel(logging.ERROR)