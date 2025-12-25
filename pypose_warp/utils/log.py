import logging
from rich.logging import RichHandler
from rich.console import Console

logging.basicConfig(
    level="INFO",
    format="PID %(process)d %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=Console())],
)


LogWriter = logging.getLogger("warp-pypose")
