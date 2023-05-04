import logging
import pathlib

# set up logging first
logfile = pathlib.Path(f"./{__package__}.log")
try:
    logfile.replace(f"{logfile}.previous")
except FileNotFoundError:
    pass

file_handler = logging.FileHandler(
    filename=logfile,
)
logging.basicConfig(
    handlers=[file_handler, logging.StreamHandler()],
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
)


def run():
    from . import Fiumicino  # TODO fix modules misbehaving on import

    Fiumicino.run_app()


run()
