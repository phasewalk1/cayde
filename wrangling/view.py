import colorlog
from colorlog import ColoredFormatter


class Viewer:
    def __init__(self):
        # Set up the Cayde viewer
        self.view = colorlog.getLogger()
        # Set the level to debug so that all messages are logged
        self.view.setLevel(colorlog.DEBUG)
        # Set up the formatter for the Cayde viewer
        self.viewfmt = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
        # Set the stream handler
        self.stream_handler = colorlog.StreamHandler()
        self.stream_handler.setFormatter(self.viewfmt)
        self.view.addHandler(self.stream_handler)

    def info(self, msg):
        self.view.info(msg)

    def debug(self, msg):
        self.view.debug(msg)

    def warning(self, msg):
        self.view.warning(msg)

    def error(self, msg):
        self.view.error(msg)

    def critical(self, msg):
        self.view.critical(msg)
