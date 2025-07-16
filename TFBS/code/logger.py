import os
import logging


class CustomLogger:
    def __init__(self, name: str, log_directory: str = '.', log_file: str = 'app.log', level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._configure_handlers(log_directory, log_file, level)

    def _configure_handlers(self, log_directory: str, log_file: str, level: int):
        # clear existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        os.makedirs(log_directory, exist_ok=True)

        log_path = os.path.join(log_directory, log_file)

        # create a file handler that writes to the specified log file
        self.file_handler = logging.FileHandler(log_path)
        self.file_handler.setLevel(level)

        # create a console (stream) handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level)

        # define a formatter and add it to both handlers
        console_formatter_without_time = logging.Formatter('%(message)s')

        # add both handlers to the logger
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

        # set console handler NOT to include time
        self.console_handler.setFormatter(console_formatter_without_time)
        self.file_handler.setFormatter(console_formatter_without_time)

    def get_logger(self):
        return self.logger

    def log_message(self, message: str, use_time: bool = False):
        # switch the console handler formatter based on the flag
        if use_time:
            self.console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        else:
            self.console_handler.setFormatter(logging.Formatter('%(message)s'))
            self.file_handler.setFormatter(logging.Formatter('%(message)s'))

        # log the message
        self.logger.info(message)