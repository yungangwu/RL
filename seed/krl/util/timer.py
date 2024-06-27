import time

ENABLE = False
class Timer:
    def __init__(self, name=None, text='{}: elapsed time {:0.4f} seconds', logger=print) -> None:
        self._start_time = None
        self.name = name
        self.text = text
        self.logger = logger

    def __enter__(self):
        if ENABLE:
            self.start()
        return self

    def __exit__(self, *exc_info):
        if ENABLE:
            self.stop()

    def start(self):
        if self._start_time is not None:
            raise RuntimeError('timer is running. use .stop() to stop it.')

        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is None:
            raise RuntimeError('timer is not running. use .start() to start it.')

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.logger:
            self.logger(self.text.format(self.name, elapsed_time))

        return elapsed_time