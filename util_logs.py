from contextlib import ContextDecorator

@contextmanager
class logger(ContextDecorator):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
