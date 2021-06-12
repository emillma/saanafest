from threading import Lock


class SoundSlot:
    def __init__(self, slot):
        self._lock = Lock()
        self._slot = slot

    def set(self, item):
        with self._lock:
            self._slot[:] = item

    def get(self):
        with self._lock:
            return self._slot.copy()
