
class ProgressBar:
    def __init__(self, pct=0):
        self._pct = pct
        self.validate_percentage(pct)

    def validate_percentage(self, pct):
        if not (0 <= pct <= 100):
            raise ValueError("Percentage must be between 0 and 100.")
        self._pct = pct

    def progress(self, pct):
        self.validate_percentage(pct)

    def get_progress(self):
        return self._pct

    def display(self):
        bar_length = 20
        filled_length = int(bar_length * self._pct // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        return f"[{bar}] {self._pct}%"