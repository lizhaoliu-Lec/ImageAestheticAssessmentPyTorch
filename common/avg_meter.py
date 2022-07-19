class AvgMeter:
    def __init__(self):
        self.sum = 0
        self.n = 0
        self.avg = 0
        self.cur = 0

    def add(self, val, num):
        self.sum += val * num
        self.n += num
        self.avg = self.sum / self.n
        self.cur = val

    def reset(self):
        self.sum = 0
        self.n = 0
        self.avg = 0
        self.cur = 0
