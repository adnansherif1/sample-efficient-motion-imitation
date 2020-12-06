import numpy as np

class RamFusionDistr(object):
    def __init__(self, buf_size, subsample_ratio=0.5):
        self.buf_size = buf_size
        self.buffer = []
        self.subsample_ratio = subsample_ratio

    def add_paths(self, paths, subsample=True):
        if subsample:
            paths = paths[:int(len(paths)*self.subsample_ratio)]
        self.buffer.extend(paths)
        overflow = len(self.buffer)-self.buf_size
        while overflow > 0:
            #self.buffer = self.buffer[overflow:]
            N = len(self.buffer)
            probs = np.arange(N)+1
            probs = probs/float(np.sum(probs))
            pidx = np.random.choice(np.arange(N), p=probs)
            self.buffer.pop(pidx)
            overflow -= 1

    def sample_paths(self, n):
        if len(self.buffer) == 0:
            return []
        else:
            pidxs = np.random.randint(0, len(self.buffer), size=(n))
            return [self.buffer[pidx] for pidx in pidxs]