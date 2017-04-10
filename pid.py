class PID(object):
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.e_i = 0
        self.e_d = 0
        self.res = 0.0
    def compute(self, err, dt):
        if dt == 0:
            return self.res
        k_p, k_i, k_d = self.k_p, self.k_i, self.k_d
        self.e_i += err * dt
        self.res = k_p * err + \
                k_i * self.e_i + \
                k_d * (err-self.e_d)/dt
        self.e_d = err # remember last error
        return self.res
