
class Transition(object):
    def __init__(self, s=None, a=None, r=None, s_=None, done=False, **kwargs):
        super().__init__()
        self.s = s
        self.a = a
        self.s_next = s_
        self.r = r
        self.done = done
        self.info = {}.update(**kwargs)

    def print(self):
        print("s:{}, a:{}, r:{}, s_:{}, done:{}".format(self.s, self.a, self.r, self.s_next, self.done))