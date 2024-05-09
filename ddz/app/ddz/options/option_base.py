import argparse


class OptionBase():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self, aws_env):
        print("aws_env:", aws_env)

        self.aws_env = aws_env
        self.initialized = True

    def parse(self, aws_env):
        if not self.initialized:
            self.initialize(aws_env)
        self.opt = self.parser.parse_args()
        print("parse options:")
        for k, v in vars(self.opt).items():
            print("option {}: {}".format(k, v))
        return self.opt

    def params(self):
        if not self.initialized:
            self.parse()
        return self.opt

