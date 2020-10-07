import numpy as np

class Foo:
    def __init__(self, x):
        self.x = x

    def add(self, y):
        return self.x + y

    def __str__(self):
        print("This is a string.")
