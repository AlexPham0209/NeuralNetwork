import math


class Metric:
    def update(self, actual, expected):
        pass

    def compute(self):
        return 0
    
    def reset(self):
        pass

    def __str__(self):
        pass
    

class Accuracy(Metric):
    def __init__(self):
        self.correct = 0
        self.total = 0    
    
    def update(self, actual, expected):
        self.total += 1 
        actual = actual.argmax(axis=-1)
        expected = expected.argmax(axis=-1)
        
        if actual == expected:
            self.correct += 1

    def reset(self):
        self.correct = 0 
        self.total = 0

    def compute(self):
        return self.correct / self.total * 100
    
    def __str__(self):
        return f"Accuracy: {self.compute():.2f}%"
           
def test_accuracy():
    import cupy as cp
    metric = Accuracy()
    metric.update(cp.array([1, 0, 0, 0]), cp.array([1, 0, 0, 0]))
    metric.update(cp.array([1, 0, 0, 0]), cp.array([0, 1, 0, 0]))
    assert math.isclose(metric.compute(), 0.5)

