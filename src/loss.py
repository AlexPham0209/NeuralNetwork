import cupy as cp

class Loss:
    def loss(self, actual, expected):
        return 0
    
    def derivative(self, actual, expected):
        return 0

    def __repr__(self):
        return ""
    
class MeanSquaredError(Loss):
    def loss(self, actual, expected):
        return ((expected - actual) ** 2).mean(1)
    
    def derivative(self, actual, expected):
        return actual - expected
    
    def __repr__(self):
        return "mean_squared"
    
class CrossEntropy(Loss):
    def loss(self, actual, expected):
        return -(expected * cp.log(actual) + (1 - expected) * cp.log(1 - actual)).mean(1)

    def derivative(self, actual, expected):
        EPSILON = 1e-8
        size = actual.shape[1]

        actual = actual + EPSILON
        expected = expected + EPSILON
        res = ((1 - expected) / (1 - actual) - expected / actual) / size
        return res

    def __repr__(self):
        return "cross_entropy"
    
losses = {
    "mean_squared" : MeanSquaredError(),
    "cross_entropy" : CrossEntropy(),
}

def create_loss(type):
    return losses[type]