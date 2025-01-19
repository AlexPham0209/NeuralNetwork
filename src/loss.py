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
        return super().loss(actual, expected)
    
    def derivative(self, actual, expected):
        return super().derivative(actual, expected)
    
    def __repr__(self):
        return "cross_entropy"
    
losses = {
    "mean_squared" : MeanSquaredError(),
    "cross_entropy" : CrossEntropy(),
}

def create_loss(type):
    return losses[type]