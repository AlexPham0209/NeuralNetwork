class Loss:
    def loss(self, actual, expected):
        pass
    
    def derivative(self, actual, expected):
        pass

    def __repr__(self):
        return ""
    
class MeanSquaredError(Loss):
    def loss(self, actual, expected):
        return ((expected - actual) ** 2).mean(1)
    
    def derivative(self, actual, expected):
        return actual - expected
    
    def __repr__(self):
        return "MeanSquared"