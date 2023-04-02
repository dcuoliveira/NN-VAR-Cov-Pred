import torch

class MLP(torch.nn.Module):
    def __init__(self, input_size, n_layers, n_units, bias):
        super().__init__()
        self.input_size = input_size
        self.n_units  = n_units
        self.n_layers = n_layers
        self.bias = bias
        
        self.input_layer = torch.nn.Linear(self.input_size, self.n_units, bias=self.bias)
        self.relu = torch.nn.ReLU()
        self.hidden_layer = torch.nn.Linear(self.n_units, self.n_units, bias=self.bias)
        self.output_layer = torch.nn.Linear(self.n_units, 1, bias=self.bias)
    def forward(self, x):
        x = self.input_layer(x)
        
        for l in range(self.n_layers):
            x = self.hidden_layer(x)
            x = self.relu(x)
                    
        output = self.output_layer(x)
        
        return output

class MLPWrapper():
    def __init__(self, 
                 learning_rate=None,
                 n_units=None,
                 n_layers=None,
                 optimizer=None,
                 input_size=None,
                 trial=None):

        self.model_name = "mlp"
        self.search_type = 'random'

        learning_rate = learning_rate if learning_rate is not None else trial.suggest_float("learning_rate", 1e-1, 1e-1)
        n_units = n_units if n_units is not None else trial.suggest_int("n_units", 500, 500)
        n_layers = n_layers if n_layers is not None else trial.suggest_int("n_layers", 500, 500)
        optimizer = optimizer if optimizer is not None else trial.suggest_categorical("optimizer", ["SGD"])
        input_size = trial.suggest_int("input_size", input_size, input_size)

        self.params = {
              'learning_rate': learning_rate,
              'n_units': n_units,
              'n_layers': n_layers,
              'optimizer': optimizer,
              'input_size': input_size,
              }
        self.epochs = 100

        self.ModelClass = MLP