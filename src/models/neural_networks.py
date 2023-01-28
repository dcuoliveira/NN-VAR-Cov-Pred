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
    def __init__(self, input_size, trial):
        self.model_name = "mlp"
        self.search_type = 'random'
        self.params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'n_units': trial.suggest_int("n_unit", 10, 100),
              'n_layers': trial.suggest_int("n_layers", 10, 10),
              'optimizer': trial.suggest_categorical("optimizer", ["SGD"]),
              'input_size': trial.suggest_int("input_size", input_size, input_size),
              }
        self.epochs = 100

        self.ModelClass = MLP(input_size=self.params["input_size"],
                              n_layers=self.params["n_layers"],
                              n_units=self.params["n_units"],
                              bias=True)