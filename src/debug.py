
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import optuna

class DynamicFeedforward(torch.nn.Module):
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

# Build neural network model
def build_model(params):    
    return DynamicFeedforward(input_size=params["input_size"], n_layers=params["n_layers"], n_units=params["n_units"], bias=True)
 
# Train and evaluate the accuarcy of neural network model
def train_and_evaluate(param, data, target_name, model, epochs, criterion, verbose, trial):
    
    # train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    # val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    y = data[target_name].values
    X = data.drop([target_name], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = torch.FloatTensor(y_train).unsqueeze(0).t()
    X_train = torch.FloatTensor(X_train)

    y_test = torch.FloatTensor(y_test).unsqueeze(0).t()
    X_test = torch.FloatTensor(X_test)

    optimizer = getattr(torch.optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

    loss_arr = []
    loss_values = []
    for i in tqdm(range(epochs), total=epochs, desc="Running backpropagation", disable=not verbose):
        # computer forward prediction
        # if first iter, use random init
        y_hat = model.forward(X_train)

        # computes the loss function
        loss = criterion(y_hat, y_train)
        loss_arr.append(loss)
        loss_values.append(loss.item())
        
        trial.report(loss.item(), i)

        # set all previous gradients to zero
        optimizer.zero_grad()

        # backpropagation
        # computes gradient of current tensor given loss and opt procedure
        loss.backward()
        # update parameters
        optimizer.step()

    loss_values_df = pd.DataFrame(loss_values, columns=["loss"])

    preds = []
    for val in X_test:
        y_hat = model.forward(val)
        preds.append(y_hat.item())
    preds = torch.FloatTensor(preds).unsqueeze(0).t()
    test_loss = criterion(preds, y_test)

    return test_loss
    
    return loss.item() 
  
def objective(data, target_name, epochs, criterion, verbose, trial):

     params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'n_units': trial.suggest_int("n_unit", 10, 100),
              'n_layers': trial.suggest_int("n_layers", 10, 10),
              'optimizer': trial.suggest_categorical("optimizer", ["SGD"]),
              'input_size': trial.suggest_int("input_size", data.drop([target_name], axis=1).shape[1], data.drop([target_name], axis=1).shape[1]),
              }
    
     model = build_model(params)
     loss = train_and_evaluate(params, data, target_name, model, epochs, criterion, verbose, trial)

     return loss


if __name__ == "__main__":
    import os

    target_name = "betas_dgp"
    target_path = os.path.join(os.getcwd(), "src", "data", "inputs", "simple_ar")
    epochs = 100
    criterion = torch.nn.MSELoss()
    verbose = False

    data = pd.read_csv(os.path.join(target_path, "betadgp_corrdgp_data.csv")).drop(["Var1", "Var2"], axis=1)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective(data, target_name, epochs, criterion, verbose, trial), n_trials=5)

    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    fim = 1