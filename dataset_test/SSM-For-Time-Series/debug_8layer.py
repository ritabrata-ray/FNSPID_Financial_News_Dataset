import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tst import Transformer
from tqdm import tqdm
import glob
from tst.model_based_portfolio import *
from run_8layer import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, d_input=None, mode=None, num_csvs=None, N=None):
    """
    Load a previously trained model from a saved state dictionary.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model state dictionary.
    d_input : int
        Model input dimension.
    mode : str, optional
        Mode used in the model name if path needs to be constructed.
    num_csvs : int, optional
        Number of CSVs used in the model name if path needs to be constructed.
    N : int, optional
        Number of layers used in the model name if path needs to be constructed.
    
    Returns
    -------
    model : Transformer
        The loaded model.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If model_path is not provided directly, construct it
    if model_path is None and mode is not None and num_csvs is not None and N is not None:
        model_path = f'model_saved/{mode}_{num_csvs}_{N}layers_ssm.pt'
    
    # Set up model parameters (same as in training)
    d_model = 32  # Latent dim
    d_output = 1  # Prediction length
    q = 8  # Query size
    v = 8  # Value size
    h = 8  # Number of heads
    attention_size = 512  # Attention window size
    dropout = 0.1  # Dropout rate
    chunk_mode = None
    
    # Create model with the same architecture
    model = Transformer(d_input, d_model, d_output, q, v, h, N, 
                      attention_size=attention_size, 
                      dropout=dropout, 
                      chunk_mode=chunk_mode, 
                      pe=None).to(device)
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode
    #model.eval()
    
    print(f"Loaded model from {model_path} onto {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    return model


#model = load_model('model_saved/Nonsentiment_25_8layers_ssm.pt')

# print("I am the edited file.")



def sentiment_predict_(csv_data,symbol, num_csvs, pred_flag, pred_names): # it loads a trained SSM model and evaluates it while also incorporating the FNSPID's sentinment information.
  mode = 'Sentiment'
  d_input = 4   # 'Volume','Open', 'Close', 'Scaled_sentiment'
  # Preparing the data for the model
  # Selecting relevant columns: 'Volume', 'Open', 'Close', and 'Scaled_sentiment'
  data = csv_data[['Volume', 'Open', 'Open','Scaled_sentiment']].values
  dataloader_train, dataloader_test, scaler_X, scaler_Y = data_processor(data, 4)
  d_output = 1 # prediction length be 3, this is confirmed
  d_model = 32 # Lattent dim
  q = 8 # Query size
  v = 8 # Value size
  h = 8 # Number of heads
  N = 8 # Number of encoder and decoder to stack

  attention_size = 512 # Attention window size
  dropout = 0.1 # Dropout rate
  pe = 'regular' # Positional encoding
  chunk_mode = None
  # Creating sequences

  # Creating the model
  model = Transformer(4, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=None).to(device)
  model.load_state_dict(torch.load('model_saved/Sentiment_25_8layers_ssm.pt'))
#   model = train_model(dataloader_train, pred_flag, symbol ,num_csvs, mode, d_input)
  return eval_model(model, dataloader_test, symbol, mode, num_csvs, scaler_X, scaler_Y), scaler_X, scaler_Y, dataloader_train

def nonsentiment_predict_(csv_data,symbol, num_csvs, pred_flag, pred_names): #It loads and evaluates without considering the news sentiment data, just pure numeric stock prices.
  mode = 'Nonsentiment'
  d_input = 3   # 'Volume','Open', 'Close', 'Scaled_sentiment'
  # Preparing the data for the model
  # Selecting relevant columns: 'Volume', 'Open', 'Close', and 'Scaled_sentiment'
  data = csv_data[['Volume', 'Open', 'Open']].values
  dataloader_train, dataloader_test, scaler_X, scaler_Y = data_processor(data, 3)
  d_output = 1 # prediction length be 3, this is confirmed
  d_model = 32 # Lattent dim
  q = 8 # Query size
  v = 8 # Value size
  h = 8 # Number of heads
  N = 8 # Number of encoder and decoder to stack

  attention_size = 512 # Attention window size
  dropout = 0.1 # Dropout rate
  pe = 'regular' # Positional encoding
  chunk_mode = None
  # Creating sequences

  # Creating the model
  model = Transformer(3, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=None).to(device) #even though it loads model from transformer.py the ssm model is written here in the transformer.py file. We apologize for the confusing file names.
  model.load_state_dict(torch.load('model_saved/Nonsentiment_25_8layers_ssm.pt'))
#   model = train_model(dataloader_train, pred_flag, symbol ,num_csvs, mode, d_input)
  return eval_model(model, dataloader_test, symbol, mode, num_csvs, scaler_X, scaler_Y), scaler_X, scaler_Y, dataloader_train
  return None, None

# We can choose to do evaluation on 1, 5, 25, or 50 stock(s). We can choose either of the names_x variable for that purpose.
# Test of 5 
# names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv','WMT.csv']
names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv','WMT.csv']

# names_5 = ['KO.csv']

# Test of 25 
names_25 = [
   'AAPL.csv', 'ABBV.csv','BABA.csv', 'BRK-B.csv',
            'bhp.csv', 'C.csv', 'COST.csv', 'CVX.csv','DIS.csv', 'GE.csv',
         'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv','QQQ.csv', 'SBUX.csv', 'T.csv', 'TSLA.csv', 'WFC.csv', 'gsk.csv',
         'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv'] # 'AMZN.csv'
# Tes of 50
names_50 = [
   'aal.csv', 'AAPL.csv', 'ABBV.csv', 'amgn.csv','BABA.csv', 'bhp.csv','biib.csv', 'bidu.csv', 'BRK-B.csv','C.csv', 'cat.csv', 'cmcsa.csv', 
   'cmg.csv', 'cop.csv', 'COST.csv', 'crm.csv', 'CVX.csv', 'DIS.csv', 'ebay.csv','GE.csv','gild.csv', 'gld.csv', 'gsk.csv', 'INTC.csv',
     'mrk.csv', 'MSFT.csv', 'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv', 'pep.csv', 'pypl.csv', 'qcom.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv',
      'tgt.csv', 'tm.csv', 'TSLA.csv', 'uso.csv', 'v.csv', 'WFC.csv', 'xlf.csv','KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv', ] #'AMZN.csv' 'dal.csv',



names_1 = ['GOOG.csv']
# names_1 = ['fakedata.csv',]
# pred_names = ['fakedata']


names = names_25
pred_names = names
# num_stocks = 1
num_stocks = len(names)
# num_stocks = 50
# num_stocks = 50



if_pred = True
  # for sentiment_type in sentiment_types:
trues_n = []
predicted_n = []
historicals_n = []
dir_acc_n = []
corr_n = []
for name in names:
      # Checking if GPU is available
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print('device = ',device)

      csv_data = read_csv_case_insensitive(os.path.join("data", name))
      symbol_name = name.split('.')[0]
      print(symbol_name, "=============================")
      # sentiment_predict(csv_data, symbol_name, num_stocks, if_pred, pred_names)
      y, scaler_X, scaler_Y, dataloader_train = nonsentiment_predict_(csv_data, symbol_name, num_stocks, True, names)
      y_true = scaler_Y.inverse_transform(y[0])
      y_pred = scaler_Y.inverse_transform(y[1])
      # breakpoint()
      trues_n.append(y_true)
      predicted_n.append(y_pred)
      historicals_n.append(scaler_Y.inverse_transform(dataloader_train[1].squeeze(0).cpu().numpy()))
      print(y[2])
      dir_acc_n.append(y[2]["dir_acc"])
      corr_n.append(y[2]["R2"])
trues_s = []
predicted_s = []
historicals_s = []
dir_acc_s = []
corr_s = []
for name in names:
      # Checking if GPU is available
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print('device = ',device)

      csv_data = read_csv_case_insensitive(os.path.join("data", name))
      symbol_name = name.split('.')[0]
      print(symbol_name, "=============================")
      # sentiment_predict(csv_data, symbol_name, num_stocks, if_pred, pred_names)
      y, scaler_X, scaler_Y, dataloader_train = sentiment_predict_(csv_data, symbol_name, num_stocks, True, names)
      y_true = scaler_Y.inverse_transform(y[0])
      y_pred = scaler_Y.inverse_transform(y[1])
      # breakpoint()
      trues_s.append(y_true)
      predicted_s.append(y_pred)
      historicals_s.append(scaler_Y.inverse_transform(dataloader_train[1].squeeze(0).cpu().numpy()))
      print(y[2])
      dir_acc_s.append(y[2]["dir_acc"])
      corr_s.append(y[2]["R2"])


def plot_lines(*ys):
    """
    Takes in one or more lists (or arrays) and plots them as separate line plots.
    The x-axis is the index of each list.

    Args:
        *ys: Any number of lists or arrays of equal or unequal length
    """
    for i, y in enumerate(ys):
        plt.plot(range(len(y)), y, label=f"Line {i+1}")
    
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Line Plot")
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_and_portfolio(historicals, true, predicted, min_days=200): # this function obtains the final accummulated wealth for the ONS baseline to get the numer reported in our final report.
    new_predicted = []
    new_true = []
    new_historicals = []

    for p, t, h in zip(predicted, true, historicals):
        if p.shape[0] >= min_days:
            new_predicted.append(p[:min_days])
            new_true.append(t[:min_days])
            new_historicals.append(h[:min_days])
    # result = portfolio_final_wealth(np.stack(new_historicals,axis=0).squeeze(2),np.stack(new_true, axis=0).squeeze(2),np.stack(new_predicted, axis=0).squeeze(2))
    result = online_newton_step_portfolio(np.stack(new_true,axis=0).squeeze(2))
    # print(result["sharpe_ratio"], result["final_wealth"])
    
    
# plot_lines(corr_n, corr_s)
# plot_lines(dir_acc_n, dir_acc_s)
# prepare_and_portfolio(historicals_n, trues_n, predicted_n)
# prepare_and_portfolio(historicals_s, trues_s, predicted_s)
prepare_and_portfolio(historicals_s, trues_s, predicted_s)

#     print(trues[0].shape)
# min_days = min(a.shape[0] for a in trues)
# trues = np.array([a[:min_days].squeeze() for a in trues])
# min_days = min(a.shape[0] for a in predicted)
# predicted = np.array([a[:min_days].squeeze() for a in predicted])
# min_days = min(a.shape[0] for a in historicals)
# historicals = np.array([a[:min_days].squeeze() for a in historicals])
# breakpoint()
# result = portfolio_final_wealth(historicals,trues,predicted)
# print(result["sharpe_ratio"], result["final_wealth"])
