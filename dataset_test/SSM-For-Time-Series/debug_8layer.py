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

print("I am the edited file.")



def sentiment_predict(csv_data,symbol, num_csvs, pred_flag, pred_names):
  mode = 'Sentiment'
  d_input = 4  # this one should be 4 assume it is 'Volume','Open', 'Close', 'Scaled_sentiment'
  # Selecting relevant columns: 'Volume', 'Open', 'Close', and 'Scaled_sentiment'
  data = csv_data[['Volume', 'Open', 'Close', 'Scaled_sentiment']].values
  dataloader_train, dataloader_test, scaler_X, scaler_Y = data_processor(data)
  model = train_model(dataloader_train, pred_flag, symbol ,num_csvs, mode, d_input)

  if pred_flag:
    if symbol in pred_names:
      eval_model(data, model, dataloader_test, symbol, mode, num_csvs, scaler_X, scaler_Y)

def nonsentiment_predict(csv_data,symbol, num_csvs, pred_flag, pred_names):
  mode = 'Nonsentiment'
  d_input = 3   # 'Volume','Open', 'Close', 'Scaled_sentiment'
  # Preparing the data for the model
  # Selecting relevant columns: 'Volume', 'Open', 'Close', and 'Scaled_sentiment'
  data = csv_data[['Volume', 'Open', 'Close']].values
  dataloader_train, dataloader_test, scaler_X, scaler_Y = data_processor(data)
  model = load_model('model_saved/Nonsentiment_25_8layers_ssm.pt')

  return eval_model(model, dataloader_test, symbol, mode, num_csvs, scaler_X, scaler_Y)
  return None, None


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


names = names_1
pred_names = names
# num_stocks = 1
num_stocks = len(names)
# num_stocks = 50
# num_stocks = 50



if_pred = True
  # for sentiment_type in sentiment_types:
trues = []
predicted = []
for name in names:
      # Checking if GPU is available
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print('device = ',device)

      csv_data = read_csv_case_insensitive(os.path.join("data", name))
      symbol_name = name.split('.')[0]
      print(symbol_name)
      # sentiment_predict(csv_data, symbol_name, num_stocks, if_pred, pred_names)
      y_true, y_pred = nonsentiment_predict(csv_data, symbol_name, num_stocks, True, names)
      trues.append(y_true)
      predicted.append(y_pred)
    # print(trues[0].shape)
min_days = min(a.shape[0] for a in trues)
trues = np.array([a[:min_days].squeeze() for a in trues])
min_days = min(a.shape[0] for a in predicted)
predicted = np.array([a[:min_days].squeeze() for a in predicted])

print(portfolio_final_wealth(trues,trues,predicted))