import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
import os
import matplotlib.pyplot as plt

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def plot_and_save_node_predictions(preds, trues, save_dir,args):
    os.makedirs(save_dir, exist_ok=True)
    num_nodes = preds.shape[1]
    time_steps = preds.shape[0]

    for node in range(num_nodes):
        plt.figure(figsize=(10, 4))
        plt.plot(range(time_steps), trues[:, node], label='True Values', color='blue')
        plt.plot(range(time_steps), preds[:, node], label='Predicted Values', color='red')
        plt.title(f'Node {node} Predictions vs True Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()

        # Save figure
        output_dir = save_dir + '/' + 'fig' + '/' + f'{args.model}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir + '/' + f"node-{args.pred_type}_fold-{args.fold}_pred_len{args.pred_len}_predictions.png"  # 保存路径
        plt.savefig(output_path)
        plt.close()

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true,args):
    eps = 2e-2
    true = true[:,-1,:]
    pred = pred[:,-1,:]
    MAPE_true = true.copy()
    MAPE_pred = pred.copy()
    MAPE_true[np.where(MAPE_true <= eps)] = np.abs(MAPE_true[np.where(MAPE_true <= eps)]) + eps
    MAPE_pred[np.where(MAPE_true <= eps)] = np.abs(MAPE_pred[np.where(MAPE_true <= eps)]) + eps

    mape = mean_absolute_percentage_error(MAPE_true, MAPE_pred)
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    rae = np.sum(abs(MAPE_pred - MAPE_true)) / np.sum(abs(np.mean(MAPE_true) - MAPE_true))
    print('MAPE: {}'.format(mape))
    print('MAE:{}'.format(mae))
    print('MSE:{}'.format(mse))
    print('RMSE:{}'.format(rmse))
    print(('RAE:{}'.format(rae)))
    output_list = [mse, rmse, mape, rae, mae]
    return output_list
