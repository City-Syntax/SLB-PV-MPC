# Author: Kaifeng ZHU
'''
This file contains the main functions for prediction generation.
'''
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from collections import deque

# PyTorch for LSTM
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

def import_dataset(elec_df_path='elec_df_5min.csv', weather_df_path='weather_df_5min.csv'):
    '''Import the dataset from the given paths.'''
    elec_df = pd.read_csv(elec_df_path)
    weather_df = pd.read_csv(weather_df_path)
    return elec_df, weather_df

def clean_dataset(elec_df, weather_df, time_interval=''):
    '''Clean the dataset by parsing the datetime and sorting the dataframe. Add the time-based features.'''
    def _prepare_df(df, name):
        """
        Ensure dataframe has a 'datetime_utc' column in datetime type,
        is sorted by time, and (optionally) resampled.
        """
        # Work on a copy to avoid side-effects
        df = df.copy()

        # Handle different possibilities for where the datetime is stored
        if 'datetime_utc' in df.columns:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        elif df.index.name == 'datetime_utc':
            # Already the index – convert index to column
            df['datetime_utc'] = pd.to_datetime(df.index)
        else:
            raise KeyError(
                f"'datetime_utc' not found in {name}_df. "
                f"Columns: {list(df.columns)}, index name: {df.index.name}"
            )

        # Sort by datetime
        df.sort_values('datetime_utc', inplace=True)

        # Optional resampling
        if time_interval:
            df.set_index('datetime_utc', inplace=True)
            df = df.resample(time_interval).mean()
            df.reset_index(inplace=True)

        return df

    elec_df_prepared = _prepare_df(elec_df, 'elec')
    weather_df_prepared = _prepare_df(weather_df, 'weather')

    # Merge on timestamp
    merged = pd.merge(elec_df_prepared, weather_df_prepared, on='datetime_utc', how='inner')

    # Create time-based features
    merged['weekday'] = merged['datetime_utc'].dt.weekday
    merged['hour_of_day'] = merged['datetime_utc'].dt.hour

    return merged

def create_one_feature_lags(merged, feature, num_lags=1, ignore_nan=True):
    '''Create the lagged features for a feature.'''
    for k in range(1, num_lags + 1):
        merged[f'{feature}_lag_{k}'] = merged[feature].shift(k)
    if ignore_nan:
        merged = merged.dropna().reset_index(drop=True)
    return merged

def select_features(enable_weather_features=True, enable_without_wind_features=False, enable_time_features=True, enable_lags_features=True, lag_paras=[], num_lags=0):
    '''Select the features to use and return a list of features.'''
    if enable_weather_features:
        feature_cols_weather = ['dc', 'igc', 'sc', 'ta', 'ua']
        if enable_without_wind_features:
            feature_cols_weather = ['igc', 'ta', 'ua']
    else:
        feature_cols_weather = []
    if enable_time_features:
        feature_cols_time = ['weekday', 'hour_of_day']
    else:
        feature_cols_time = []
    if enable_lags_features:
        feature_cols_lags = []
        for lag_para in lag_paras:
            feature_cols_lags += [f'{lag_para}_lag_{k}' for k in range(1, num_lags+1)]
    else:
        feature_cols_lags = []
    X_cols = feature_cols_weather + feature_cols_time + feature_cols_lags
    return X_cols

def split_train_test_dataset(merged, X_cols, y_col, split_ratio=0.75):
    '''Split the dataset into train and test sets.'''
    merged = merged.sort_values('datetime_utc').reset_index(drop=True)
    split_index = int(len(merged) * split_ratio)

    X_train = merged.loc[:split_index-1, X_cols]
    y_train = merged.loc[:split_index-1, y_col]
    X_test = merged.loc[split_index:, X_cols]
    y_test = merged.loc[split_index:, y_col]
    return X_train, y_train, X_test, y_test


def split_train_test_dataset_time_series(merged, X_cols, y_col, n_splits, max_train_size, test_size):
    '''NOT USED.'''
    t_list = range(len(merged))
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size)
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    for train_index, test_index in tscv.split(t_list):
        X_train = merged.iloc[train_index][X_cols]
        y_train = merged.iloc[train_index][y_col]
        X_test = merged.iloc[test_index][X_cols]
        y_test = merged.iloc[test_index][y_col]
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
    return X_train_list, y_train_list, X_test_list, y_test_list

def split_train_test_dataset_sequential(merged, X_cols, y_col, train_size, test_size):
    '''Split the dataset into train and test sets sequentially.'''
    i = 0
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    while i < (len(merged) - train_size - test_size):
        X_train_list.append(merged.iloc[i:i+train_size][X_cols])
        y_train_list.append(merged.iloc[i:i+train_size][y_col])
        X_test_list.append(merged.iloc[i+train_size:i+train_size+test_size][X_cols])
        y_test_list.append(merged.iloc[i+train_size:i+train_size+test_size][y_col])
        i += test_size
    return X_train_list, y_train_list, X_test_list, y_test_list

def train_RF_model(X_train, y_train, n_estimators=300, max_depth=10, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, random_state=42):
    '''Train the Random Forest model.'''
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf

def predict_RF_model(rf, X_test):
    '''Predict the model.'''
    return rf.predict(X_test)

def evaluate_RF_model(y_test, y_pred):
    '''Evaluate the model.'''
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mean_actual = np.mean(y_test)
    cvrmse = rmse / mean_actual
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return cvrmse, r2, rmse, mae

def plot_actual_vs_predicted_single(merged, split_ratio, y_col, y_pred, title='Actual vs Predicted Demand', width=1000, height=600):
    '''Plot the actual vs predicted demand.'''
    split_index = int(len(merged) * split_ratio)
    plot_df = merged.loc[split_index:, ['datetime_utc', y_col]].copy()
    plot_df['Predicted'] = y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['datetime_utc'], y=plot_df[y_col],
                            mode='lines', name='Actual'))
    
    fig.add_trace(go.Scatter(x=plot_df['datetime_utc'], y=plot_df['Predicted'],
                            mode='lines', name='Predicted'))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Power (W)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_layout(width=width, height=height, template='plotly_white')
    fig.show()

def cal_and_show_feature_importances(rf_model, X_cols, width=1000, height=600, title='Random Forest Feature Importances', enable_plot=True):
    '''Calculate and show the feature importances.'''
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': X_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    if enable_plot:
        fig = px.bar(fi_df, x='importance', y='feature', orientation='h',
                    title=title)
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.update_layout(width=width, height=height, template='plotly_white')
        fig.show()
    return fi_df

def create_sliding_window_data(data, window_size, target_col, feature_cols):
    """
    Create sliding window dataset for time series prediction.
    
    Parameters:
    - data: DataFrame with time series data
    - window_size: Number of time steps to use as input features
    - target_col: Name of the target column
    - feature_cols: List of feature columns to use
    
    Returns:
    - X: Feature matrix with sliding windows
    - y: Target values
    """
    X_list = []
    y_list = []
    
    for i in range(window_size, len(data)):
        # Create window of features
        window_features = []
        for j in range(i - window_size, i):
            window_features.extend(data.iloc[j][feature_cols].values)
        
        X_list.append(window_features)
        y_list.append(data.iloc[i][target_col])
    
    return np.array(X_list), np.array(y_list)

def sliding_window_rf_prediction(data, window_size, target_col, feature_cols, 
                                train_ratio=0.7, n_estimators=100, max_depth=10, 
                                random_state=42):
    """
    Perform sliding window Random Forest prediction.
    
    Parameters:
    - data: DataFrame with time series data
    - window_size: Number of time steps to use as input features
    - target_col: Name of the target column
    - feature_cols: List of feature columns to use
    - train_ratio: Ratio of data to use for training
    - n_estimators: Number of trees in Random Forest
    - max_depth: Maximum depth of trees
    - random_state: Random state for reproducibility
    
    Returns:
    - rf_model: Trained Random Forest model
    - X_train, y_train: Training data
    - X_test, y_test: Test data
    - y_pred: Predictions on test data
    - performance_metrics: Dictionary with performance metrics
    """
    # Create sliding window data
    X, y = create_sliding_window_data(data, window_size, target_col, feature_cols)
    
    # Split into train and test
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate performance metrics
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mean_actual = np.mean(y_test)
    cvrmse = rmse / mean_actual
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    performance_metrics = {
        'CVRMSE': cvrmse,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }
    
    return rf_model, X_train, y_train, X_test, y_test, y_pred, performance_metrics

def plot_sliding_window_results(data, window_size, train_ratio, target_col, 
                               y_pred, title='Sliding Window Random Forest Prediction'):
    """
    Plot the results of sliding window prediction.
    
    Parameters:
    - data: Original DataFrame
    - window_size: Window size used
    - train_ratio: Training ratio used
    - target_col: Target column name
    - y_pred: Predictions
    - title: Plot title
    """
    # Calculate test start index
    total_samples = len(data) - window_size
    test_start_idx = int(total_samples * train_ratio) + window_size
    
    # Get test period data
    test_data = data.iloc[test_start_idx:test_start_idx + len(y_pred)].copy()
    test_data['Predicted'] = y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['datetime_utc'], y=test_data[target_col],
                            mode='lines', name='Actual', line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=test_data['datetime_utc'], y=test_data['Predicted'],
                            mode='lines', name='Predicted', line=dict(color='red')))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Power (W)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_layout(width=1000, height=600, template='plotly_white')
    fig.show()


# ===========================
# PyTorch LSTM utilities
# ===========================

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


def create_sliding_window_sequences_for_lstm(data, window_size, target_col, feature_cols):
    """
    Create 3D tensors for LSTM with shape (samples, window_size, num_features)

    Returns
    - X: np.ndarray (N, T, F)
    - y: np.ndarray (N,)
    """
    # 确保window_size是Python原生int类型，避免索引错误
    window_size = int(window_size)
    
    # 确保DataFrame有整数位置索引（重置索引以避免索引类型问题）
    if not isinstance(data.index, pd.RangeIndex):
        data = data.reset_index(drop=True)
    
    # 获取列的整数位置索引，避免字符串列名导致的索引问题
    # 确保所有列索引都是Python原生int类型，并转换为tuple以支持iloc索引
    feature_col_indices = tuple(int(data.columns.get_loc(col)) for col in feature_cols)
    target_col_idx = int(data.columns.get_loc(target_col))
    
    # 将DataFrame转换为numpy数组以提高索引性能并避免类型问题
    data_array = data.values
    
    X_list = []
    y_list = []
    data_len = len(data)
    for i in range(window_size, data_len):
        # 确保所有索引都是Python原生int类型
        start_idx = int(i - window_size)
        end_idx = int(i)
        
        # 直接使用numpy数组索引，避免pandas索引问题
        window_data = data_array[start_idx:end_idx, feature_col_indices]
        X_list.append(window_data)
        
        # 获取目标值
        target_value = data_array[end_idx, target_col_idx]
        y_list.append(float(target_value))
    
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def prepare_lstm_data(data, window_size, target_col, feature_cols, scale=True):
    """
    Prepare LSTM inputs with optional MinMax scaling.

    Returns
    - X_scaled: np.ndarray (N, T, F)
    - y_scaled: np.ndarray (N,)
    - x_scaler: fitted MinMaxScaler for X (or None)
    - y_scaler: fitted MinMaxScaler for y (or None)
    """
    # 确保window_size是Python原生int类型
    window_size = int(window_size)
    
    # 确保传入的是DataFrame的副本，避免修改原始数据
    if isinstance(data, pd.DataFrame):
        data = data.copy()
    
    X, y = create_sliding_window_sequences_for_lstm(data, window_size, target_col, feature_cols)
    if not scale:
        return X, y, None, None

    # Scale per-feature across all time steps by reshaping to (N*T, F)
    n, t, f = X.shape
    X_2d = X.reshape(n * t, f)
    x_scaler = MinMaxScaler()
    X_2d_scaled = x_scaler.fit_transform(X_2d)
    X_scaled = X_2d_scaled.reshape(n, t, f).astype(np.float32)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    return X_scaled, y_scaled, x_scaler, y_scaler


def train_LSTM_model(X_train, y_train, input_size, hidden_size=64, num_layers=2,
                     epochs=20, batch_size=64, lr=1e-3, dropout=0.0, device=None,
                     verbose=False, early_stopping=False, patience=5, min_delta=0.0):
    """Train a PyTorch LSTM regressor and return the trained model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMRegressor(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        avg_loss = epoch_loss/len(dataset)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - MSE: {avg_loss:.6f}")

        if early_stopping:
            # monitor training loss
            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1} (best train MSE={best_loss:.6f})")
                    break

    return model


def predict_LSTM_model(model, X_test, device=None):
    """Predict using a trained PyTorch LSTM model. Returns np.ndarray of shape (N,)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_test).to(device)
        preds = model(xt).detach().cpu().numpy()
    return preds


def sliding_window_lstm_prediction(data, window_size, target_col, feature_cols,
                                   train_ratio=0.7, hidden_size=64, num_layers=2,
                                   epochs=20, batch_size=64, lr=1e-3, dropout=0.0,
                                   scale=True, verbose=False, early_stopping=False,
                                   patience=5, min_delta=0.0):
    """
    Perform sliding window LSTM prediction using PyTorch, mirroring RF utility.

    Returns
    - model: trained LSTMRegressor
    - X_train, y_train: training arrays (scaled if scale=True)
    - X_test, y_test: test arrays (scaled if scale=True)
    - y_pred_inv: predictions in original scale
    - performance_metrics: dict with CVRMSE, R2, RMSE, MAE (original scale)
    - scalers: (x_scaler, y_scaler)
    """
    X_all, y_all, x_scaler, y_scaler = prepare_lstm_data(
        data, window_size, target_col, feature_cols, scale=scale
    )

    split_idx = int(len(X_all) * train_ratio)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    model = train_LSTM_model(
        X_train, y_train, input_size=len(feature_cols), hidden_size=hidden_size,
        num_layers=num_layers, epochs=epochs, batch_size=batch_size, lr=lr,
        dropout=dropout, verbose=verbose, early_stopping=early_stopping,
        patience=patience, min_delta=min_delta
    )

    y_pred = predict_LSTM_model(model, X_test)

    # Inverse scale to original units for metrics
    if y_scaler is not None:
        y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
        y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    else:
        y_test_inv = y_test
        y_pred_inv = y_pred

    rmse = float(np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2)))
    mean_actual = float(np.mean(y_test_inv)) if len(y_test_inv) > 0 else 0.0
    cvrmse = (rmse / mean_actual) if mean_actual != 0 else np.nan
    r2 = float(r2_score(y_test_inv, y_pred_inv)) if len(y_test_inv) > 1 else np.nan
    mae = float(mean_absolute_error(y_test_inv, y_pred_inv)) if len(y_test_inv) > 0 else np.nan

    performance_metrics = {
        'CVRMSE': cvrmse,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
    }

    return (
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        y_pred_inv,
        performance_metrics,
        (x_scaler, y_scaler),
    )


def evaluate_LSTM_model(y_true_original_scale, y_pred_original_scale):
    """Evaluate predictions on original scale to mirror RF metrics."""
    rmse = np.sqrt(np.mean((y_true_original_scale - y_pred_original_scale) ** 2))
    mean_actual = np.mean(y_true_original_scale)
    cvrmse = rmse / mean_actual if mean_actual != 0 else np.nan
    r2 = r2_score(y_true_original_scale, y_pred_original_scale) if len(y_true_original_scale) > 1 else np.nan
    mae = mean_absolute_error(y_true_original_scale, y_pred_original_scale)
    return cvrmse, r2, rmse, mae


def split_train_test_lstm_dataset(
    merged,
    feature_cols,
    target_col,
    window_size,
    split_ratio=0.75,
    scale=True,
    train_start=None,
    train_end=None,
    test_end=None,
):
    """
    Split dataset into LSTM-ready train/test sets using sliding windows, similar to RF helper.

    Returns
    - X_train, y_train, X_test, y_test (scaled if scale=True)
    - scalers: (x_scaler, y_scaler)
    - indices: dict with 'test_start_idx' to align back to original dataframe
    """
    # 确保window_size和split_ratio是Python原生类型
    window_size = int(window_size)
    X_all, y_all, x_scaler, y_scaler = prepare_lstm_data(
        merged, window_size, target_col, feature_cols, scale=scale
    )

    # 统一使用 RangeIndex，方便根据样本索引定位到原始 DataFrame 行
    if not isinstance(merged.index, pd.RangeIndex):
        merged = merged.reset_index(drop=True)

    total_samples = len(merged) - window_size

    # 如果没有指定时间范围，使用原来的按比例划分逻辑
    if train_start is None and train_end is None and test_end is None:
        split_ratio = float(split_ratio)
        split_idx = int(len(X_all) * split_ratio)
        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]

        test_start_idx = int(total_samples * split_ratio) + window_size
        
        # 计算时间范围（如果 merged 有 datetime_utc 列）
        result_dict = {"test_start_idx": test_start_idx}
        if 'datetime_utc' in merged.columns:
            # 确保按时间排序
            merged = merged.sort_values('datetime_utc').reset_index(drop=True)
            
            # 训练集时间范围
            train_input_start_idx = 0
            train_target_start_idx = window_size
            train_target_end_idx = split_idx + window_size - 1
            
            # 测试集时间范围
            test_input_start_idx = split_idx
            test_target_start_idx = split_idx + window_size
            test_target_end_idx = len(X_all) + window_size - 1
            
            # 确保索引不越界
            train_target_end_idx = min(train_target_end_idx, len(merged) - 1)
            test_target_end_idx = min(test_target_end_idx, len(merged) - 1)
            
            result_dict["train_time_range"] = {
                "input_start": merged.iloc[train_input_start_idx]['datetime_utc'],
                "target_start": merged.iloc[train_target_start_idx]['datetime_utc'],
                "target_end": merged.iloc[train_target_end_idx]['datetime_utc'],
            }
            result_dict["test_time_range"] = {
                "input_start": merged.iloc[test_input_start_idx]['datetime_utc'],
                "target_start": merged.iloc[test_target_start_idx]['datetime_utc'],
                "target_end": merged.iloc[test_target_end_idx]['datetime_utc'],
            }
        
        return X_train, y_train, X_test, y_test, (x_scaler, y_scaler), result_dict

    # ===== 按指定时间范围划分 =====
    if 'datetime_utc' not in merged.columns:
        raise KeyError(
            "'datetime_utc' column is required in merged for date-based splitting."
        )

    # 转成 pandas 时间类型
    if train_start is not None:
        train_start = pd.to_datetime(train_start)
    if train_end is not None:
        train_end = pd.to_datetime(train_end)
    if test_end is not None:
        test_end = pd.to_datetime(test_end)

    # 确保按时间排序
    merged = merged.sort_values('datetime_utc').reset_index(drop=True)

    # 找到训练集和测试集在原始 DataFrame 中的“目标值”行索引
    dt = merged['datetime_utc']

    # 训练集：datetime_utc 在 [train_start, train_end] 之间
    if train_start is None or train_end is None:
        raise ValueError("For date-based splitting, both train_start and train_end must be provided.")

    train_mask_data = (dt >= train_start) & (dt <= train_end)
    train_target_indices = np.where(train_mask_data)[0]
    if len(train_target_indices) == 0:
        raise ValueError("No training samples found in the specified train_start/train_end range.")

    # 测试集：从 train_end 之后开始，到 test_end（如果给定） 或数据末尾
    if test_end is not None:
        test_mask_data = (dt > train_end) & (dt <= test_end)
    else:
        test_mask_data = (dt > train_end)
    test_target_indices = np.where(test_mask_data)[0]
    if len(test_target_indices) == 0:
        raise ValueError("No test samples found in the specified range after train_end.")

    # 将“目标值行索引”映射到滑动窗口样本索引：
    # 在 create_sliding_window_sequences_for_lstm 中，
    # 第 s 个样本的目标值来自原始 DataFrame 的行索引 (s + window_size)
    sample_indices = np.arange(total_samples, dtype=int)
    target_indices_from_samples = sample_indices + window_size

    train_sample_mask = np.isin(target_indices_from_samples, train_target_indices)
    test_sample_mask = np.isin(target_indices_from_samples, test_target_indices)

    if not train_sample_mask.any():
        raise ValueError("No training sliding-window samples fall into the specified train_start/train_end range.")
    if not test_sample_mask.any():
        raise ValueError("No testing sliding-window samples fall into the specified test range.")

    X_train, y_train = X_all[train_sample_mask], y_all[train_sample_mask]
    X_test, y_test = X_all[test_sample_mask], y_all[test_sample_mask]

    # 用于对齐回原始 DataFrame 的测试起始索引：第一个测试样本的目标值在 merged 中的行索引
    first_test_target_idx = int(target_indices_from_samples[test_sample_mask][0])
    
    # 计算训练集和测试集的实际时间范围
    # 训练集：第一个样本的输入窗口起始时间，最后一个样本的目标值时间
    train_target_indices_actual = target_indices_from_samples[train_sample_mask]
    train_input_start_idx = int(train_target_indices_actual[0] - window_size)
    train_target_start_idx = int(train_target_indices_actual[0])
    train_target_end_idx = int(train_target_indices_actual[-1])
    
    # 测试集：第一个样本的输入窗口起始时间，最后一个样本的目标值时间
    test_target_indices_actual = target_indices_from_samples[test_sample_mask]
    test_input_start_idx = int(test_target_indices_actual[0] - window_size)
    test_target_start_idx = int(test_target_indices_actual[0])
    test_target_end_idx = int(test_target_indices_actual[-1])
    
    # 获取实际的时间戳
    train_input_start_time = merged.iloc[train_input_start_idx]['datetime_utc']
    train_target_start_time = merged.iloc[train_target_start_idx]['datetime_utc']
    train_target_end_time = merged.iloc[train_target_end_idx]['datetime_utc']
    
    test_input_start_time = merged.iloc[test_input_start_idx]['datetime_utc']
    test_target_start_time = merged.iloc[test_target_start_idx]['datetime_utc']
    test_target_end_time = merged.iloc[test_target_end_idx]['datetime_utc']

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        (x_scaler, y_scaler),
        {
            "test_start_idx": first_test_target_idx,
            "train_time_range": {
                "input_start": train_input_start_time,
                "target_start": train_target_start_time,
                "target_end": train_target_end_time,
            },
            "test_time_range": {
                "input_start": test_input_start_time,
                "target_start": test_target_start_time,
                "target_end": test_target_end_time,
            },
        },
    )


# ===========================
# PyTorch Transformer utilities
# ===========================

class TransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection to d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoder = nn.Parameter(torch.randn(1000, d_model))  # Max sequence length 1000
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        seq_len = x.size(1)
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoder[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        x = x + pos_enc
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Take the last time step
        x = x[:, -1, :]  # (batch, d_model)
        
        # Output projection
        x = self.dropout(x)
        x = self.fc(x)  # (batch, 1)
        
        return x.squeeze(-1)  # (batch,)


def create_sliding_window_sequences_for_transformer(data, window_size, target_col, feature_cols):
    """
    Create 3D tensors for Transformer with shape (samples, window_size, num_features)
    Same as LSTM function - Transformer can reuse the same data preparation.

    Returns
    - X: np.ndarray (N, T, F)
    - y: np.ndarray (N,)
    """
    # Reuse LSTM data preparation function
    return create_sliding_window_sequences_for_lstm(data, window_size, target_col, feature_cols)


def prepare_transformer_data(data, window_size, target_col, feature_cols, scale=True):
    """
    Prepare Transformer inputs with optional MinMax scaling.
    Same as LSTM - Transformer can reuse the same data preparation.

    Returns
    - X_scaled: np.ndarray (N, T, F)
    - y_scaled: np.ndarray (N,)
    - x_scaler: fitted MinMaxScaler for X (or None)
    - y_scaler: fitted MinMaxScaler for y (or None)
    """
    # Reuse LSTM data preparation function
    return prepare_lstm_data(data, window_size, target_col, feature_cols, scale=scale)


def train_Transformer_model(X_train, y_train, input_size, d_model=64, nhead=4, num_layers=2,
                             dim_feedforward=256, epochs=20, batch_size=64, lr=1e-3, dropout=0.1,
                             device=None, verbose=False, early_stopping=False, patience=5,
                             min_delta=0.0, val_ratio=0.2):
    """Train a PyTorch Transformer regressor and return the trained model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerRegressor(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    dataset = TensorDataset(X_train_tensor, y_train_tensor)

    if early_stopping and len(dataset) > 1 and val_ratio > 0.0:
        val_size = max(1, int(len(dataset) * float(val_ratio)))
        train_size = len(dataset) - val_size
        if train_size < 1:
            # Fallback to no validation if dataset too small
            train_size = len(dataset)
            val_size = 0
        if val_size > 0:
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            val_loader = None
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = None

    best_loss = float('inf')
    best_state = None
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        train_epoch_loss = epoch_loss / len(train_loader.dataset)

        val_epoch_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * len(xb)
            val_epoch_loss = val_loss / len(val_loader.dataset)

        metric_to_monitor = val_epoch_loss if val_epoch_loss is not None else train_epoch_loss

        if metric_to_monitor + float(min_delta) < best_loss:
            best_loss = metric_to_monitor
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if verbose:
            if val_epoch_loss is not None:
                print(f"Epoch {epoch+1}/{epochs} - Train MSE: {train_epoch_loss:.6f} | Val MSE: {val_epoch_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - MSE: {train_epoch_loss:.6f}")

        if early_stopping and val_loader is not None and no_improve_epochs >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_Transformer_model(model, X_test, device=None):
    """Predict using a trained PyTorch Transformer model. Returns np.ndarray of shape (N,)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_test).to(device)
        preds = model(xt).detach().cpu().numpy()
    return preds


def split_train_test_transformer_dataset(
    merged,
    feature_cols,
    target_col,
    window_size,
    split_ratio=0.75,
    scale=True,
    train_start=None,
    train_end=None,
    test_end=None,
):
    """
    Split dataset into Transformer-ready train/test sets using sliding windows.
    Same as LSTM - Transformer can reuse the same data preparation.

    Returns
    - X_train, y_train, X_test, y_test (scaled if scale=True)
    - scalers: (x_scaler, y_scaler)
    - indices: dict with 'test_start_idx' to align back to original dataframe
    """
    # Reuse LSTM data splitting function
    return split_train_test_lstm_dataset(
        merged,
        feature_cols,
        target_col,
        window_size,
        split_ratio,
        scale,
        train_start=train_start,
        train_end=train_end,
        test_end=test_end,
    )


def evaluate_Transformer_model(y_true_original_scale, y_pred_original_scale):
    """Evaluate predictions on original scale to mirror LSTM metrics."""
    # Reuse LSTM evaluation function
    return evaluate_LSTM_model(y_true_original_scale, y_pred_original_scale)

def online_update_transformer(
    model,
    X_new, y_new,
    optimizer=None,
    lr=1e-4,
    batch_size=32,
    grad_clip=1.0,
    steps=1,
    device=None,
):
    """
    Online learning: run a few gradient steps on the newest batch/window.
    X_new: np.ndarray (N, T, F)
    y_new: np.ndarray (N,)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_new).float().to(device)
    y_t = torch.from_numpy(y_new).float().to(device)

    # 小数据就直接全量；大一点就 mini-batch
    N = X_t.size(0)
    for _ in range(int(steps)):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_t[idx], y_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

    return optimizer

class OnlineWindowBuffer:
    def __init__(self, window_size: int, n_features: int):
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.buf = deque(maxlen=self.window_size)

    def push(self, x_t: np.ndarray):
        # x_t shape: (F,)
        self.buf.append(np.asarray(x_t, dtype=np.float32))

    def ready(self) -> bool:
        return len(self.buf) == self.window_size

    def get_X(self) -> np.ndarray:
        # returns (1, T, F)
        X = np.stack(list(self.buf), axis=0)  # (T, F)
        return X[None, :, :]


