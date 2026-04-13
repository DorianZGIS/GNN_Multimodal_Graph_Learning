import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import pickle
import random
import os
from datetime import datetime
import geopandas as gpd

from sklearn.preprocessing import StandardScaler
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset



    # StaticGraphTemporalSignal
from typing import Sequence, Union

Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = Sequence[Union[np.ndarray, None]]
Targets = Sequence[Union[np.ndarray, None]]
Additional_Features = Sequence[np.ndarray]

class StaticGraphTemporalSignal(object):
    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Data object. Between two
    temporal snapshots the features and optionally passed attributes might change.
    However, the underlying graph is the same.

    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """

    def __init__(
        self,
        edge_index: Edge_Index,
        edge_weight: Edge_Weight,
        features: Node_Features,
        targets: Targets,
        **kwargs: Additional_Features
    ):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

        # This function was added by me so that this dataset can be used with the split_train_test function from scikit learn
    def __len__(self): 
        return self.snapshot_count

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return torch.LongTensor(self.edge_index)

    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return torch.FloatTensor(self.edge_weight)

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == "i":
                return torch.LongTensor(self.targets[time_index])
            elif self.targets[time_index].dtype.kind == "f":
                return torch.FloatTensor(self.targets[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = StaticGraphTemporalSignal(
                self.edge_index,
                self.edge_weight,
                self.features[time_index],
                self.targets[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index()
            edge_weight = self._get_edge_weight()
            y = self._get_target(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, **additional_features)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self

# DynamicGraphTemporalSignal

import torch
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data


Edge_Indices = Sequence[Union[np.ndarray, None]]
Edge_Weights = Sequence[Union[np.ndarray, None]]
Node_Features = Sequence[Union[np.ndarray, None]]
Targets = Sequence[Union[np.ndarray, None]]
Additional_Features = Sequence[np.ndarray]


class DynamicGraphTemporalSignal(object):
    r"""A data iterator object to contain a dynamic graph with a
    changing edge set and weights . The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric Data object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.

    Args:
        edge_indices (Sequence of Numpy arrays): Sequence of edge index tensors.
        edge_weights (Sequence of Numpy arrays): Sequence of edge weight tensors.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """

    def __init__(
        self,
        edge_indices: Edge_Indices,
        edge_weights: Edge_Weights,
        features: Node_Features,
        targets: Targets,
        **kwargs: Additional_Features
    ):
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.features = features
        self.targets = targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()
        
    # This function was added by me so that this dataset can be used with the split_train_test function from scikit learn
    def __len__(self): 
        return self.snapshot_count

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        assert len(self.edge_indices) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        assert len(self.features) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self, time_index: int):
        if self.edge_indices[time_index] is None:
            return self.edge_indices[time_index]
        else:
            return torch.LongTensor(self.edge_indices[time_index])

    def _get_edge_weight(self, time_index: int):
        if self.edge_weights[time_index] is None:
            return self.edge_weights[time_index]
        else:
            return torch.FloatTensor(self.edge_weights[time_index])

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == "i":
                return torch.LongTensor(self.targets[time_index])
            elif self.targets[time_index].dtype.kind == "f":
                return torch.FloatTensor(self.targets[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = DynamicGraphTemporalSignal(
                self.edge_indices[time_index],
                self.edge_weights[time_index],
                self.features[time_index],
                self.targets[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index(time_index)
            edge_weight = self._get_edge_weight(time_index)
            y = self._get_target(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, **additional_features)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self


def eval( test_models, opti, lr_rate, loss_type, dfs, counties, model_name='cases_tweets_static',
          metric_eps: float = 1e-8, device_in='cpu'):

    valid_loss_lists = []# Store results from each fold

    train_loss_lists = []
    valid_loss_lists = []
    all_results = []  # NEW: To store losses for each fold in dict format
    all_fold_dfs = []

    for fold in tqdm(range(len(test_models[0]))):
        # print('Fold:', fold)    
        # train_dataset, valid_dataset = train_test_split(staticGnnDataset, test_size= 1 - train_size, random_state=fold)
        # valid_dataset, test_dataset = train_test_split(valid_dataset, test_size=0.5, random_state=fold)
        test_loader = test_models[0][fold]#DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

        # Define model
        # model = TemporalGNN(node_features=len(train_valid_dfs), periods=num_timesteps_out).to(device) # len(dfs) to 1 because only cases input
        model = test_models[1][fold]# .load_state_dict(torch.load('model_checkpoints/fold_'+str(fold)+'/'+model_name+'.pt'))

        train_mse_list = []
        valid_mse_list = []
        edges_list = []
        
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # criterion = nn.MSELoss()
        if opti == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
        elif opti == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=1e-4)
        elif opti == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9)

        loss_fn = get_loss_fn(loss_type=loss_type, eps=metric_eps)

        model.eval()  # Set the model to evaluation mode
        test_loss = 0
        test_steps = 0
        predictions = []
        y_ = []
        x_ = []
        with torch.no_grad():
            for snapshot in test_loader:
                snapshot = snapshot.to(device_in)
                y_hat = model(snapshot.x, snapshot.edge_index)
                x_.append(snapshot.x)
                predictions.append(y_hat)
                y_.append(snapshot.y)
                test_loss += loss_fn(y_hat, snapshot.y)
                test_steps += 1
        test_loss = test_loss / test_steps
        test_mse = test_loss.item()

        mse_per_region = []
        y_hat_regions = []
        y_regions = []
        for region in range(len(dfs[0].columns)):
            y_hat_region = torch.stack([y_hat[region] for y_hat in predictions])
            y_region = torch.stack([y[region] for y in y_])

            y_hat_regions.append(y_hat_region.cpu().numpy())
            y_regions.append(y_region.cpu().numpy())
            # Calculate the MSE for the current region
            mse = loss_fn(y_region, y_hat_region).cpu()

            mse_per_region.append(mse.item())

        df_mse = pd.DataFrame({'county': list(dfs[0].columns), 'mse': mse_per_region, 'y_region':y_regions, 'y_hat_region':y_hat_regions })
        df_mse = pd.DataFrame({
            'county': list(dfs[0].columns),
            'mse': mse_per_region,
            'y_region': y_regions,
            'y_hat_region': y_hat_regions
        })

        # tag with fold and collect
        df_mse['fold'] = fold
        all_fold_dfs.append(df_mse)

    # once all folds are processed:
    big_df = pd.concat(all_fold_dfs, ignore_index=True)

    # 1) compute mean MSE per county
    mean_mse_df = big_df.groupby('county').agg(
        mean_mse = ('mse', 'mean'),
        n_folds  = ('fold', 'nunique')
    ).reset_index()

    # 2) min–max normalize the mean_mse
    min_mse = mean_mse_df['mean_mse'].min()
    max_mse = mean_mse_df['mean_mse'].max()
    mean_mse_df['mse_norm'] = (mean_mse_df['mean_mse'] - min_mse) / (max_mse - min_mse)

    mean = mean_mse_df['mean_mse'].mean()
    std  = mean_mse_df['mean_mse'].std(ddof=0)
    mean_mse_df['mse_zscore'] = (mean_mse_df['mean_mse'] - mean) / std

    region_mse_with_errors = pd.merge(mean_mse_df, counties, left_on='county',right_on='countyfips', how='left')
    region_mse_with_errors = region_mse_with_errors[[ 'mean_mse','mse_norm', 'mse_zscore', 'id', 'name','countyfips', 'state', 'population', 'state_full', 'geom_5070','geometry']]
    region_mse_with_errors = gpd.GeoDataFrame(region_mse_with_errors, geometry='geometry')
    out_dir = "regional_mse"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"region_mse_{model_name}.shp")

    region_mse_with_errors.to_file(out_path)
    # region_mse_with_errors.to_file('regional_mse/region_mse'+model_name+'.shp')

    

def cross_val_yeojohnson_norm(model_name,
               dfs,
               gnndataset,
               TemporalGNN,
               device,
               lr_rate=0.001,
               opti = 'adam',
               num_folds=12,
               num_epochs=100,
               train_size=0.7,
               batch_size=1,
               model_checkpoint_dir='model_checkpoints',
               random_seed=48,
               num_timesteps_out=3,
               loss_type="mse",
               metric_eps=1e-8,
               scaling_method = 'z_score'):

    """
    Cross validation loop.

    Args:
        model_name:        name of the model (used for logs and checkpoint paths)
        dfs:               list of dataframes, used to infer node_features = len(dfs)
        gnndataset:        full dataset (list or Dataset of Data objects with x, y, edge_index)
        TemporalGNN:       model class
        device:            torch.device
        num_folds:         number of random folds
        num_epochs:        epochs per fold
        train_size:        proportion of data in the training split
        batch_size:        DataLoader batch size
        model_checkpoint_dir: base directory for saving checkpoints
        random_seed:       seed for fold generation
        num_timesteps_out: number of output periods for the model
        loss_type:         "mse", "mae", "huber" / "smoothl1", or "mape" (not recommended)
        metric_eps:        epsilon for MAPE stability

    Returns:
        test_loaders, models
    """

    log_filename = f"logs/{model_name}_log.txt"
    if os.path.exists(log_filename):
        os.remove(log_filename)

    train_loss_lists = []
    valid_loss_lists = []
    all_results = []
    models = []
    test_loaders = []

    random_folds = generate_random_list(length=num_folds, seed=random_seed)
    #random_folds = [356, 149, 428, 152, 392, 51, 143, 385, 227, 5, 406, 399, 342, 383, 153] #301
    log("Random Folds: " + str(random_folds), log_filename)

    for fold in tqdm(random_folds):
        log(f'\n--- Fold: {fold} ---', log_filename)

        # Split raw, unscaled dataset
        train_dataset_unscaled, valid_dataset_unscaled = train_test_split(
            gnndataset, test_size=1 - train_size, random_state=fold
        )
        valid_dataset_unscaled, test_dataset_unscaled = train_test_split(
            valid_dataset_unscaled, test_size=0.5, random_state=fold
        )

        # Fit scalers on training set only
        params = fit_scalers_split_first_channel_yeojohnson(
            train_dataset_unscaled, device=None, eps=1e-8, unbiased=False, method=scaling_method
        )

        # Scaled copies, do not mutate gnndataset
        train_dataset = make_normalized_dataset(train_dataset_unscaled, params, device=None)
        valid_dataset = make_normalized_dataset(valid_dataset_unscaled, params, device=None)
        test_dataset = make_normalized_dataset(test_dataset_unscaled, params, device=None)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

        # Model and optimizer
        model = TemporalGNN(node_features=len(dfs), periods=num_timesteps_out).to(device)
        if opti == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
        elif opti == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=1e-4)
        elif opti == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9)

        loss_fn = get_loss_fn(loss_type=loss_type, eps=metric_eps)

        # Metric histories per fold
        train_metrics_hist = {"mse": [], "rmse": [], "mae": [], "mape": []}
        valid_metrics_hist = {"mse": [], "rmse": [], "mae": [], "mape": []}

        log("Running training...", log_filename)
        for epoch in range(num_epochs):
            model.train()
            train_loss_list = []
            batch_train_metrics = {"mse": [], "rmse": [], "mae": [], "mape": []}

            for snapshot in train_loader:
                snapshot = snapshot.to(device)
                y_hat = model(snapshot.x, snapshot.edge_index)

                loss = loss_fn(y_hat, snapshot.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())

                with torch.no_grad():
                    m = compute_metrics(y_hat, snapshot.y, eps=metric_eps)
                    for k in batch_train_metrics:
                        batch_train_metrics[k].append(m[k])

            # Average train metrics over batches
            train_mse = sum(batch_train_metrics["mse"]) / len(batch_train_metrics["mse"])
            train_rmse = sum(batch_train_metrics["rmse"]) / len(batch_train_metrics["rmse"])
            train_mae = sum(batch_train_metrics["mae"]) / len(batch_train_metrics["mae"])
            train_mape = sum(batch_train_metrics["mape"]) / len(batch_train_metrics["mape"])

            train_loss = sum(train_loss_list) / len(train_loss_list)
            train_loss_lists.append(train_loss)

            train_metrics_hist["mse"].append(train_mse)
            train_metrics_hist["rmse"].append(train_rmse)
            train_metrics_hist["mae"].append(train_mae)
            train_metrics_hist["mape"].append(train_mape)

            # Validation
            model.eval()
            valid_loss_list = []
            batch_valid_metrics = {"mse": [], "rmse": [], "mae": [], "mape": []}

            with torch.no_grad():
                for snapshot in valid_loader:
                    snapshot = snapshot.to(device)
                    y_hat = model(snapshot.x, snapshot.edge_index)

                    v_loss = loss_fn(y_hat, snapshot.y)
                    valid_loss_list.append(v_loss.item())

                    m = compute_metrics(y_hat, snapshot.y, eps=metric_eps)
                    for k in batch_valid_metrics:
                        batch_valid_metrics[k].append(m[k])

            valid_mse = sum(batch_valid_metrics["mse"]) / len(batch_valid_metrics["mse"])
            valid_rmse = sum(batch_valid_metrics["rmse"]) / len(batch_valid_metrics["rmse"])
            valid_mae = sum(batch_valid_metrics["mae"]) / len(batch_valid_metrics["mae"])
            valid_mape = sum(batch_valid_metrics["mape"]) / len(batch_valid_metrics["mape"])

            valid_loss = sum(valid_loss_list) / len(valid_loss_list)
            valid_loss_lists.append(valid_loss)

            valid_metrics_hist["mse"].append(valid_mse)
            valid_metrics_hist["rmse"].append(valid_rmse)
            valid_metrics_hist["mae"].append(valid_mae)
            valid_metrics_hist["mape"].append(valid_mape)

            # Log: use the actual optimized loss plus all extra metrics
            log(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train {loss_type.upper()}: {train_loss:.4f}, "
                f"Valid {loss_type.upper()}: {valid_loss:.4f}, "
                f"[MSE: {valid_mse:.4f}, MAE: {valid_mae:.4f}, RMSE: {valid_rmse:.4f}, MAPE: {valid_mape:.4f}]",
                log_filename
            )

        # Store fold metrics
        all_results.append({
            "fold": fold,
            "loss_type": loss_type,
            "train_metrics": train_metrics_hist,
            "valid_metrics": valid_metrics_hist,
        })

        # Save model and metrics
        # model_dir = os.path.join(model_checkpoint_dir, f'fold_{fold}')
        # os.makedirs(model_dir, exist_ok=True)
        # model_path = os.path.join(model_dir, model_name + '.pt')
        # torch.save(model.state_dict(), model_path)

        from pathlib import Path

        # base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        # model_dir = base / model_checkpoint_dir / f"fold_{fold}"
        # model_dir.mkdir(parents=True, exist_ok=True)
        # model_path = model_dir / f"{model_name}.pt"
        # p = Path(model_path)
        # print("abs:", p.resolve())
        # print("len:", len(str(p.resolve())))
        # print("dir exists:", p.parent.exists())
        # print("write access:", os.access(str(p.parent), os.W_OK))
        # torch.save(model.state_dict(), str(model_path))
 # Save model and metrics
        base_dir = os.path.dirname(os.path.abspath(__file__))

        model_dir = os.path.join(base_dir, model_checkpoint_dir, f"fold_{fold}")
        os.makedirs(model_dir, exist_ok=True)

        # ----- model path -----
        model_path = os.path.join(model_dir, model_name + ".pt")

        # ----- results path -----
        results_path = os.path.join(model_dir, "training_results_" + model_name + ".pkl")

        # Windows long-path fix for BOTH files
        if os.name == "nt":
            model_path = r"\\?\\" + os.path.abspath(model_path)
            results_path = r"\\?\\" + os.path.abspath(results_path)

        torch.save(model.state_dict(), model_path)

        with open(results_path, "wb") as f:
            pickle.dump(all_results, f)




        # results_path = os.path.join(model_dir, 'training_results_' + model_name + '.pkl')
        # with open(results_path, 'wb') as f:
        #     pickle.dump(all_results, f)

        models.append(model)
        test_loaders.append(test_loader)

    return (test_loaders, models)



def compute_metrics(y_hat, y, eps: float = 1e-8):
    """
    Compute scalar metrics from predictions and targets on the (scaled) domain.

    Returns a dict of plain Python floats:
      - mse
      - rmse
      - mae
      - mape
    """
    diff = y_hat - y
    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()
    rmse = mse ** 0.5
    denom = y.abs() + eps
    mape = (diff.abs() / denom).mean().item()
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }



class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y) + self.eps)




def get_loss_fn(loss_type: str = "mse", eps: float = 1e-8):
    """
    Return a loss function based on loss_type.

    loss_type: "mse", "rmse", "mae", "huber"/"smoothl1", or "mape".
    """
    loss_type = loss_type.lower()

    if loss_type == "mse":
        return nn.MSELoss()

    elif loss_type == "rmse":
        return RMSELoss(eps=eps)

    elif loss_type == "mae":
        return nn.L1Loss()

    elif loss_type in ("huber", "smoothl1"):
        return nn.SmoothL1Loss()

    elif loss_type == "mape":
        def mape_loss(y_hat, y):
            denom = y.abs() + eps
            return ((y_hat - y).abs() / denom).mean()
        return mape_loss

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")



def custom_collate(batch):
    # Extract individual elements from the Data objects
    x = [sample.x for sample in batch][0]
    edge_index = [sample.edge_index for sample in batch][0]
    edge_attr = [sample.edge_attr for sample in batch][0]
    y = [sample.y for sample in batch][0]

    # Collate them into a new Data object
    batched_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return batched_data

@torch.no_grad()
def _fit_channel_params(tensor4d, method: str, eps=1e-8, unbiased=False):
    """
    tensor4d: [N, N_nodes, Cch, T]
    Fit mean/std and (if needed) lambda/shift per channel.
    Returns dict with tensors of shape [Cch].
    """
    N, N_nodes, Cch, T = tensor4d.shape

    # flatten to [Cch, N * N_nodes * T]
    data = (
        tensor4d
        .permute(2, 0, 1, 3)      # [Cch, N, nodes, T]
        .reshape(Cch, -1)         # [Cch, N*nodes*T]
        .cpu()
        .numpy()
        .astype(float)
    )

    means, stds, lambdas, shifts = [], [], [], []

    ddof = 1 if unbiased else 0

    for c in range(Cch):
        vals = data[c]

        if method == "z_score":
            m = vals.mean()
            s = vals.std(ddof=ddof)
            lam = 0.0
            shift = 0.0

        elif method == "box_cox":
            # ensure strictly positive
            if np.any(vals <= 0):
                shift = -vals.min() + eps
                vals_pos = vals + shift
            else:
                shift = 0.0
                vals_pos = vals

            bc, lam = stats.boxcox(vals_pos)
            m = bc.mean()
            s = bc.std(ddof=ddof)

        elif method == "yeo_johnson":
            shift = 0.0
            yj, lam = stats.yeojohnson(vals)
            m = yj.mean()
            s = yj.std(ddof=ddof)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        means.append(m)
        stds.append(max(s, eps))
        lambdas.append(lam)
        shifts.append(shift)

    return {
        "mean":   torch.tensor(means,   dtype=torch.float32),
        "std":    torch.tensor(stds,    dtype=torch.float32),
        "lambda": torch.tensor(lambdas, dtype=torch.float32),
        "shift":  torch.tensor(shifts,  dtype=torch.float32),
    }



def gather_all_tensors_advanced(dataset, device=None):
    """Stack all x and y tensors from dataset."""
    X, Y = [], []
    for d in dataset:
        x, y = d.x, d.y              # x: [594, C, Tx], y: [594, Ty]
        if device is not None:
            x, y = x.to(device), y.to(device)
        X.append(x.unsqueeze(0))     # [1, 594, C, Tx]
        Y.append(y.unsqueeze(0))     # [1, 594, Ty]
    # X: [N, 594, C, Tx], Y: [N, 594, Ty]
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)


@torch.no_grad()
def fit_scalers_split_first_channel_yeojohnson(dataset, method: str = "z_score",
                                    device=None, eps=1e-8, unbiased=False):
    """
    Fit normalization parameters for chosen method:
      - 'z_score'
      - 'box_cox'
      - 'yeo_johnson'

    Channel 1 (index 0): uses X and Y concatenated in time.
    Remaining channels: use only X.

    Returns:
        params: dict with keys:
          'method' : str
          'c1'     : dict(mean, std, lambda, shift)  each [1]
          'c2'     : dict(...) or None
    """
    X, Y = gather_all_tensors_advanced(dataset, device=device)
    # X: [N, 594, C, Tx]
    # Y: [N, 594, Ty]

    N, N_nodes, C, Tx = X.shape
    _, _, Ty = Y.shape

    # Split channels
    X_c1 = X[:, :, 0:1, :]      # [N, 594, 1, Tx]
    X_c2 = X[:, :, 1:, :]       # [N, 594, C2, Tx] (possibly C2 = 0)

    # For first channel, concat past (X) and future (Y) along time
    Y_expanded = Y.unsqueeze(2)  # [N, 594, 1, Ty]
    X_c1_full = torch.cat([X_c1, Y_expanded], dim=3)  # [N, 594, 1, Tx+Ty]

    c1_params = _fit_channel_params(X_c1_full, method, eps=eps, unbiased=unbiased)

    if X_c2.shape[2] > 0:
        c2_params = _fit_channel_params(X_c2, method, eps=eps, unbiased=unbiased)
    else:
        c2_params = None

    return {
        "method": method,
        "c1": c1_params,   # dict with mean/std/lambda/shift [1]
        "c2": c2_params,   # dict with mean/std/lambda/shift [C2] or None
    }


def _boxcox_torch(x, lam, shift):
    """
    Box-Cox transform in torch.
    x     : tensor [..., C, ...]
    lam   : tensor broadcastable to x
    shift : tensor broadcastable to x  (for positivity)
    """
    x_pos = x + shift
    x_pos = x_pos.clamp_min(1e-6)
    return (x_pos.pow(lam) - 1.0) / lam


def _yeojohnson_torch(x, lam):
    """
    Yeo-Johnson transform in torch.
    x   : tensor
    lam : tensor broadcastable to x
    """
    out = torch.empty_like(x)
    pos = x >= 0
    neg = ~pos

    # expand lam to same shape as x for easy masking
    lam_exp = lam.expand_as(x)

    # x >= 0
    out[pos] = ((x[pos] + 1.0).pow(lam_exp[pos]) - 1.0) / lam_exp[pos]

    # x < 0
    out[neg] = -(
        ((-x[neg] + 1.0).pow(2.0 - lam_exp[neg]) - 1.0) /
        (2.0 - lam_exp[neg])
    )

    return out

@torch.no_grad()
def normalize_xy_sample(x, y, params):
    """
    x: [594, C, Tx]
    y: [594, Ty]
    params: dict returned by fit_scalers_split_first_channel

    Returns:
        x_norm, y_norm (same shapes).
    """
    method = params["method"]
    c1 = params["c1"]
    c2 = params["c2"]

    # split x
    x_c1 = x[:, 0:1, :]      # [594, 1, Tx]
    x_c2 = x[:, 1:, :]       # [594, C2, Tx]

    # ----- first channel (cases) -----
    mean_c1 = c1["mean"][0].to(x.device)
    std_c1  = c1["std"][0].to(x.device)
    lam_c1  = c1["lambda"][0].to(x.device)
    shift_c1 = c1["shift"][0].to(x.device)

    # shape for broadcasting
    mean_c1_x = mean_c1.view(1, 1, 1)
    std_c1_x  = std_c1.view(1, 1, 1)
    mean_c1_y = mean_c1.view(1, 1)
    std_c1_y  = std_c1.view(1, 1)

    if method == "z_score":
        tx_c1 = x_c1
        ty    = y
    elif method == "box_cox":
        lam_bc_x = lam_c1.view(1, 1, 1)
        shift_bc_x = shift_c1.view(1, 1, 1)
        lam_bc_y = lam_c1.view(1, 1)
        shift_bc_y = shift_c1.view(1, 1)
        tx_c1 = _boxcox_torch(x_c1, lam_bc_x, shift_bc_x)
        ty    = _boxcox_torch(y,    lam_bc_y, shift_bc_y)
    elif method == "yeo_johnson":
        lam_yj_x = lam_c1.view(1, 1, 1)
        lam_yj_y = lam_c1.view(1, 1)
        tx_c1 = _yeojohnson_torch(x_c1, lam_yj_x)
        ty    = _yeojohnson_torch(y,    lam_yj_y)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # z-score after chosen transform
    x_c1_norm = (tx_c1 - mean_c1_x) / std_c1_x
    y_norm    = (ty    - mean_c1_y) / std_c1_y

    # ----- remaining channels (if any) -----
    if x_c2.shape[1] > 0 and c2 is not None:
        mean_c2 = c2["mean"].to(x.device)     # [C2]
        std_c2  = c2["std"].to(x.device)      # [C2]
        lam_c2  = c2["lambda"].to(x.device)   # [C2]
        shift_c2 = c2["shift"].to(x.device)   # [C2]

        mean_c2_bc = mean_c2.view(1, -1, 1)   # [1, C2, 1]
        std_c2_bc  = std_c2.view(1, -1, 1)
        lam_c2_bc  = lam_c2.view(1, -1, 1)
        shift_c2_bc = shift_c2.view(1, -1, 1)

        if method == "z_score":
            tx_c2 = x_c2
        elif method == "box_cox":
            tx_c2 = _boxcox_torch(x_c2, lam_c2_bc, shift_c2_bc)
        elif method == "yeo_johnson":
            tx_c2 = _yeojohnson_torch(x_c2, lam_c2_bc)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        x_c2_norm = (tx_c2 - mean_c2_bc) / std_c2_bc
        x_norm = torch.cat([x_c1_norm, x_c2_norm], dim=1)
    else:
        x_norm = x_c1_norm

    return x_norm, y_norm

@torch.no_grad()
def make_normalized_dataset(dataset, params, device=None):
    import copy
    new_dataset = []
    for d in dataset:
        d_new = copy.copy(d)  # shallow copy container
        x, y = d.x, d.y

        if device is not None:
            x = x.to(device)
            y = y.to(device)

        x_norm, y_norm = normalize_xy_sample(x, y, params)

        d_new.x = x_norm
        d_new.y = y_norm
        new_dataset.append(d_new)

    return new_dataset


def log(msg, log_filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    filename = os.path.basename(log_filename)
    full_path = os.path.join(log_dir, filename)

    # Windows long-path fix
    if os.name == "nt":
        full_path = r"\\?\\" + os.path.abspath(full_path)

    with open(full_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")



def generate_random_list(length, seed=None, lower=0, upper=500):
    if seed is not None:
        random.seed(seed)
    return [random.randint(lower, upper) for _ in range(length)]

def prepare_model_dataset(
    MODEL_MAP,
    model_code,
    train_valid_dfs,
    start_date,
    end_date,
    best_counties,
    percent=25,
    rolling_window=14,
    folder='datasets/', #datasets/
    device='cuda',
    num_timesteps_in = 14,
    num_timesteps_out = 3):
    
    model_name = MODEL_MAP.get(model_code.upper())
    if not model_name:
        raise ValueError(f"Model code {model_code} is not defined in MODEL_MAP.")
    print(f"\n=== Preparing Model: {model_name} (in={num_timesteps_in}, out={num_timesteps_out}) ===")

    # Select which dataframes to use
    if 'casesonly' in model_name:
        dfs = [train_valid_dfs[0]]
    else:
        dfs = train_valid_dfs
        if 'randts' in model_name:
            #dfs = add_rand_time_series(dfs, rolling_window=rolling_window, viz=False)
            dfs[1].index = dfs[0].index

    # Construct dataset filename with input/output timesteps
    filename = f"dataset_{model_code}_{model_name}_in{num_timesteps_in}_out{num_timesteps_out}.pkl"
    full_path = os.path.join(folder, filename)
    print('full_path:'+ full_path)
    # Load or create dataset
    if os.path.exists(full_path):
        print(f"Loaded dataset from: {full_path}")
        with open(full_path, "rb") as f_in:
            gnndataset = pickle.load(f_in)
    else:
        print(f"Creating dataset and saving to: {full_path}")

        # Load adjacency matrix
        if 'static' in model_name:
            adjacencies = load_static_adj(best_counties)
        elif 'mobs' in model_name:
            adjacencies = load_dynamic_adj(start_date, end_date, dfs, percent=percent, load_func=load_mobility_flows)
        elif 'sem' in model_name:
            adjacencies = load_dynamic_adj(start_date, end_date, dfs, percent=percent, load_func=load_semantic_sim)
        elif 'dynamic' in model_name:
            adjacencies = load_dynamic_adj(start_date, end_date, dfs, percent=percent, load_func=load_mobility_flows)
        else:
            raise ValueError(f"Unknown adjacency type in model name: {model_name}")

        # Replace adjacencies with random ones if required
        if 'randadj' in model_name:
            if isinstance(adjacencies, list):
                adjacencies = [keep_topk_percent_torch(torch.rand_like(adj), percent) for adj in adjacencies]
            else:
                adjacencies = keep_topk_percent_torch(torch.rand_like(adjacencies), percent)

        # Create dataset
        gnndataset = create_dataset(
            dfs,
            adjacencies,
            generate_features_target_multi_input,
            get_edge_index_features,
            num_timesteps_in,
            num_timesteps_out,
            GraphDataSet=[DynamicGraphTemporalSignal, StaticGraphTemporalSignal]
        )

        # Save dataset
        with open(full_path, "wb") as f_out:
            pickle.dump(gnndataset, f_out)

        print(f"Dataset saved successfully: {full_path}")

    return gnndataset, dfs



from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import butter, filtfilt
def apply_hp_filter_to_dataframe(df, lamb=1600):
    trend_dict = {}
    cycle_dict = {}
    
    for col in df.columns:
        trend, cycle = hpfilter(df[col], lamb=lamb)
        trend_dict[col] = trend
        cycle_dict[col] = cycle

    trend_df = pd.DataFrame.from_dict(trend_dict)
    cycle_df = pd.DataFrame.from_dict(cycle_dict)
    trend_df.index = df.index
    cycle_df.index = df.index

    return trend_df, cycle_df

def butter_lowpass_filter(df, cutoff, fs, order=4):

    data = df.values
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)  # Zero-phase filtering
    out_df = pd.DataFrame(y, index=df.index, columns=df.columns)
    return out_df

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                      # for CPU
    torch.cuda.manual_seed(seed)                 # for current GPU
    torch.cuda.manual_seed_all(seed)             # for multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepro_time_series_data(df,counties, best_counties, start_date, end_date, skip_dates, rolling_window=28):
    county_to_state = dict(zip(best_counties['countyfips'], best_counties['state']))

    df = df[df.index.isin(skip_dates)==False]

    df = df.loc[start_date:end_date]
    # pop norm cases
    df = df.copy()
    df = df.rolling(window=rolling_window).mean().dropna()
    df = df.T
    df['countyfips'] =df.index.astype('int')
    merged = pd.merge(df, counties, on ='countyfips', how='left')
    merged.index = merged.countyfips
    merged = merged[merged['countyfips'].isin(best_counties['countyfips'])]
    merged = merged.iloc[:,:-len(counties.columns)].T/merged['population'].astype('float')
    merged.index.name = None
    merged = merged*100000

    _, df_hp = apply_hp_filter_to_dataframe(merged)
    df_hp = df_hp + abs(min(df_hp.min()))
    df_hp_log = np.log(df_hp+1)
    merged_28 = df_hp
    #merged_28 = merged.rolling(window=14).mean().dropna()
    # 28 day rolling mean cases
    merged_28_log = np.log(merged_28+1)
    # diff
    merged_diff = merged_28.diff().rolling(window=14).mean().fillna(0)
    # std
    merged_28_std = merged_28.rolling(window=14).std().fillna(0)

    cutoff = 1/7  # allow trends >7 days
    fs = 1        # sampling rate = 1/day
    #df_filtered = butter_lowpass_filter_df(df, cutoff=1/14)

    butterworth = butter_lowpass_filter(merged_28, cutoff=cutoff, fs=fs)

    butterworth_log = np.log(butterworth+1)#.shift(14).fillna(0)
    butterworth_log_lag = butterworth_log.shift(-14).fillna(0)
    # 1. Exponential Moving Average (EMA)
    merged_28_ema = merged_28.ewm(span=14, adjust=False).mean()

    # 2. Gaussia'n Filter
    # df['Gaussian'] = gaussian_filter1d(df['signal'], sigma=3)

    # 3. Median Filter
    # # df['Median']' = median_filter(df['signal'], size=5)
    states_df = merged_28.copy()
    new_cols = []
    for col in states_df.columns:
        new_cols.append(county_to_state[col])
    states_df.columns = new_cols
    states_df.columns.name = 'state'
    states_df_grouped = states_df.groupby(level='state', axis=1).mean()
    merged_28_state_mean = merged_28.copy()
    for col in merged_28.columns:
        merged_28_state_mean[col] = states_df_grouped[county_to_state[col]]
 
    merged_pct_change = merged_28 - merged_28.shift(28)#merged_28.shift(42)#merged.ewm(span=14, adjust=False).mean()#((merged_28 - merged_28.rolling(14).mean()) / merged_28.rolling(14).std()).fillna(0)

    return merged_28, merged_28_log, merged_diff, butterworth, butterworth_log, butterworth_log_lag, merged_28_std, merged_28_ema,df_hp, df_hp_log ,merged_28_state_mean
