import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch import tensor
import matplotlib.pyplot as plt
import numpy as np
import random
from time import time
import math
from copy import deepcopy
import pandas as pd
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_seq_length = 120  # 2 hours past minutes
output_seq_length = 60  # Next 1 hour future minutes

lr = 0.00001
num_epochs = 2000
batch_size = 512
hidden_size = 32 # Reduced for stability
num_gru_layers = 1
grad_clip = 1.0
scheduled_sampling_decay = 10
dropout = 0.0  # Added for regularization

probabilistic = True
use_attention = True
target_indices = [0]

def inverse_sigmoid_decay(decay):
    def compute(index):
        return decay / (decay + math.exp(index / decay))
    return compute

calc_teacher_force_prob = inverse_sigmoid_decay(scheduled_sampling_decay)
print(f'At epoch {num_epochs} teacher force prob will be {calc_teacher_force_prob(num_epochs - 1)}')

device = torch.device(device)

class TensorNormalizer:
    def __init__(self, standardize=False):  # Min-max for robustness
        self.standardize = standardize
        self.center = None
        self.std = None
        self.mi = None
        self.range = None
    
    def _check(self, X, been_fit=False):
        assert len(X.shape) == 2
        if been_fit:
            if self.standardize: assert self.center is not None and self.std is not None
            else: assert self.mi is not None and self.range is not None
    def fit(self, X):
        self._check(X)
        if self.standardize:
            self.center = X.mean(axis=0)
            self.std = X.std(axis=0)
        else:
            self.mi = X.min(axis=0)[0]
            self.range = X.max(axis=0)[0] - self.mi
        return self
    
    def transform(self, X):
        self._check(X, been_fit=True)
        if self.standardize:
            return (X - self.center) / self.std
        else:
            return (X - self.mi) / self.range
    
    def fit_transform(self, X):
        self.fit(X)
        return self, self.transform(X)
    
    def inverse_transform(self, X_scaled):
        self._check(X_scaled, been_fit=True)
        if self.standardize:
            return (X_scaled * self.std) + self.center
        else:
            return (X_scaled * self.range) + self.mi
    
    def set_keep_columns(self, indices):
        if self.standardize:
            self.center = self.center[indices]
            self.std = self.std[indices]
        else:
            self.mi = self.mi[indices]
            self.range = self.range[indices]

def load_data(device):
    df = pd.read_csv('Modified_Timeseries_Temperature_and_Humidity.csv')
    timestamps = pd.to_datetime(df['Time steps'])
    
    # Minute-level: Compute sin/cos for daily cycle
    time_minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
    time_sin = np.sin(2 * np.pi * time_minutes / 1440)
    time_cos = np.cos(2 * np.pi * time_minutes / 1440)
    
    temperature = df['Temperature'].to_numpy()
    #features = np.stack([temperature, time_sin, time_cos], axis=1)  # (23181, 3)
    features = np.array([temperature]).reshape(-1, 1)
    df_array = np.expand_dims(features, 0)  # (1, 23181, 3)
    
    # Lag: period=59 for 60-min lag minus 1
    period = 59
    lag = np.array([temperature[i - period] for i in range(period, len(temperature))])  # (23122,)
    lag_time_series = np.expand_dims(lag, (0, 2))  # (1, 23122, 1)
    
    df_array = df_array[:, period:, :]  # (1, 23122, 3)
    df_array = np.concatenate((df_array, lag_time_series), 2)  # (1, 23122, 4)
    
    data = torch.tensor(df_array, dtype=torch.float).to(device)
    return data

data = load_data(device)
print(data.shape)

def split_data(data):
    num_timesteps = data.shape[1]
    train_ratio = 0.8  # 80% train for small data
    train_end_index = round(train_ratio * num_timesteps)
    val_end_index = round((train_ratio + 0.1) * num_timesteps)  # 10% val
    train_data = data[:, :train_end_index]
    val_data = data[:, train_end_index:val_end_index]
    test_data = data[:, val_end_index:]
    print(f"Splits: Train {train_data.shape[1]}, Val {val_data.shape[1]}, Test {test_data.shape[1]} timesteps")
    if val_data.shape[1] < input_seq_length + output_seq_length:
        print("Warning: Val too small - consider shorter sequences or more data")
    return train_data, val_data, test_data

data_splits = split_data(data)
print(data_splits[0].shape, data_splits[1].shape, data_splits[2].shape)

def create_sequences(data, input_seq_length, output_seq_length, target_indices):
    enc_inputs, dec_inputs, dec_targets, scalers = [], [], [], []
    # Check for small split
    num_possible = data.shape[1] - (input_seq_length + output_seq_length) + 1
    if num_possible <= 0:
        print(f"Skipping {data.shape[1]} timesteps split - too small for seqs")
        return {'enc_inputs': torch.tensor([]).unsqueeze(0), 'dec_inputs': torch.tensor([]).unsqueeze(0), 'dec_targets': torch.tensor([]).unsqueeze(0), 'scalers': np.array([])}
    
    # Loop over starting timesteps of the sequences
    for timestep in range(num_possible):
        # enc_inputs: (num time series, input seq len, num features)
        enc_inputs_at_t = deepcopy(data[:, timestep : timestep + input_seq_length, :])
        dec_at_t = deepcopy(data[:, timestep + input_seq_length - 1 : timestep + input_seq_length + output_seq_length, :])
        # dec_inputs: (num time series, output seq len, num features)
        dec_inputs_at_t = deepcopy(dec_at_t[:, :-1, :])
        # dec_targets: (num time series, output seq len, num targets)
        dec_targets_at_t = deepcopy(dec_at_t[:, 1:, target_indices])
        # Scale each time series separately
        all_ts_enc_inputs, all_ts_dec_inputs, all_ts_dec_targets, all_ts_scalers = [], [], [], []
        for ts_indx in range(enc_inputs_at_t.shape[0]):
            ts_scaler, ts_enc_inputs = TensorNormalizer(standardize=False).fit_transform(deepcopy(enc_inputs_at_t[ts_indx]))  # Min-max
            ts_dec_inputs = ts_scaler.transform(deepcopy(dec_inputs_at_t[ts_indx]))
            ts_scaler.set_keep_columns(target_indices)
            ts_dec_targets = ts_scaler.transform(deepcopy(dec_targets_at_t[ts_indx]))
            all_ts_enc_inputs.append(ts_enc_inputs); all_ts_dec_inputs.append(ts_dec_inputs)
            all_ts_dec_targets.append(ts_dec_targets); all_ts_scalers.append(ts_scaler)
        enc_inputs.append(torch.stack(all_ts_enc_inputs))
        dec_inputs.append(torch.stack(all_ts_dec_inputs))
        dec_targets.append(torch.stack(all_ts_dec_targets))
        scalers.append(np.stack(all_ts_scalers))
    enc_inputs = torch.stack(enc_inputs); dec_inputs = torch.stack(dec_inputs); 
    dec_targets = torch.stack(dec_targets); scalers = np.stack(scalers)
    return {'enc_inputs': enc_inputs, 'dec_inputs': dec_inputs, 'dec_targets': dec_targets, 'scalers': scalers}

data_splits = (create_sequences(data_splits[0], input_seq_length, output_seq_length, target_indices),
               create_sequences(data_splits[1], input_seq_length, output_seq_length, target_indices),
               create_sequences(data_splits[2], input_seq_length, output_seq_length, target_indices))
print(data_splits[0]['enc_inputs'].shape)

def reshape_data(data):
    for k, v in data.items():
        if k == 'scalers':
            if len(v.shape) > 0:
                data[k] = v.reshape(-1)
            else:
                data[k] = np.array([])
        else:
            if v.numel() == 0 or len(v.shape) < 4:
                print(f"Warning: Empty tensor for {k} - skipping reshape")
                seq_len = output_seq_length if k == 'dec_inputs' or k == 'dec_targets' else input_seq_length
                features = data['enc_inputs'].shape[-1] if 'enc_inputs' in data and data['enc_inputs'].numel() > 0 else 4
                data[k] = torch.zeros(0, seq_len, features)
            else:
                data[k] = v.reshape(-1, v.shape[2], v.shape[3])
    return data

train_data, val_data, test_data = (reshape_data(data_splits[0]), reshape_data(data_splits[1]), reshape_data(data_splits[2]))

print(train_data['enc_inputs'].shape, val_data['enc_inputs'].shape, test_data['enc_inputs'].shape)
print(train_data['enc_inputs'].shape, train_data['dec_inputs'].shape, train_data['dec_targets'].shape, train_data['scalers'].shape)

def layer_init(layer, w_scale=1.0, is_sigma=False):
    nn.init.kaiming_uniform_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    if is_sigma:
        nn.init.constant_(layer.bias.data, math.log(0.1))  # Start sigma ~0.1
    else:
        nn.init.constant_(layer.bias.data, 0.)
    return layer

class Encoder(nn.Module):
    def __init__(self, enc_feature_size, hidden_size, num_gru_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(enc_feature_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        
    def forward(self, inputs):
        output, hidden = self.gru(inputs)
        return output, hidden

class DecoderBase(nn.Module):
    def __init__(self, device, dec_target_size, target_indices, dist_size, probabilistic):
        super().__init__()
        self.device = device
        self.target_indices = target_indices
        self.target_size = dec_target_size
        self.dist_size = dist_size
        self.probabilistic = probabilistic
    
    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        raise NotImplementedError()
    
    def forward(self, inputs, hidden, enc_outputs, teacher_force_prob=None):
        batch_size, dec_output_seq_length, _ = inputs.shape
        outputs = torch.zeros(batch_size, dec_output_seq_length, self.target_size, self.dist_size, dtype=torch.float).to(self.device)
        curr_input = inputs[:, 0:1, :]
        for t in range(dec_output_seq_length):
            dec_output, hidden = self.run_single_recurrent_step(curr_input, hidden, enc_outputs)
            dec_output = torch.nan_to_num(dec_output, nan=0.0)  # Guard NaN
            outputs[:, t:t+1, :, :] = dec_output
            dec_output = Seq2Seq.sample_from_output(dec_output)
            teacher_force = random.random() < teacher_force_prob if teacher_force_prob is not None else False
            curr_input = inputs[:, t:t+1, :].clone()
            if not teacher_force:
                curr_input[:, :, self.target_indices] = dec_output
        outputs = torch.nan_to_num(outputs, nan=0.0)  # Final guard
        return outputs

class DecoderVanilla(DecoderBase):
    def __init__(self, dec_feature_size, dec_target_size, hidden_size, 
                 num_gru_layers, target_indices, dropout, dist_size,
                 probabilistic, device):
        super().__init__(device, dec_target_size, target_indices, dist_size, probabilistic)
        self.gru = nn.GRU(dec_feature_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        self.out = layer_init(nn.Linear(hidden_size + dec_feature_size, dec_target_size * dist_size), is_sigma=True)
    
    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        output, hidden = self.gru(inputs, hidden)
        output = self.out(torch.cat((output, inputs), dim=2))
        output = output.reshape(output.shape[0], output.shape[1], self.target_size, self.dist_size)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size, num_gru_layers):
        super().__init__()
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden_final_layer, encoder_outputs):
        hidden = decoder_hidden_final_layer.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        weightings = F.softmax(attention, dim=1)
        return weightings

class DecoderWithAttention(DecoderBase):
    def __init__(self, dec_feature_size, dec_target_size, hidden_size, 
                 num_gru_layers, target_indices, dropout, dist_size,
                 probabilistic, device):
        super().__init__(device, dec_target_size, target_indices, dist_size, probabilistic)
        self.attention_model = Attention(hidden_size, num_gru_layers)
        self.gru = nn.GRU(dec_feature_size + hidden_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        self.out = layer_init(nn.Linear(hidden_size + hidden_size + dec_feature_size, dec_target_size * dist_size), is_sigma=True)

    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        weightings = self.attention_model(hidden[-1], enc_outputs)
        weighted_sum = torch.bmm(weightings.unsqueeze(1), enc_outputs)
        output, hidden = self.gru(torch.cat((inputs, weighted_sum), dim=2), hidden)
        output = self.out(torch.cat((output, weighted_sum, inputs), dim=2))
        output = output.reshape(output.shape[0], output.shape[1], self.target_size, self.dist_size)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, lr, grad_clip, probabilistic):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.opt = torch.optim.Adam(self.parameters(), lr)
        self.loss_func = nn.GaussianNLLLoss() if probabilistic else nn.L1Loss()
        self.grad_clip = grad_clip
        self.probabilistic = probabilistic
    
    @staticmethod
    def compute_smape(prediction, target):
        return torch.mean(torch.abs(prediction - target) / ((torch.abs(target) + torch.abs(prediction)) / 2. + 1e-8)) * 100.
    
    @staticmethod
    def get_dist_params(output):
        if torch.isnan(output).any():
            print("NaN in output - using fallback")
            mu = torch.zeros_like(output[:,:,:,0])
            sigma = torch.ones_like(output[:,:,:,0]) * 0.1
            return mu, sigma
        mu = output[:, :, :, 0]
        sigma = F.softplus(output[:, :, :, 1])
        sigma = torch.clamp(sigma, min=1e-6)
        return mu, sigma
    
    @staticmethod
    def sample_from_output(output):
        if output.shape[-1] > 1:
            mu, sigma = Seq2Seq.get_dist_params(output)
            return torch.normal(mu, sigma)
        return output.squeeze(-1)
    
    def forward(self, enc_inputs, dec_inputs, teacher_force_prob=None):
        enc_outputs, hidden = self.encoder(enc_inputs)
        outputs = self.decoder(dec_inputs, hidden, enc_outputs, teacher_force_prob)
        outputs = torch.nan_to_num(outputs, nan=0.0)
        return outputs

    def compute_loss(self, prediction, target, override_func=None):
        if self.probabilistic:
            mu, sigma = Seq2Seq.get_dist_params(prediction)
            var = sigma ** 2
            loss = self.loss_func(mu, target, var)
        else:
            loss = self.loss_func(prediction.squeeze(-1), target)
        return loss if self.training else loss.item()
    
    def optimize(self, prediction, target):
        self.opt.zero_grad()
        loss = self.compute_loss(prediction, target)
        if torch.isnan(loss):
            print("NaN loss - skipping step")
            return 0.0
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.opt.step()
        return loss.item()

def batch_generator(data, batch_size, unscale=False):
    if 'enc_inputs' not in data or data['enc_inputs'].shape[0] == 0:
        return  # No yield if empty
    enc_inputs, dec_inputs, dec_targets, scalers = \
        data['enc_inputs'], data['dec_inputs'], data['dec_targets'], data['scalers']
    indices = torch.randperm(enc_inputs.shape[0])
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_enc_inputs = enc_inputs[batch_indices]
        batch_dec_inputs = dec_inputs[batch_indices]
        batch_dec_targets = dec_targets[batch_indices]
        batch_scalers = None
        if unscale:
            batch_scalers = scalers[batch_indices]
            if isinstance(batch_scalers, TensorNormalizer): batch_scalers = np.array([batch_scalers])
        if batch_enc_inputs.shape[0] < batch_size:
            break
        yield batch_enc_inputs, batch_dec_inputs, batch_dec_targets, batch_scalers

def train(model, train_data, batch_size, teacher_force_prob):
    model.train()
    epoch_loss = 0.
    num_batches = 0
    for batch_enc_inputs, batch_dec_inputs, batch_dec_targets, _ in batch_generator(train_data, batch_size):
        output = model(batch_enc_inputs, batch_dec_inputs, teacher_force_prob)
        loss = model.optimize(output, batch_dec_targets)
        epoch_loss += loss
        num_batches += 1
    if num_batches == 0:
        print("No batches in train")
        return float('inf')
    return epoch_loss / num_batches

def evaluate(model, val_data, batch_size):
    model.eval()
    epoch_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for batch_enc_inputs, batch_dec_inputs, batch_dec_targets, _ in batch_generator(val_data, batch_size):
            output = model(batch_enc_inputs, batch_dec_inputs)
            loss = model.compute_loss(output, batch_dec_targets)
            epoch_loss += loss
            num_batches += 1
    if num_batches == 0:
        print("No batches in eval")
        return float('inf')
    return epoch_loss / num_batches

dist_size = 2 if probabilistic else 1
enc_feature_size = train_data['enc_inputs'].shape[-1]
dec_feature_size = train_data['dec_inputs'].shape[-1]
dec_target_size = train_data['dec_targets'].shape[-1]

encoder = Encoder(enc_feature_size, hidden_size, num_gru_layers, dropout)
decoder_args = (dec_feature_size, dec_target_size, hidden_size, num_gru_layers, target_indices, dropout, dist_size, probabilistic, device)
decoder = DecoderWithAttention(*decoder_args) if use_attention else DecoderVanilla(*decoder_args)
seq2seq = Seq2Seq(encoder, decoder, lr, grad_clip, probabilistic).to(device)

best_val, best_model = float('inf'), None
for epoch in range(num_epochs):
    start_t = time()
    teacher_force_prob = calc_teacher_force_prob(epoch)
    train_loss = train(seq2seq, train_data, batch_size, teacher_force_prob)
    val_loss = evaluate(seq2seq, val_data, batch_size)
    new_best_val = False
    if val_loss < best_val:
        new_best_val = True
        best_val = val_loss
        best_model = deepcopy(seq2seq)
        torch.save(best_model.state_dict(), f'best_seq2seq_model_epoch_{epoch+1}.pth')
    print(f'Epoch {epoch+1} => Train loss: {train_loss:.5f}, Val: {val_loss:.5f}, Teach: {teacher_force_prob:.2f}, Took {(time() - start_t):.1f} s{"      (NEW BEST)" if new_best_val else ""}')

# Test Evaluation
data_to_eval = test_data
best_model.eval()

mean_losses, norm_losses, repeat_losses, trained_model_losses = [], [], [], []
for batch_enc_inputs, batch_dec_inputs, batch_dec_targets, _ in batch_generator(data_to_eval, 32):
    mean_baseline_preds = torch.repeat_interleave(batch_enc_inputs[:, :, target_indices].mean(axis=1, keepdims=True), data_to_eval['dec_targets'].shape[1], 1).unsqueeze(-1)
    if probabilistic:
        stds = torch.zeros(mean_baseline_preds.shape[:-1] + (1,), dtype=torch.float).to(device)
        mean_baseline_preds = torch.cat((mean_baseline_preds, stds), dim=3)
    mean_loss = best_model.compute_loss(mean_baseline_preds, batch_dec_targets)

    test_inputs = batch_enc_inputs[:, :, target_indices]
    test_inputs_mean = torch.repeat_interleave(test_inputs.mean(axis=1, keepdims=True), batch_dec_targets.shape[1], 1).unsqueeze(-1)
    test_inputs_std = torch.repeat_interleave(test_inputs.std(axis=1, keepdims=True), batch_dec_targets.shape[1], 1).unsqueeze(-1)
    if probabilistic:
        norm_baseline_preds = torch.cat((test_inputs_mean, test_inputs_std), dim=3)
    else:
        norm_baseline_preds = torch.normal(test_inputs_mean, test_inputs_std)
    norm_loss = best_model.compute_loss(norm_baseline_preds, batch_dec_targets)

    repeat_baseline_preds = torch.repeat_interleave(batch_enc_inputs[:, -1:, target_indices], batch_dec_targets.shape[1], 1).unsqueeze(-1)
    if probabilistic:
        stds = torch.zeros(repeat_baseline_preds.shape[:-1] + (1,), dtype=torch.float).to(device)
        repeat_baseline_preds = torch.cat((repeat_baseline_preds, stds), dim=3)
    repeat_loss = best_model.compute_loss(repeat_baseline_preds, batch_dec_targets)

    outputs = best_model(batch_enc_inputs, batch_dec_inputs)
    trained_model_loss = best_model.compute_loss(outputs, batch_dec_targets)

    mean_losses.append(mean_loss); norm_losses.append(norm_loss)
    repeat_losses.append(repeat_loss); trained_model_losses.append(trained_model_loss)
print(np.mean(mean_losses), np.mean(norm_losses), np.mean(repeat_losses), np.mean(trained_model_losses))

# Visualize (adjusted for minutes, input=120)
target_to_vis = 0
num_vis = min(10, test_data['enc_inputs'].shape[0]) if test_data['enc_inputs'].numel() > 0 else 1
num_rollouts = 50 if probabilistic else 1

best_model.eval()

with torch.no_grad():
    batch_enc_inputs, batch_dec_inputs, batch_dec_targets, scalers = next(batch_generator(data_to_eval, num_vis, unscale=True))

    outputs = []
    for r in range(num_rollouts):
        outputs.append(Seq2Seq.sample_from_output(best_model(batch_enc_inputs, batch_dec_inputs)))
    outputs = torch.stack(outputs, dim=1)

for indx in range(batch_enc_inputs.shape[0]):
    scaler = scalers[indx]
    sample_enc_inputs = scaler.inverse_transform(batch_enc_inputs[indx])[:, target_to_vis].cpu().numpy().tolist()
    sample_dec_targets = scaler.inverse_transform(batch_dec_targets[indx])[:, target_to_vis].cpu().numpy().tolist()
    output_rollouts = np.array([scaler.inverse_transform(out)[:, target_to_vis].cpu().numpy().tolist() for out in outputs[indx]])
    
    plt.figure(figsize=(10,5))
    enc_x = list(range(input_seq_length))  # Past minutes
    future_x = list(range(input_seq_length, input_seq_length + output_seq_length))  # Future minutes
    plt.plot(enc_x, sample_enc_inputs, label='Past 2 Hours')
    plt.plot(future_x, sample_dec_targets, 'o-', label='Actual Next Hour')
    plt.plot(future_x, np.median(output_rollouts, axis=0), 'r-', label='Median Pred')
    plt.fill_between(future_x, np.quantile(output_rollouts, 0.05, axis=0), np.quantile(output_rollouts, 0.95, axis=0), alpha=0.3, label='90% CI')
    plt.legend()
    plt.xlabel('Minutes')
    plt.ylabel('Temperature')
    plt.title(f'Sequence {indx+1}: 2-Hour Past + 1-Hour Forecast')
    plt.show()