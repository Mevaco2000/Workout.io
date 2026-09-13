from __future__ import annotations

import numpy as np
import torch
from torch import nn


def validate_position_normalization_mode(mode: str) -> str:
    return mode.strip().lower()


def normalize_xy_positions(sequence: np.ndarray, mode: str) -> np.ndarray:
    normalized_mode = validate_position_normalization_mode(mode)
    result = np.asarray(sequence, dtype=np.float32)
    if normalized_mode != "bbox":
        return result

    result = result.copy()
    coordinates = result[..., :2]
    if result.shape[-1] >= 3:
        valid_mask = result[..., 2] > 0.0
    else:
        valid_mask = np.any(np.abs(coordinates) > 1e-6, axis=-1)

    for frame_index in range(result.shape[0]):
        frame_mask = valid_mask[frame_index]
        if not np.any(frame_mask):
            continue

        visible_points = coordinates[frame_index, frame_mask]
        min_xy = visible_points.min(axis=0)
        max_xy = visible_points.max(axis=0)
        center_xy = (min_xy + max_xy) / 2.0
        scale = float(np.max(max_xy - min_xy))
        if scale < 1e-6:
            scale = 1.0

        coordinates[frame_index, frame_mask] = (visible_points - center_xy) / scale
        coordinates[frame_index, ~frame_mask] = 0.0

    return result


def apply_standardization(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    view_shape = (1,) * (features.ndim - 1) + (features.shape[-1],)
    return ((features - mean.reshape(view_shape)) / std.reshape(view_shape)).astype(np.float32)


class MLPClassifier(nn.Module):
    def __init__(self, sequence_length: int, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        input_dim = sequence_length * input_size
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(max(num_layers, 1)):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.ReLU())
            current_dim = hidden_size
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.reshape(inputs.shape[0], -1))


class CNN1DClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_size
        for _ in range(max(num_layers, 1)):
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs.transpose(1, 2)
        encoded = self.encoder(features)
        pooled = self.pool(encoded).squeeze(-1)
        return self.classifier(pooled)


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(inputs)
        return self.classifier(outputs[:, -1, :])


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        return self.classifier(outputs[:, -1, :])


def _resolve_attention_heads(hidden_size: int) -> int:
    for num_heads in (8, 6, 4, 3, 2):
        if hidden_size % num_heads == 0:
            return num_heads
    return 1


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, sequence_length: int, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=_resolve_attention_heads(hidden_size),
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(num_layers, 1))
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(inputs) + self.position_embedding[:, : inputs.shape[1], :]
        encoded = self.encoder(encoded)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


def _build_chain_adjacency(num_nodes: int) -> torch.Tensor:
    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    for node_index in range(num_nodes - 1):
        adjacency[node_index, node_index + 1] = 1.0
        adjacency[node_index + 1, node_index] = 1.0
    adjacency = adjacency / adjacency.sum(dim=1, keepdim=True)
    return adjacency


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = self.residual(inputs)
        outputs = torch.einsum("nctv,vw->nctw", inputs, adjacency)
        outputs = self.spatial(outputs)
        outputs = self.activation(outputs)
        outputs = self.temporal(outputs)
        return self.activation(outputs + residual)


class STGCNClassifier(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        adjacency = _build_chain_adjacency(num_nodes)
        self.adjacency = nn.Parameter(adjacency)
        blocks: list[STGCNBlock] = []
        input_channels = in_channels
        for _ in range(max(num_layers, 1)):
            blocks.append(STGCNBlock(input_channels, hidden_size))
            input_channels = hidden_size
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.permute(0, 3, 1, 2)
        adjacency = torch.softmax(self.adjacency, dim=-1)
        for block in self.blocks:
            outputs = block(outputs, adjacency)
        pooled = outputs.mean(dim=(2, 3))
        return self.classifier(pooled)