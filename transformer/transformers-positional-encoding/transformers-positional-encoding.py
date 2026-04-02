import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    pos = np.arange(seq_length)[:, np.newaxis]          # (seq_length, 1)
    i = np.arange(d_model)[np.newaxis, :]               # (1, d_model)

    # вычисляем делитель
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates                         # (seq_length, d_model)

    pe = np.zeros((seq_length, d_model))

    # четные индексы → sin
    pe[:, 0::2] = np.sin(angles[:, 0::2])

    # нечетные индексы → cos
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe