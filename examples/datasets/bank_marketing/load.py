import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_bank_marketing(num_partitions: int) -> None:
    """Loads the Bank Marketing dataset and splits it into the specified number of partitions.

    Args:
        num_partitions: The number of partitions to create. Must be >= 2.
    """
    if num_partitions < 2:
        raise ValueError("num_partitions must be at least 2.")

    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    data_len = len(X)
    partition_size = data_len // num_partitions
    remainder = data_len % num_partitions

    start_index = 0
    for i in range(1, num_partitions + 1):
        end_index = start_index + partition_size
        if i <= remainder:
            end_index += 1

        X_part = pd.DataFrame(X[start_index:end_index])
        y_part = pd.DataFrame(y[start_index:end_index])

        X_part.to_csv(f"X_train_{i}.csv", index=False)
        y_part.to_csv(f"y_train_{i}.csv", index=False)

        start_index = end_index
