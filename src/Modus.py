from enum import Enum


class FilterModus(Enum):
    KALMAN_FILTER = 1
    DISTRIBUTED_KALMAN_FILTER = 2
    FEDERATED_KALMAN_FILTER = 3
