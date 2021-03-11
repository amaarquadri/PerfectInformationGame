import numpy as np


class EmptyTablebaseManager:
    """
    Each tablebase has a descriptor, in a form such as KQkn (king and queen vs king and knight).
    The tablebases are stored in {get_training_path(GameClass)}/tablebases/{descriptor}.pickle

    Each tablebase consists of a dictionary that maps board_bytes to move_bytes.
    move_bytes can be converted to and from this tuple: (outcome, start_i, start_j, target_i, target_j, distance).
    Only the symmetric variants of each position are stored in the tablebases.
    """
    def __init__(self, GameClass):
        self.GameClass = GameClass

    def update_tablebase_list(self):
        pass

    def ensure_loaded(self, descriptor):
        raise NotImplementedError(f'No tablebase available for descriptor = {descriptor}')

    def query_position(self, state, outcome_only=False):
        if self.GameClass.is_over(state):
            return (self.GameClass.get_winner(state), 0) if outcome_only \
                else (None, self.GameClass.get_winner(state), 0)

        return (np.nan, np.nan) if outcome_only else (None, np.nan, np.nan)

    def get_random_endgame(self, descriptor, condition=None):
        raise NotImplementedError(f'No tablebase available for descriptor = {descriptor}')

    def get_random_endgame_with_outcome(self, descriptor, outcome):
        raise NotImplementedError(f'No tablebase available for descriptor = {descriptor}')
