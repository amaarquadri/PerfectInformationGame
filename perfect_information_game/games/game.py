from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Sequence, Tuple, Literal


# noinspection PyUnresolvedReferences
class Game(ABC):
    # REQUIRED CLASS VARIABLES
    # STARTING_STATE = ndarray
    # STATE_SHAPE = STARTING_STATE.shape
    # ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE
    # BOARD_SHAPE = (ROWS, COLUMNS)
    # MOVE_SHAPE = ()
    # REPRESENTATION_LETTERS = []
    # REPRESENTATION_FILES = []
    # CLICKS_PER_MOVE = int

    # INSTANCE FUNCTIONS

    def __init__(self, state: Optional[np.ndarray] = None):
        self.state = np.copy(state if state is not None else self.STARTING_STATE)

    def get_state(self) -> np.ndarray:
        """
        The result will be a matrix with shape (n, m, k) where the board has dimensions of nxm and
        k is the number of features.
        For example, in checkers n=m=8 and k=5 where the features are
        [red pieces, red kings, black pieces, black kings, is white to move]
        In chess (ignoring 50 move rule and threefold repetition) n=m=8, k=14 and features=[white king, white queens,
        white rooks, white bishops, white knights, white pawns, black king, black queens, black rooks, black bishops,
        black knights, black pawns, special move booleans, is white to move].


        :return: A numpy matrix representation of the current state of this Game.
        """
        return self.state

    def set_state(self, state: np.ndarray):
        """
        Set the state of this Game.

        :param state: The state in the format specified by get_ML_representation
        """
        self.state = np.copy(state)

    def reset_game(self):
        self.set_state(self.STARTING_STATE)

    @classmethod
    def to_string(cls, state: np.ndarray) -> str:
        """
        The board should be drawn in book-reading fashion. i.e. The first index represents the row from top to bottom
        and the second index represents the column from left to right.
        """
        representation = np.full(cls.BOARD_SHAPE, '_', dtype=str)
        for i, letter in enumerate(cls.REPRESENTATION_LETTERS):
            representation[state[:, :, i] == 1] = letter

        return '\n'.join([' '.join(representation[i, :]) for i in range(representation.shape[0])])

    def __str__(self):
        return self.to_string(self.state)

    @abstractmethod
    def perform_user_move(self, clicks: Sequence[Tuple[int, int]]):
        """
        Performs the move specified by the clicks, on the specified state and returns the resulting state.
        """
        pass

    # CLASS LEVEL GAME SPECIFIC ABSTRACT FUNCTIONS

    @classmethod
    def is_player_1_turn(cls, state: np.ndarray) -> bool:
        return np.all(state[..., -1])

    @classmethod
    @abstractmethod
    def get_possible_moves(cls, state: np.ndarray) -> Sequence[np.ndarray]:
        """
        The order of the returned states must be sorted based on the flattened versions of MOVE_SHAPE.

        :return: A list of all possible board states that could result from the given state.
        """
        pass

    @classmethod
    @abstractmethod
    def get_legal_moves(cls, state: np.ndarray) -> np.ndarray:
        """
        :return: A numpy array with shape=MOVE_SHAPE where False corresponds to an illegal move
                 and True corresponds to a legal move.
        """
        pass

    @classmethod
    @abstractmethod
    def is_over(cls, state: np.ndarray) -> bool:
        pass

    @classmethod
    @abstractmethod
    def get_winner(cls, state: np.ndarray) -> Literal[-1, 0, 1]:
        """
        :return: 1 if player 1 won, 0 if draw, -1 if player 2 won.
        """
        pass

    @classmethod
    def get_img_index_representation(cls, state: np.ndarray) -> np.ndarray:
        """
        The result will be a matrix with shape (n, m) and dtype=int. Each element will be an integer corresponding to
        which image to use to represent that square. The mapping from indices to file names should be provided in a
        class-level constant list called REPRESENTATION_FILES.

        :return: A numpy matrix indicating which images to use for each square in the grid.
        """
        representation = np.zeros(cls.BOARD_SHAPE, dtype=int)  # squares where every feature is 0 default to the 0th image
        for i in range(len(cls.REPRESENTATION_FILES) - 1):  # -1 to exclude squares where every feature is 0
            representation[state[:, :, i] == 1] = i + 1
        return representation

    @classmethod
    def needs_checkerboard(cls) -> bool:
        """
        If this game needs the UI to draw a checkerboard below the pieces, then this should return True.
        """
        return False

    @classmethod
    def get_ruleset(cls) -> Optional[str]:
        """
        Returns the ruleset that the game is configured to be using. For example, the board size.
        If only 1 ruleset is configured for the game, then None will be returned.
        """
        return None

    @classmethod
    def null_move(cls, state: np.ndarray) -> np.ndarray:
        move = np.copy(state)
        move[..., -1] = 0 if cls.is_player_1_turn(state) else 1
        return move

    @staticmethod
    def is_board_full(state: np.ndarray) -> bool:
        combined_board = state[..., 0]
        # loop over all indices of the last axis except for -1, which corresponds to turn information
        for i in range(1, state.shape[-1] - 1):
            combined_board = np.logical_or(combined_board, state[..., i])
        return np.all(combined_board == 1)

    @classmethod
    def is_valid(cls, i: int, j: int) -> bool:
        return 0 <= i < cls.ROWS and 0 <= j < cls.COLUMNS

    @classmethod
    def heuristic(cls, state: np.ndarray) -> float:
        raise NotImplemented()
