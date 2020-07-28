import numpy as np


class MiniMax:
    """
    Implementation of MiniMax with alpha-beta pruning.
    """

    def __init__(self, GameClass, heuristic_func, depth):
        """
        
        :param heuristic_func: Take's positions for the game and returns a heuristic approximation in [-1, 1]
        :param depth: 
        """
        self.GameClass = GameClass
        self.heuristic_func = heuristic_func
        self.depth = depth

    @staticmethod
    def from_network(GameClass, network, depth):
        network.initialize()

        def heuristic_func(state):
            _, evaluation = network.predict(state[np.newaxis, ...])[0]
            return evaluation
        return MiniMax(GameClass, heuristic_func, depth)

    @staticmethod
    def solver(GameClass):
        return MiniMax(GameClass, None, np.inf)

    def choose_move(self, position, return_heuristic=False):
        if self.GameClass.is_over(position):
            raise Exception('Game Finished!')

        is_maximizing = self.GameClass.is_player_1_turn(position)
        best_move = None
        best_heuristic = -np.inf if is_maximizing else np.inf

        for move in self.GameClass.get_possible_moves(position):
            heuristic = self.evaluate_position_recursive(move, self.depth - 1, not is_maximizing, best_heuristic)

            if (is_maximizing and heuristic > best_heuristic) or (not is_maximizing and heuristic < best_heuristic):
                best_heuristic = heuristic
                best_move = move
        return (best_move, best_heuristic) if return_heuristic else best_move

    def evaluate_position_recursive(self, position, depth, is_maximizing, value_to_beat):
        if self.GameClass.is_over(position):
            return self.GameClass.get_winner(position)

        if depth == 0:
            return self.heuristic_func(position)

        best_heuristic = -np.inf if is_maximizing else np.inf
        for child_position in self.GameClass.get_possible_moves(position):
            heuristic = self.evaluate_position_recursive(child_position, depth - 1, not is_maximizing, best_heuristic)

            if is_maximizing and heuristic > best_heuristic:
                if heuristic > value_to_beat:
                    # prune
                    return heuristic
                best_heuristic = heuristic
            if not is_maximizing and heuristic < best_heuristic:
                if heuristic < value_to_beat:
                    # prune
                    return heuristic
                best_heuristic = heuristic

        return best_heuristic


def main():
    from src.utils.active_game import ActiveGame as GameClass
    print(MiniMax.solver(GameClass).choose_move(GameClass.STARTING_STATE, return_heuristic=True))


if __name__ == '__main__':
    main()

