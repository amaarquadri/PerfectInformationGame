from abc import ABC, abstractmethod
import numpy as np


class AbstractNode(ABC):
    def __init__(self, position, parent, GameClass, tablebase_manager, c=np.sqrt(2), verbose=False):
        """

        :param position:
        :param parent:
        :param GameClass:
        :param tablebase_manager: The tablebase manager.
                                  If no tablebases are implemented, then use an EmptyTablebaseManager.
        :param c:
        :param verbose:
        """
        self.position = position
        self.parent = parent
        self.GameClass = GameClass
        self.c = c
        self.tablebase_manager = tablebase_manager
        self.tablebase_results = tablebase_manager.query_position(position)
        self.fully_expanded = not np.isnan(self.tablebase_results[1])
        self.is_maximizing = GameClass.is_player_1_turn(position)
        self.children = None
        self.verbose = verbose

    @abstractmethod
    def get_evaluation(self):
        pass

    @abstractmethod
    def count_expansions(self):
        pass

    @abstractmethod
    def ensure_children(self):
        pass

    @abstractmethod
    def get_puct_heuristic_for_child(self, i):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def set_fully_expanded(self, minimax_evaluation):
        pass

    def choose_best_node(self, return_probability_distribution=False, optimal=False):
        distribution = []

        optimal_value = 1 if self.is_maximizing else -1
        if self.fully_expanded:
            if self.verbose:
                if self.get_evaluation() == optimal_value:
                    print('I\'m going to win')
                elif self.get_evaluation() == 0:
                    print('It\'s a draw')
                else:
                    print('I resign')

            for child in self.children:
                # only consider children that result in the optimal outcome
                if child.fully_expanded and child.get_evaluation() == self.get_evaluation():
                    # TODO: when losing, consider the number of ways the opponent can win in response to a move
                    depth_to_endgame = child.depth_to_end_game()
                    # if we are winning, weight smaller depths much more strongly by using e^-x
                    # if we are losing or drawing, weight larger depths much more strongly by using e^x
                    relative_probability = np.exp(-depth_to_endgame if self.get_evaluation() == optimal_value else
                                                  depth_to_endgame)
                    distribution.append(relative_probability)
                else:
                    distribution.append(0)
        else:
            for child in self.children:
                if not child.fully_expanded:
                    distribution.append(child.count_expansions())
                elif child.get_evaluation() == -optimal_value:
                    # this move is guaranteed to lose
                    distribution.append(0)
                else:
                    # use the self.heuristic as a proxy for the chance of winning the game
                    # the greater the perceived chance of winning the less appealing a draw is, and vice versa
                    winning_chance = (self.get_evaluation() * optimal_value) / 2 + 0.5
                    distribution.append(self.count_expansions() * (1 - winning_chance))

        distribution = np.array(distribution) / sum(distribution) if sum(distribution) > 0 else \
            np.full_like(distribution, 1 / len(distribution), dtype=float)
        idx = np.argmax(distribution) if optimal else np.random.choice(np.arange(len(distribution)), p=distribution)
        best_child = self.children[idx]
        return (best_child, distribution) if return_probability_distribution else best_child

    def choose_expansion_node(self):
        # TODO: continue tree search in case the user makes a mistake and the game continues
        if self.fully_expanded:
            return None

        if self.count_expansions() == 0:
            return self

        self.ensure_children()
        best_heuristic = -np.inf if self.is_maximizing else np.inf
        best_child = None
        for i, child in enumerate(self.children):
            # don't bother exploring fully expanded children
            if child.fully_expanded:
                optimal_value = 1 if self.is_maximizing else -1
                # If this child is already optimal, then self is fully expanded and there is no point searching further
                if child.get_evaluation() == optimal_value:
                    self.set_fully_expanded(optimal_value)
                    return self.parent.choose_expansion_node() if self.parent is not None else None
                # continue searching other children, there may be another child that is more optimal
                continue

            # check puct heuristic before calling child.get_evaluation() because it may result in division by 0
            puct_heuristic = self.get_puct_heuristic_for_child(i)
            if np.isinf(puct_heuristic):
                return child

            if self.is_maximizing:
                combined_heuristic = child.get_evaluation() + puct_heuristic
                if combined_heuristic > best_heuristic:
                    best_heuristic = combined_heuristic
                    best_child = child
            else:
                combined_heuristic = child.get_evaluation() - puct_heuristic
                if combined_heuristic < best_heuristic:
                    best_heuristic = combined_heuristic
                    best_child = child

        # if nothing was found because all children are fully expanded
        if best_child is None:
            if self.verbose and not self.fully_expanded and self.parent is None:
                print('Fully expanded tree!')

            minimax_evaluation = max([child.get_evaluation() for child in self.children]) if self.is_maximizing \
                else min([child.get_evaluation() for child in self.children])
            self.set_fully_expanded(minimax_evaluation)
            # this node is now fully expanded, so ask the parent to try to choose again
            # if no parent is available (i.e. this is the root node) then the entire search tree has been expanded
            return self.parent.choose_expansion_node() if self.parent is not None else None

        return best_child.choose_expansion_node()

    def depth_to_end_game(self):
        if not self.fully_expanded:
            raise Exception('Node not fully expanded!')

        if self.children is None:
            return 0

        optimal_value = 1 if self.is_maximizing else -1
        if self.get_evaluation() == optimal_value:
            # if we are winning, win as fast as possible
            return 1 + min(child.depth_to_end_game() for child in self.children
                           if child.fully_expanded and child.get_evaluation() == self.get_evaluation())
        else:
            # if we are losing or it is a draw, lose as slow as possible
            return 1 + max(child.depth_to_end_game() for child in self.children
                           if child.fully_expanded and child.get_evaluation() == self.get_evaluation())
