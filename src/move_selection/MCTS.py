import numpy as np
from multiprocessing import Process, Pipe, Pool
from time import time


class AsyncMCTS:
    """
    Implementation of Monte Carlo Tree Search that uses the other player's time to continue thinking.
    This is achieved using multiprocessing, and a Pipe for transferring data to and from the worker process.
    """
    def __init__(self, GameClass, position, time_limit=3, c=np.sqrt(2), threads=7):
        self.parent_pipe, worker_pipe = Pipe()
        self.worker_process = Process(target=self.loop_func,
                                      args=(GameClass, position, time_limit, c, threads, worker_pipe))

    def start(self):
        self.worker_process.start()

    def choose_move(self, user_chosen_position):
        """
        Instructs the worker thread that the user has chosen the move specified by the given position.
        The worker thread will then continue thinking for time_limit, and then return its chosen move.

        :param user_chosen_position: The board position resulting from the user's move.
        :return: The move chosen by monte carlo tree search.
        """
        self.parent_pipe.send(user_chosen_position)
        return self.parent_pipe.recv()

    def terminate(self):
        self.worker_process.terminate()
        self.worker_process.join()

    @staticmethod
    def loop_func(GameClass, position, time_limit, c, threads, worker_pipe):
        root = Node(position, None, GameClass)
        pool = Pool(threads) if threads > 1 else None

        while True:
            root.choose_rollout_node(c).rollout(threads, pool)

            if root.children is not None and worker_pipe.poll():
                user_chosen_position = worker_pipe.recv()
                for child in root.children:
                    if np.all(child.position == user_chosen_position):
                        root = child
                        root.parent = None
                        break
                else:
                    print(user_chosen_position)
                    raise Exception('Invalid user chosen move!')

                start_time = time()
                while time() - start_time < time_limit:
                    root.choose_rollout_node(c).rollout(threads, pool)

                root = root.choose_best_node()
                root.parent = None
                worker_pipe.send(root.position)


class MCTS:
    """
    Implementation of Monte Carlo Tree Search
    https://www.youtube.com/watch?v=UXW2yZndl7U
    """
    def __init__(self, GameClass, c=np.sqrt(2), threads=7):
        self.GameClass = GameClass
        self.c = c
        self.threads = threads
        self.pool = Pool(threads) if threads > 1 else None

    def choose_move(self, position, time_limit=10):
        if self.GameClass.is_over(position):
            raise Exception('Game Finished!')

        root = Node(position, parent=None, GameClass=self.GameClass)
        start_time = time()
        while time() - start_time < time_limit:
            root.choose_rollout_node(self.c).rollout(self.threads, self.pool)

        best_child = root.choose_best_node()
        # TODO: contemplate best_child while user is making their move
        return best_child.position


class Node:
    def __init__(self, position, parent, GameClass):
        self.position = position
        self.parent = parent
        self.GameClass = GameClass
        self.is_maximizing = GameClass.is_player_1_turn(position)

        self.children = None
        self.rollout_sum = 0
        self.rollout_count = 0

    def ensure_children(self):
        if self.children is None:
            self.children = [Node(move, self, self.GameClass) for move in
                             self.GameClass.get_possible_moves(self.position)]

    def choose_best_node(self):
        best_heuristic = -np.inf if self.is_maximizing else np.inf
        best_child = None
        for child in self.children:
            if child.rollout_count > 0:
                child_heuristic = child.rollout_sum / child.rollout_count
                if (self.is_maximizing and child_heuristic > best_heuristic) or \
                        (not self.is_maximizing and child_heuristic < best_heuristic):
                    best_heuristic = child_heuristic
                    best_child = child
        return best_child

    def choose_rollout_node(self, c=np.sqrt(2)):
        if self.rollout_count == 0:
            return self
        
        self.ensure_children()
        best_heuristic = -np.inf if self.is_maximizing else np.inf
        best_child = None
        for child in self.children:
            # any children that have not been visited will have an exploration term of infinity, so we can just choose
            # the first such child node
            if child.rollout_count == 0:
                return child

            child_heuristic = child.rollout_sum / child.rollout_count
            exploration_term = c * np.sqrt(np.log(self.rollout_count) / child.rollout_count)
            if self.is_maximizing:
                child_heuristic += exploration_term
                if child_heuristic > best_heuristic:
                    best_heuristic = child_heuristic
                    best_child = child
            else:
                child_heuristic -= exploration_term
                if child_heuristic < best_heuristic:
                    best_heuristic = child_heuristic
                    best_child = child

        return best_child.choose_rollout_node(c)

    def rollout(self, rollouts=7, pool=None):
        rollout_sum = sum(pool.starmap(self.execute_single_rollout, [() for _ in range(rollouts)]) if pool is not None
                          else [self.execute_single_rollout() for _ in range(rollouts)])

        # update this node and all its parents
        node = self
        while node is not None:
            node.rollout_sum += rollout_sum
            node.rollout_count += rollouts
            node = node.parent

    def execute_single_rollout(self):
        state = self.position
        while not self.GameClass.is_over(state):
            sub_states = self.GameClass.get_possible_moves(state)
            state = sub_states[np.random.randint(len(sub_states))]

        return self.GameClass.get_winner(state)
