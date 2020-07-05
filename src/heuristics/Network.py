from time import time
import numpy as np
from multiprocessing import Process, Pipe


class Network:
    """
    A deep convolutional neural network that takes as input the ML representation of a game (n, m, k) and tries to
    return 1 if player 1 (white in chess, red in checkers etc.) is going to win, -1 if player 2 is going to win,
    and 0 if the game is going to be a draw.

    Network will receive all inputs and output moves in the format of flattened arrays of legal moves. \
    It will internally handle conversion too and from move distribution matrices using GameClass.get_legal_moves
    """

    def __init__(self, GameClass, model_path=None):
        self.GameClass = GameClass
        self.model_path = model_path

        # lazily initialized so Network can be passed between processes before being initialized
        self.model = None

    def initialize(self):
        # Note: keras imports are within functions to prevent initializing keras in processes that import from this file
        from keras.models import load_model

        if self.model is not None:
            return

        if self.model_path is not None:
            input_shape = self.GameClass.STATE_SHAPE
            output_shape = self.GameClass.MOVE_SHAPE
            self.model = load_model(self.model_path)
            if self.model.input_shape != (None,) + input_shape:
                raise Exception('Input shape of loaded model doesn\'t match!')
            if self.model.output_shape != [(None,) + output_shape, (None, 1)]:
                raise Exception('Output shape of loaded model doesn\'t match!')
        else:
            self.model = self.create_model()

    def create_model(self):
        """
        https://www.youtube.com/watch?v=OPgRNY3FaxA
        """
        # Note: keras imports are within functions to prevent initializing keras in processes that import from this file
        from keras.models import Model
        from keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Activation, Add, Reshape

        input_shape = self.GameClass.STATE_SHAPE
        output_shape = self.GameClass.MOVE_SHAPE
        output_neurons = np.product(output_shape)

        input_tensor = Input(input_shape)

        # convolutional layer
        x = Conv2D(16, (3, 3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # residual layers
        for _ in range(3):
            y = Conv2D(16, (3, 3), padding='same')(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(16, (3, 3), padding='same')(y)
            y = BatchNormalization()(y)
            # noinspection PyTypeChecker
            x = Add()([x, y])
            x = Activation('relu')(x)

        # policy head
        policy = Conv2D(2, (1, 1), padding='same')(x)
        policy = BatchNormalization()(policy)
        policy = Activation('relu')(policy)
        policy = Flatten()(policy)
        policy = Dense(output_neurons, activation='softmax')(policy)
        policy = Reshape(output_shape, name='policy')(policy)

        # value head
        value = Conv2D(1, (1, 1), padding='same')(x)
        value = BatchNormalization()(value)
        value = Activation('relu')(value)
        value = Flatten()(value)
        value = Dense(16, activation='relu')(value)
        value = Dense(1, activation='tanh', name='value')(value)

        model = Model(input_tensor, [policy, value])
        model.compile(optimizer='adam', loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
                      metrics=['mean_squared_error'])
        return model

    def predict(self, states):
        return self.model.predict(states)

    def call(self, states):
        """
        :param states: The input positions with shape (k,) + GameClass.State_Shape, where k is the number of positions.
        :return: A list of length k. Each element of the list is a tuple where the 0th element is the probability
                 distribution on legal moves, and the 1st element is the evaluation (a float in (-1, 1)).
        """
        # start_time = time()
        raw_policies, evaluations = self.predict(states)
        # print('Prediction time', time() - start_time)

        filtered_policies = [raw_policy[self.GameClass.get_legal_moves(state)]
                             for state, raw_policy in zip(states, raw_policies)]
        filtered_policies = [filtered_policy / np.sum(filtered_policy) for filtered_policy in filtered_policies]

        evaluations = evaluations.reshape(states.shape[0])
        return [(filtered_policy, evaluation) for filtered_policy, evaluation in zip(filtered_policies, evaluations)]

    def choose_move(self, position, return_evaluation=False):
        distribution, evaluation = self.call(position[np.newaxis, ...])[0]
        idx = np.random.choice(np.arange(len(distribution)), p=distribution)
        move = self.GameClass.get_possible_moves(position)[idx]
        return (move, evaluation) if return_evaluation else move

    def train(self, data, validation_fraction=0.2):
        # Note: keras imports are within functions to prevent initializing keras in processes that import from this file
        from keras.callbacks import TensorBoard, EarlyStopping

        split = int((1 - validation_fraction) * len(data))
        train_input, train_output = self.process_data(data[:split])
        test_input, test_output = self.process_data(data[split:])
        print('Training Samples:', train_input.shape[0])
        print('Validation Samples:', test_input.shape[0])

        self.model.fit(train_input, train_output, epochs=100, validation_data=(test_input, test_output),
                       callbacks=[TensorBoard(log_dir=f'../heuristics/logs/model-{time()}'),
                                  EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

    def process_data(self, data, one_hot=False):
        states = []
        policy_outputs = []
        value_outputs = []

        for game, outcome in data:
            for position, distribution in game:
                legal_moves = self.GameClass.get_legal_moves(position)
                policy = np.zeros_like(legal_moves, dtype=float)
                policy[legal_moves] = distribution

                if one_hot:
                    idx = np.unravel_index(policy.argmax(), policy.shape)
                    policy = np.zeros_like(policy)
                    policy[idx] = 1

                states.append(position)
                policy_outputs.append(policy)
                value_outputs.append(outcome)

        input_data = np.stack(states, axis=0)
        policy_outputs = np.stack(policy_outputs, axis=0)
        value_outputs = np.array(value_outputs)

        shuffle = np.arange(input_data.shape[0])
        np.random.shuffle(shuffle)
        input_data = input_data[shuffle, ...]
        policy_outputs = policy_outputs[shuffle, ...]
        value_outputs = value_outputs[shuffle, ...]

        return input_data, [policy_outputs, value_outputs]

    def save(self, model_path):
        self.model.save(model_path)

    @staticmethod
    def spawn_process(GameClass, model_path=None, pipes=1):
        parent_pipes, worker_pipes = zip(*[Pipe() for _ in range(pipes)])
        process = Process(target=Network.process_loop, args=(GameClass, model_path, worker_pipes))

        proxy_networks = [ProxyNetwork(parent_pipe) for parent_pipe in parent_pipes]
        return process, proxy_networks

    @staticmethod
    def process_loop(GameClass, model_path, worker_pipes):
        network = Network(GameClass, model_path)
        network.initialize()

        while True:
            for worker_pipe in worker_pipes:
                if worker_pipe.poll():
                    worker_pipe.send(network.call(worker_pipe.recv()))

    @staticmethod
    def spawn_dual_architecture_process(GameClass, model_path=None, pipes_per_section=1):
        parent_a_pipes, worker_a_pipes = zip(*[Pipe() for _ in range(pipes_per_section)])
        parent_b_pipes, worker_b_pipes = zip(*[Pipe() for _ in range(pipes_per_section)])
        process = Process(target=Network.dual_architecture_process_loop,
                          args=(GameClass, model_path, worker_a_pipes, worker_b_pipes))
        proxy_a_networks = [ProxyNetwork(parent_a_pipe) for parent_a_pipe in parent_a_pipes]
        proxy_b_networks = [ProxyNetwork(parent_b_pipe) for parent_b_pipe in parent_b_pipes]
        return process, proxy_a_networks, proxy_b_networks

    @staticmethod
    def dual_architecture_process_loop(GameClass, model_path, worker_a_pipes, worker_b_pipes):
        network = Network(GameClass, model_path)
        network.initialize()

        def send_results(requests, results, pipes):
            pos = 0
            for request, pipe in zip(requests, pipes):
                new_pos = pos + request.shape[0]
                pipe.send(results[pos:new_pos])
                pos = new_pos

        requests_a = [worker_a_pipe.recv() for worker_a_pipe in worker_a_pipes]
        # concatenate is used instead of stack because each request already has shape (k,) + GameClass.STATE_SHAPE
        results_a = network.call(np.concatenate(requests_a, axis=0))

        while True:
            # receive B requests
            # start_time = time()
            requests_b = [worker_b_pipe.recv() for worker_b_pipe in worker_b_pipes]
            # print('Network received B requests in ', time() - start_time)

            # return A results
            # start_time = time()
            send_results(requests_a, results_a, worker_a_pipes)
            # print('Network sent A results in ', time() - start_time)

            # compute B results
            # start_time = time()
            results_b = network.call(np.concatenate(requests_b, axis=0))
            # print('Network computed B results in', time() - start_time)

            # receive A requests
            # start_time = time()
            requests_a = [worker_a_pipe.recv() for worker_a_pipe in worker_a_pipes]
            # print('Network received A requests in ', time() - start_time)

            # return B results
            # start_time = time()
            send_results(requests_b, results_b, worker_b_pipes)
            # print('Network sent B results in ', time() - start_time)

            # compute A results
            # start_time = time()
            results_a = network.call(np.concatenate(requests_a, axis=0))
            # print('Network computed A results in', time() - start_time)


class ProxyNetwork:
    def __init__(self, model_pipe):
        self.model_pipe = model_pipe

    def initialize(self):
        """
        Added so that a ProxyNetwork can be used in place of a regular Network with no issues when initialize is called.
        """
        pass

    def call(self, state):
        self.model_pipe.send(state)
        return self.model_pipe.recv()


def train():
    from src.games.Connect4 import Connect4
    import os
    import pickle
    net = Network(Connect4, '../heuristics/models/model0.h5')
    net.initialize()
    print('Network size: ', net.model.count_params())

    data = []
    for file in os.listdir('../heuristics/games/mcts_network0_games'):
        with open(f'../heuristics/games/mcts_network0_games/{file}', 'rb') as fin:
            data.append(pickle.load(fin))
    net.train(data)
    net.save('../heuristics/models/model1.h5')


if __name__ == '__main__':
    train()
