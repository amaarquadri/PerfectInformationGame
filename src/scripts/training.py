import numpy as np
import easygui
from ..utils.active_game import ActiveGame as GameClass
from ..learning.learning import SelfPlayReinforcementLearning, MCTSRolloutGameGenerator
from ..utils.utils import get_training_path


def main():
    # trainer = SelfPlayReinforcementLearning(GameClass,
    #                                         f'{get_training_path(GameClass)}/models/model_reinforcement.h5',
    #                                         threads_per_section=14, game_batch_count=7, expansions_per_move=1500,
    #                                         c=np.sqrt(2), d=1)
    trainer = MCTSRolloutGameGenerator(GameClass, threads=14, expansions_per_move=1500, c=np.sqrt(2))
    trainer.start()

    text = easygui.enterbox('End Training? Timeout (seconds):', title=f'{GameClass.__name__} Training', default='120')
    if text is None:
        print('Timeout not given! Using default of 1 hour.')
        timeout = 3600
    else:
        try:
            timeout = int(text)
        except ValueError:
            print('Timeout invalid! Must be an integer. Using default of 1 hour.')
            timeout = 3600

    print(f'Ending training with timeout of {timeout}')
    trainer.terminate(timeout)


if __name__ == '__main__':
    main()
