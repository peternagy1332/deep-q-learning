import tensorflow as tf
import os
from game_runner import GameRunner


def main():
    # E.g.: "models/model-YYYY-MM-DD--HH:MM:SS"
    training = True
    model_dir = None

    with tf.Session() as session:
        game_runner = GameRunner(session, model_dir)

        if training:
            game_runner.train()
        else:
            game_runner.evaluation()


if __name__ == '__main__':
    main()
