import tensorflow as tf
import os
import argparse
from game_runner import GameRunner


def main():
    parser = argparse.ArgumentParser(description='Deep Q Learning')

    parser.add_argument(
        '-m',
        '--model-dir',
        required=False,
        default=None,
        help='Path to a new or existing model directory.'
    )

    parser.add_argument(
        '-t',
        '--train-model',
        required=False,
        default=True,
        action='store_true',
        help='If true, model will be trained. Otherwise, it will be evaluated.'
    )

    parser.add_argument(
        '-d',
        '--default-config',
        required=False,
        default=os.path.join('default_configs', 'CartPoleRawImg-v0.yaml'),
        help='The default config to use when creating a new model.'
    )

    args = parser.parse_args()

    with tf.Session() as session:
        game_runner = GameRunner(session, args.default_config, args.model_dir)

        if args.train_model:
            game_runner.train()
        else:
            game_runner.evaluation()


if __name__ == '__main__':
    main()
