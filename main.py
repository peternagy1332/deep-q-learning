import tensorflow as tf
from game_runner import GameRunner


def main():
    # E.g.: "models/model-YYYY-MM-DD--HH:MM:SS"
    model_dir = None

    with tf.Session() as session:
        game_runner = GameRunner(session, model_dir)

        if model:
            game_runner.evaluation()
        else:
            game_runner.train()


if __name__ == '__main__':
    main()
