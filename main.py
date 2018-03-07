import tensorflow as tf
from game_runner import GameRunner

testing = False

def main(_):
    with tf.Session() as session:
        game_runner = GameRunner(session)

        if testing:
            game_runner.evaluation()
        else:
            game_runner.train()

     
if __name__ == '__main__':
    tf.app.run()
