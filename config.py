import os
import yaml
import datetime
import time
import pickle


class Config(object):
    def __init__(self, default_config, model_dir):

        self.model_dir = self.generate_model_name() if model_dir is None else model_dir
        self.scores_file = os.path.join(self.model_dir, "scores.pickle")
        self.eval_stats_file = os.path.join(self.model_dir, "eval_stats.pickle")
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model_config = os.path.join(self.model_dir, "config.yaml")
        self.keys = set()

        # If it is a new model, or the config file is deleted
        if not os.path.exists(self.model_config):
            with open(default_config, "r", encoding="utf8") as default_config_file:
                default_config = yaml.safe_load(default_config_file)

                for key in default_config:
                    self.keys.add(key)
                    setattr(self, key, default_config[key])

            with open(self.model_config, "w", encoding="utf8") as config_file:
                yaml.dump(default_config, config_file, default_flow_style=False)
        else:
            # Load the existing configuration
            with open(self.model_config, "r", encoding="utf8") as config_file:
                config = yaml.safe_load(config_file)
                for key in config:
                    self.keys.add(key)
                    setattr(self, key, config[key])
    
        self.epsilon_annealer = (self.initial_exploration - self.final_exploration) / self.final_exploration_frame

    def save(self, scores, eval_stats):
        """Overwriting existing config file with the current values."""
        with open(self.model_config, "w", encoding="utf8") as config_file:
            current_config = {key: getattr(self, key) for key in self.keys}
            yaml.dump(current_config, config_file, default_flow_style=False)
        
        with open(self.scores_file, "wb") as f_scores:
            pickle.dump(scores, f_scores, pickle.HIGHEST_PROTOCOL)

        with open(self.eval_stats_file, "wb") as f_eval_stats:
            pickle.dump(eval_stats, f_eval_stats, pickle.HIGHEST_PROTOCOL)

    def generate_model_name(self):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d--%H-%M-%S')
        return os.path.join("models", "model--"+st)

    def get_scores_list(self):
        if os.path.exists(self.scores_file):
            with open(self.scores_file, "rb") as f_scores:
                return pickle.load(f_scores)
        else:
            return []

    def get_eval_stats(self):
        if os.path.exists(self.eval_stats_file):
            with open(self.eval_stats_file, "rb") as f_eval_stats:
                return pickle.load(f_eval_stats)
        else:
            return {}
