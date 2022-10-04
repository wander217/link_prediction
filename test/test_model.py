import yaml
from structure import DocLinkPrediction
from dataset import DocAlphabet

config_path = r'F:\project\python\link_prediction\asset\config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

alphabet = DocAlphabet(**config['alphabet'])
model = DocLinkPrediction(**config['structure'], alphabet=alphabet)
print(model.params_counter())
