from ETIA.CausalLearning import CausalLearner
from ETIA.CausalLearning.configurations.configurations import Configurations

# Load configuration and learn model
conf = Configurations('example_dataset.csv')
learner = CausalLearner(configurations=conf)
results = learner.learn_model()
print(results)