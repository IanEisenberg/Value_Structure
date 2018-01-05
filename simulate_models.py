from utils.data_preparation_utils import process_data
from models import BasicRLModel, SR_RLModel


data = process_data('651')

RL = data['RL']
structure = data['structure']
meta = data['meta']
descriptive_stats = data['descriptive_stats']

m = BasicRLModel(RL)
m.optimize()
SRm = SR_RLModel(RL, structure)
SRm.update_params(m.get_params())
models = [('Basic', m), ('SR', SRm)]

for name, model in models:
sim_data, tracked_values = m.simulate(prob_match=True)

sim_data, tracked_values = SRm.simulate(prob_match=True)

# compare models
