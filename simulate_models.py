from utils.data_preparation_utils import process_data
from models import BasicRLModel, GraphRLModel, SR_RLModel


data = process_data('651')

RL = data['RL']
structure = data['structure']
meta = data['meta']
descriptive_stats = data['descriptive_stats']

m = BasicRLModel(RL)
sim_data, tracked_values = m.simulate()
