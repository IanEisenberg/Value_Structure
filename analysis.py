import statsmodels.formula.api as smf
from utils.data_preparation_utils import process_data

#test code

data = process_data('test')

RL = data['RL']
structure = data['structure']

m = smf.ols(formula='rt ~ C(rotation) + correct.shift() + community_transition + steps_since_seen',
            data = structure)
res = m.fit()
res.summary()