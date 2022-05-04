# This script is used to generate map embedding for GraphMapEnvV3. 
from karateclub import FeatherGraph

model = FeatherGraph()
model.fit(graphs)
X = model.get_embedding()