import os

from smartredis import Client
from smartsim import Experiment
import torch

import simplenet

os.environ["SMARTSIM_DB_FILE_PARSE_INTERVAL"] = "5"
language = "fortran"
exe = "build/simplenet_fortran_smartsim"
launcher = "local"
interface = "lo"

exp = Experiment("smartsim_experiment", launcher=launcher)
db = exp.create_database(port=6780, interface=interface)

exp.generate(db, overwrite=True)
exp.start(db)
print(f"Database started at address: {db.get_address()}")

# set simulation parameters we can pass as executable arguments
exe_args = []
# create "run settings" for the simulation which define how
# the simulation will be executed when passed to Experiment.start()
settings = exp.create_run_settings(exe, exe_args=exe_args)
settings.set_nodes(1)
settings.set_tasks(1)

model = exp.create_model("smartsim_model", run_settings=settings)

exp.generate(model, overwrite=True)
exp.start(model, block=True, summary=False)

# Connect a SmartRedis client to retrieve data
client = Client(address=db.get_address()[0], cluster=False)
trained_model_dummy_input = torch.from_numpy(client.get_tensor("in_data"))
print(f"Input (Python): {trained_model_dummy_input}")

# Use PyTorch to run the forward method on the SimpleNet model
trained_model = simplenet.SimpleNet()
trained_model_dummy_output = trained_model(trained_model_dummy_input)

print(f"Output (Python): {trained_model_dummy_output}")
client.put_tensor("out_data", trained_model_dummy_output.detach().numpy())
