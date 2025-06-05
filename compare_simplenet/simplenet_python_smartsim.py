import os
from smartredis import Client
from smartsim import Experiment

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
in_data = client.get_tensor("in_data")
print(f"Input: {in_data}")

# TODO: Actually use PyTorch
out_data = 2 * in_data

print(f"Output: {out_data}")
client.put_tensor("out_data", out_data)
