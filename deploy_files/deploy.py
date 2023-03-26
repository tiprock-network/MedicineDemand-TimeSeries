from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import LocalWebservice
import json
import requests
import urllib.request

#sepcify and connect to the workspace
ws=Workspace(
    subscription_id='ee37f2c8-a225-4d2e-ac0e-50e3c96cd661',
    resource_group='Theophilus.Owiti-rg',
    workspace_name='mlworkspace'
)
#create a config file that saves the workspace for multiple environments
#ws.write_config(path='./config',file_name='ml_ws.json')

#register model
#model=Model.register(ws,model_name='reg_model',model_path='./model/reg_model.pkl')

#MODEL DEPLOYMENT
#inference config
env=Environment(name='project_environment')
dummy_inf_conf=InferenceConfig(environment=env,source_directory='./source_dir',entry_script='./echo_score.py')

#local deployment configuration
local_deploy_config=LocalWebservice.deploy_configuration(port=1360)

#create service
current_model=Model(ws,'reg_model')
service=Model.deploy(ws,'medicinepred',[current_model],dummy_inf_conf,local_deploy_config,overwrite=True)
service.wait_for_deployment(show_output=True)
print(service.get_logs())

#calling the model using the service- locally
uri=service.scoring_uri
requests.get('http://localhost:1360')
headers={'Content-Type','application/json'}
data={
    "Inputs":{
        "data":[{
            "year":2019,
            "month":11,
            "day":10
        }]
    }
}

data=json.dumps(data)
response=requests.post(uri,data=data,headers=headers)
print(response.json())





    