import os
import dill
import numpy as np
import pathlib
import torch
import requests
from typing import Union
from itertools import chain

try:
    NEURODIFF_API_URL = os.environ["NEURODIFF_API_URL"]
except KeyError:
    NEURODIFF_API_URL = "http://dev.neurodiff.io/api/v1"

def is_solution_name(name):
    if name.startswith('./'):
        return False
    else:
        return True

def process_response(response):
    """
    Process a `requests.Reponse` object, returning the decoded contents or
    raising an APIError (or subclass) exception on request failure.
    """
    return response.json()

def get_file(url, solution_name):
    cache_dir = os.path.join(os.path.expanduser('~'), '.neurodiff')
    if not os.path.exists(cache_dir):
      os.mkdir(cache_dir)
    solution_file_path = os.path.join(cache_dir,solution_name)
    if not os.path.exists(solution_file_path):
        url = url +"?name="+solution_name
        # Download the solution
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(solution_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return solution_file_path

class SolverConfig():
    conditions = None
    ode_system = None
    nets = None
    optimizer = None
    #optimizer_params = {}
    train_generator = None
    valid_generator = None

class PretrainedSolver():

    #Saving selected attributes of model in dict
    def save(self,solution_name_or_path: Union[str, os.PathLike],save_remote=False):
        # Check if optimizer is existing in pytorch
        optimizer_class=None
        for cls in torch.optim.Optimizer.__subclasses__():
            if self.optimizer.__class__.__name__ == cls.__name__:
                optimizer_class = self.optimizer.__class__

        save_dict = {
            "metrics": self.metrics_fn,
            "criterion": self.criterion,
            "conditions": self.conditions,
            "global_epoch": self.global_epoch, #loss_history
            "nets": self.nets,
            "optimizer": self.optimizer,
            "optimizer_state": self.optimizer.state_dict(),
            "optimizer_class": optimizer_class,
            "diff_eqs": self.diff_eqs,
            "generator": self.generator,
            "type": self.__class__
        }

        # Save solution locally
        with open(solution_name_or_path,'wb') as file:
            dill.dump(save_dict,file)

        # Save remote if needed
        if is_solution_name(solution_name_or_path):
            # Save remote
            print("Saving solution to:",NEURODIFF_API_URL)
            url = NEURODIFF_API_URL + "/solutions"
            # Create a solution
            solution = {
                "name":solution_name_or_path,
                "description":solution_name_or_path
            }
            response = requests.post(
                url,
                json=solution
            )
            solution = process_response(response)
            print(solution)

            # Upload the solution file
            url = NEURODIFF_API_URL + "/solutions/{id}/upload"
            response = requests.post(
                url.format(id=solution["id"]),
                files={"file": open(solution_name_or_path, "rb")}
            )

            # Remove local solution
            os.remove(solution_name_or_path)
            
			
    #Loading saved attributes into new solver object   

    # Have to add check and warning/error     
    @classmethod		
    def load(cls, solution_name_or_path, config=SolverConfig()):
        
        # Load from remote
        if is_solution_name(solution_name_or_path):
            url = NEURODIFF_API_URL + "/solutions/download"
            solution_file_path = get_file(url,solution_name_or_path)
        else:
            solution_file_path = solution_name_or_path
        # Load the solution
        with open(solution_file_path,'rb') as file:
            load_dict = dill.load(file)

        # Loading user defined generator and extracting time domain information
        if config.train_generator == None:
            train_generator = load_dict['generator']['train']
            valid_generator = load_dict['generator']['train']
        else:
            train_generator = config.train_generator
            valid_generator = config.valid_generator

        t = train_generator.get_examples()[0].detach().numpy()
        t_min = np.round(min(t))
        t_max = np.round(max(t))

        # Loading user defined ode_system or system from load file 
        if config.ode_system == None:
            ode = load_dict['diff_eqs']
        else:
            ode = config.ode_system
        
        # Loading user defined conditions or conditions from load file
        if config.conditions == None:
            cond = load_dict['conditions']
        else:
            cond = config.conditions

        # Loading user defined nets or nets from load file
        if config.nets == None:
            nets = load_dict['nets']
        else:
            nets = config.nets
        
        # Loading user defined optimizer or optimizer from load file
        if config.optimizer == None:
            #optimizer = load_dict['optimizer']
            if load_dict['optimizer_class'] is not None:
                optimizer = load_dict['optimizer_class'](chain.from_iterable(n.parameters() for n in nets))
                optimizer = optimizer.load_state_dict(load_dict['optimizer_state'])
            else:
                optimizer = load_dict['optimizer']

        else:
            check_flag=False
            for cls in torch.optim.Optimizer.__subclasses__():
                if config.optimizer.__class__.__name__ == cls.__name__:
                    optimizer =  config.optimizer(chain.from_iterable(n.parameters() for n in nets))
                    optimizer = optimizer.load_state_dict(load_dict['optimizer_state'])
                    check_flag = True

            if check_flag=False:
                optimizer = load_dict['optimizer']

        solver = cls(ode_system = ode,
                    conditions = cond,
                    criterion = load_dict['criterion'],
                    metrics = load_dict['metrics'],
                    nets = nets,
                    optimizer = optimizer,
                    train_generator = train_generator,
                    valid_generator = valid_generator,
                    t_min = t_min,
                    t_max = t_max)

        return solver

    #Saving the Solver Object
    def save_solver(self,filename,path=pathlib.Path().absolute()):
        PATH = os.path.join(path,filename)
        try:
            with open(PATH,'wb') as file:
                dill.dump(self,file)
            print("Solver has been saved.")
            return True
        except:
            return False  

    #Loading the Solver Object  
    @classmethod
    def load_solver(cls,path,retrain=False):
        with open(path,'rb') as file:
                solver = dill.load(file)
        return solver

def load_solver(solver_name_or_path):
    ...
    # Check if the solver exist locally

    # Load the dict

    # How do we know what type
