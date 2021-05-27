import os
import dill
import numpy as np
import pathlib
import torch
import requests
import tempfile
from typing import Union
from itertools import chain
import inspect
import ast

try:
    NEURODIFF_API_URL = os.environ["NEURODIFF_API_URL"]
except KeyError:
    NEURODIFF_API_URL = "http://dev.neurodiff.io/api/v1"

try:
    NEURODIFF_API_KEY = os.environ["NEURODIFF_API_KEY"]
except KeyError:
    print("No API Key was found in environment variable NEURODIFF_API_KEY")
    NEURODIFF_API_KEY = ""

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

def _make_api_headers():
    headers = {}

    headers["api_key"] = NEURODIFF_API_KEY

    return headers

def get_file(url, solution_name):
    cache_dir = os.path.join(os.path.expanduser('~'), '.neurodiff')
    if not os.path.exists(cache_dir):
      os.mkdir(cache_dir)
    solution_file_path = os.path.join(cache_dir,solution_name)
    if not os.path.exists(solution_file_path):
        url = url +"?name="+solution_name
        # Download the solution
        with requests.get(url, stream=True,headers=_make_api_headers()) as r:
            r.raise_for_status()
            with open(solution_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return solution_file_path

def get_source(lambda_function):
    lambda_text = ""
    try:
        source_lines, _ = inspect.getsourcelines(lambda_function)
        lambda_text = "".join([line.strip() for line in source_lines])
        source_ast = ast.parse(lambda_text)
        lambda_node = next((node for node in ast.walk(source_ast)
                            if isinstance(node, ast.Lambda)), None)
        lambda_text = lambda_text[lambda_node.col_offset:]
    except:
        pass

    return lambda_text

def get_parameters(lambda_function):
    parameters = {}
    try:
        closures = lambda_function.__closure__
        if closures is not None:
            freevars = lambda_function.__code__.co_freevars
            for i,c in enumerate(closures):
                parameters[freevars[i]] = c.cell_contents
        else:
            gbs = lambda_function.__globals__
            co_names = lambda_function.__code__.co_names
            for i,c in enumerate(co_names):
                if c != "diff":
                    parameters[c] = gbs[c]
    except:
        pass

    return parameters

class SolverConfig():
    conditions = None
    ode_system = None
    pde_system = None
    nets = None
    optimizer = None
    optimizer_params = None
    train_generator = None
    valid_generator = None

class PretrainedSolver():
    diff_eqs_source = ""

    def print_diff_eqs(self):
        lambda_text = get_source(self.diff_eqs)
        if lambda_text == "":
            lambda_text = self.diff_eqs_source
        print(lambda_text)


    #Saving selected attributes of model in dict
    def save(self,
        path: str = None,
        name: str = None,
        repo: str = None,
        save_to_hub=False):
        
        # Check params
        if path is None and save_to_hub == False:
            raise Exception("path cannot be empty when save_to_hub=False")
        if name is None and save_to_hub == True:
            raise Exception("name cannot be empty when save_to_hub=True")

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
            "diff_eqs_source": get_source(self.diff_eqs),
            "generator": self.generator,
            "train_loss_history": self.metrics_history['train_loss'],
            "valid_loss_history": self.metrics_history['valid_loss'],
            "type": self.__class__,
            "type_name": self.__class__.__name__
        }

        # Save remote if needed
        if save_to_hub:
            # Save solution in temp file
            with tempfile.NamedTemporaryFile() as tmp_file:
                dill.dump(save_dict,tmp_file)

                # Save remote
                print("Saving solution to:",NEURODIFF_API_URL)
                if repo is None:
                    print("Default repo will be used to save solution")

                url = NEURODIFF_API_URL + "/solutions"
                # Create a solution
                solution = {
                    "name":name,
                    "description":name,
                    "diff_eqs_source": save_dict["diff_eqs_source"]
                }
                print(solution)
                response = requests.post(
                    url,
                    json=solution,
                    headers=_make_api_headers()
                )
                solution = process_response(response)
                print(solution)

                # Upload the solution file
                url = NEURODIFF_API_URL + "/solutions/{id}/upload"
                response = requests.post(
                    url.format(id=solution["id"]),
                    files={"file": open(tmp_file.name, "rb")},
                    headers=_make_api_headers()
                )

        else:
            # Save solution locally
            with open(path,'wb') as file:
                dill.dump(save_dict,file)
            
			
    #Loading saved attributes into new solver object   

    # Have to add check and warning/error     
    @classmethod		
    def load(cls, 
        path: str = None,
        name: str = None,
        config=SolverConfig()):
        
        # Check params
        if path is None and name is None:
            raise Exception("Either path or name is required to load solver")
        
        # Load from remote
        if path is None:
            url = NEURODIFF_API_URL + "/solutions/download"
            solution_file_path = get_file(url,name)
        else:
            solution_file_path = path
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

        # Loading user defined ode_system or system from load file 
        if (config.ode_system == None) and (config.pde_system == None):
            de_system = load_dict['diff_eqs']
        elif config.ode_system is not None:
            de_system = config.ode_system
        elif config.pde_system is not None:
            de_system = config.pde_system
        
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

        train_loss = []
        valid_loss = []
        if config.optimizer == None:
            #optimizer = load_dict['optimizer']
            if load_dict['optimizer_class'] is not None:
                optimizer = load_dict['optimizer_class'](chain.from_iterable(n.parameters() for n in nets))
                optimizer.load_state_dict(load_dict['optimizer_state'])
            else:
                optimizer = load_dict['optimizer']
            
            #As older network/optimizer is loaded load loss history as well
            train_loss = load_dict['train_loss_history']
            valid_loss = load_dict['valid_loss_history']
        else:
            # Declare a flag to check if optimizer is passed as a class. Link parameters and load state to the new class
            check_flag=False

            #for classes in torch.optim.Optimizer.__subclasses__():
            #    if config.optimizer.__class__.__name__ == classes.__name__:
            #        if config.nets == None:
            #            optimizer = config.optimizer
            #            optimizer.param_groups[0]['state'] = load_dict['optimizer_state']['state']
            #            optimizer.param_groups[0]['params'] = load_dict['optimizer_state']['param_groups'][0]['params']
            #            check_flag=True

            if config.optimizer_params is not None: 
                optimizer = config.optimizer(chain.from_iterable(n.parameters() for n in nets),**config.optimizer_params)   
                train_loss = load_dict['train_loss_history']
                valid_loss = load_dict['valid_loss_history']
                check_flag=True    

            # In case of custom optimizer or optimizer externally linked to the network, load complete optimizer
            if check_flag==False:
                optimizer = load_dict['optimizer']
                

        # Initiate a new Solver
        if load_dict["type_name"] == "Solver1D":
            # t min/max
            t_min = load_dict['generator']['train'].__dict__['generator'].__dict__['t_min']
            t_max = load_dict['generator']['train'].__dict__['generator'].__dict__['t_max']

            solver = cls(ode_system = de_system,
                        conditions = cond,
                        criterion = load_dict['criterion'],
                        metrics = load_dict['metrics'],
                        nets = nets,
                        optimizer = optimizer,
                        train_generator = train_generator,
                        valid_generator = valid_generator,
                        t_min = t_min,
                        t_max = t_max)
        elif load_dict["type_name"] == "Solver2D":
            xy_min = load_dict['generator']['train'].__dict__['generator'].__dict__['xy_min']
            xy_max = load_dict['generator']['train'].__dict__['generator'].__dict__['xy_max']

            solver = cls(pde_system = de_system,
                        conditions = cond, 
                        xy_min = xy_min, 
                        xy_max = xy_max,
                        nets=nets, 
                        train_generator=train_generator, 
                        valid_generator=valid_generator, 
                        optimizer=optimizer,
                        criterion=load_dict['criterion'], 
                        metrics=load_dict['metrics'])
        
        solver.metrics_history['train_loss'] = train_loss
        solver.metrics_history['valid_loss'] = valid_loss

        try:
            solver.diff_eqs_source = load_dict["diff_eqs_source"]
        except:
            pass
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
