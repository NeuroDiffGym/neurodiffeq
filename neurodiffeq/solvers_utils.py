import os
import json
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
import types

# Is Dev mode
try:
    DEV = os.environ["DEV"]
    DEV = True
except KeyError:
    DEV = False

try:
    NEURODIFF_API_URL = os.environ["NEURODIFF_API_URL"]
except KeyError:
    if DEV:
        NEURODIFF_API_URL = "http://dev.neurodiff.io/api/v1"
    else:
        NEURODIFF_API_URL = "http://www.neurodiff.io/api/v1"


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
    try:
        NEURODIFF_API_KEY = os.environ["NEURODIFF_API_KEY"]
    except KeyError:
        print("No API Key was found in environment variable NEURODIFF_API_KEY")
        NEURODIFF_API_KEY = ""

    headers["apikey"] = NEURODIFF_API_KEY

    return headers


def create_cache_dir():
    cache_dir = os.path.join(os.path.expanduser('~'), '.neurodiff')
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    return cache_dir


def get_file(url, name):
    cache_dir = create_cache_dir()
    solution_file_path = os.path.join(cache_dir, name.replace("/", "_"))

    # if not os.path.exists(solution_file_path):
    url = url + "?name="+name
    # Download the solution
    with requests.get(url, stream=True, headers=_make_api_headers()) as r:
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
            for i, c in enumerate(closures):
                parameters[freevars[i]] = c.cell_contents
        else:
            gbs = lambda_function.__globals__
            co_names = lambda_function.__code__.co_names
            for i, c in enumerate(co_names):
                if c != "diff" and c != "torch":
                    parameters[c] = gbs[c]
    except:
        pass

    return parameters


def get_conditions(conditions):
    condition_list = []
    try:
        for condition in conditions:
            cond_dict = condition.__dict__
            cond_dict["condition_type"] = condition.__class__.__name__

            # Get the lambda functions in the condition
            for key, value in cond_dict.items():
                if isinstance(value, types.FunctionType):
                    function_source = get_source(value)
                    if function_source != "":
                        cond_dict[key] = function_source

            condition_list.append(cond_dict)
    except:
        pass
    return condition_list


def get_generator(generator):
    gen_dict = {}
    try:
        gen_dict = generator["train"].__dict__['generator'].__dict__.copy()
        if "examples" in gen_dict:
            del gen_dict['examples']

        if "grid_x" in gen_dict:
            del gen_dict['grid_x']
        if "grid_y" in gen_dict:
            del gen_dict['grid_y']
        del gen_dict['getter']
    except:
        pass
    return gen_dict


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def get_sample_solution1D(solver):
    sample_solution_curve = []
    try:
        t = np.linspace(solver.t_min, solver.t_max, 10 *
                        (int(solver.t_max-solver.t_min)))
        sample_solution = solver.get_solution()(t)

        if not isinstance(sample_solution, list):
            sample_solution = [sample_solution]

        for i in range(len(sample_solution)):
            sample_solution[i] = sample_solution[i].cpu(
            ).detach().numpy().tolist()

        sample_solution_curve = [t.tolist(), sample_solution]
    except:
        pass
    return sample_solution_curve


def get_sample_solution2D(solver):
    sample_solution_curve = []

    try:
        inputs = solver.generator['train'].get_examples()
        sample_solution = solver.get_solution()(
            inputs[0].view(-1), inputs[1].view(-1))
        sample_solution = sample_solution.cpu().detach().numpy()
        for i in range(2):
            inputs[i] = inputs[i].view(-1).cpu().detach().numpy().tolist()
        sample_solution_curve = [inputs, sample_solution.reshape(
            solver.generator["train"].__dict__['generator'].__dict__['grid']).tolist()]
    except:
        pass
    return sample_solution_curve


def get_networks(solver):
    networks = []
    for idx, net in enumerate(solver.nets):
        layers = []
        for layer in net.NN:
            layer_params = {
                "layer": layer.__class__.__name__
            }
            layer_dict = layer.__dict__
            if "in_features" in layer_dict:
                layer_params["in_features"] = layer_dict["in_features"]
            if "out_features" in layer_dict:
                layer_params["out_features"] = layer_dict["out_features"]
            if "bias" in layer_dict:
                layer_params["bias"] = layer_dict["bias"]

            layers.append(layer_params)
        networks.append({"layers": layers})

    return networks


def get_loss(loss):
    try:
        ...
#        loss_sample = json.dumps(loss,cls = JsonEncoder)
#        loss_sample = loss
    except:
        pass
    return loss_sample


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

    # Saving selected attributes of model in dict

    def save(self,
             path: str = None,
             name: str = None,
             save_to_hub=False):

        # Check params
        if path is None and save_to_hub == False:
            raise Exception("path cannot be empty when save_to_hub=False")
        if name is None and save_to_hub == True:
            raise Exception("name cannot be empty when save_to_hub=True")

        # Check if optimizer is existing in pytorch
        optimizer_class = None
        for cls in torch.optim.Optimizer.__subclasses__():
            if self.optimizer.__class__.__name__ == cls.__name__:
                optimizer_class = self.optimizer.__class__

        # Get Diff equations details
        if self.__class__.__name__ == "Solver1D":
            sample_solution = get_sample_solution1D(self)
        elif self.__class__.__name__ == "Solver2D":
            sample_solution = get_sample_solution2D(self)

        diff_equation_details = {
            "equation": get_source(self.diff_eqs),
            "parameters": get_parameters(self.diff_eqs),
            "conditions": get_conditions(self.conditions),
            "generator": get_generator(self.generator),
            "sample_solution": sample_solution,
            "sample_loss": self.metrics_history['valid_loss'],
            "criterion": get_source(self.criterion),
            "networks": get_networks(self),
            "optimizer": {
                "name": self.optimizer.__class__.__name__,
                "params": self.optimizer.state_dict()['param_groups']
            },
        }

        save_dict = {
            "metrics": self.metrics_fn,
            "criterion": self.criterion,
            "conditions": self.conditions,
            "global_epoch": self.global_epoch,  # loss_history
            "nets": self.nets,
            "optimizer": self.optimizer,
            "optimizer_state": self.optimizer.state_dict(),
            "optimizer_class": optimizer_class,
            "diff_eqs": self.diff_eqs,
            "diff_equation_details": diff_equation_details,
            "generator": self.generator,
            "train_loss_history": self.metrics_history['train_loss'],
            "valid_loss_history": self.metrics_history['valid_loss'],
            "type": self.__class__,
            "type_name": self.__class__.__name__,
            "solver": self
        }

        # Save remote if needed
        if save_to_hub:
            cache_dir = create_cache_dir()
            solution_file_path = os.path.join(cache_dir, "solution_to_upload")
            # Save solution in temp file
            # with tempfile.NamedTemporaryFile() as tmp_file:
            with open(solution_file_path, 'wb') as tmp_file:
                dill.dump(save_dict, tmp_file)
                # tmp_file.flush()

                # Save remote
                print("Saving solution to:", NEURODIFF_API_URL)
                project = None
                if "/" in name:
                    project = name.split('/')[0]
                if project is None:
                    print("Default project will be used to save solution")
                else:
                    # Check if user has access to project
                    url = NEURODIFF_API_URL + \
                        "/projects/check_access/{project}"
                    response = requests.get(
                        url.format(project=project),
                        headers=_make_api_headers()
                    )
                    if not response.ok:
                        # response.raise_for_status()
                        print("You do not have access to the project:", project)

                # Upload the solution
                url = NEURODIFF_API_URL + "/solutions/upload"
                solution = {
                    "name": name,
                    "description": name,
                    "diff_equation_details": save_dict["diff_equation_details"],
                    "type_name": save_dict["type_name"]
                }
                response = requests.post(
                    url,
                    data=solution,
                    files={"file": open(tmp_file.name, "rb"), "solution": (
                        "solution.json", json.dumps(solution))},
                    headers=_make_api_headers()
                )
                if not response.ok:
                    print("Could not upload solution")

        else:
            # Save solution locally
            with open(path, 'wb') as file:
                dill.dump(save_dict, file)

    # Loading saved attributes into new solver object

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
            print("Loading solution from:", NEURODIFF_API_URL, name)
            url = NEURODIFF_API_URL + "/solutions/download"
            solution_file_path = get_file(url, name)
        else:
            print("Loading solution from:", path)
            solution_file_path = path
        # Load the solution
        with open(solution_file_path, 'rb') as file:
            load_dict = dill.load(file)

        # Loading user defined generator and extracting time domain information
        if config.train_generator == None:
            train_generator = load_dict['generator']['train'].generator
            valid_generator = load_dict['generator']['valid'].generator
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
                optimizer = load_dict['optimizer_class'](
                    chain.from_iterable(n.parameters() for n in nets))
                optimizer.load_state_dict(load_dict['optimizer_state'])
            else:
                optimizer = load_dict['optimizer']

            # As older network/optimizer is loaded load loss history as well
            train_loss = load_dict['train_loss_history']
            valid_loss = load_dict['valid_loss_history']
        else:
            # Declare a flag to check if optimizer is passed as a class. Link parameters and load state to the new class
            check_flag = False

            if config.optimizer_params is not None:
                optimizer = config.optimizer(chain.from_iterable(
                    n.parameters() for n in nets), **config.optimizer_params)
                train_loss = load_dict['train_loss_history']
                valid_loss = load_dict['valid_loss_history']
                check_flag = True

            # In case of custom optimizer or optimizer externally linked to the network, load complete optimizer
            if check_flag == False:
                optimizer = load_dict['optimizer']

        # Initiate a new Solver
        if load_dict["type_name"] == "Solver1D":
            # t min/max
            t_min = load_dict['generator']['train'].__dict__[
                'generator'].__dict__['t_min']
            t_max = load_dict['generator']['train'].__dict__[
                'generator'].__dict__['t_max']

            solver = cls(ode_system=de_system,
                         conditions=cond,
                         criterion=load_dict['criterion'],
                         metrics=load_dict['metrics'],
                         nets=nets,
                         optimizer=optimizer,
                         train_generator=train_generator,
                         valid_generator=valid_generator,
                         t_min=t_min,
                         t_max=t_max)
        elif load_dict["type_name"] == "Solver2D":
            xy_min = load_dict['generator']['train'].__dict__[
                'generator'].__dict__['xy_min']
            xy_max = load_dict['generator']['train'].__dict__[
                'generator'].__dict__['xy_max']

            solver = cls(pde_system=de_system,
                         conditions=cond,
                         xy_min=xy_min,
                         xy_max=xy_max,
                         nets=nets,
                         train_generator=train_generator,
                         valid_generator=valid_generator,
                         optimizer=optimizer,
                         criterion=load_dict['criterion'],
                         metrics=load_dict['metrics'])

        solver.metrics_history['train_loss'] = train_loss
        solver.metrics_history['valid_loss'] = valid_loss

        try:
            solver.diff_eqs_source = load_dict["diff_equation_details"]["equation"]
        except:
            pass
        return solver
