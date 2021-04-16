import os
import dill
import numpy as np
import pathlib
import torch

class SolverConfig():
    conditions = None
    ode_system = None
    nets = None
    optimizer = None
    optimizer_params = {}
    train_generator = None
    valid_generator = None

class PretrainedSolver():

    #Saving selected attributes of model in dict
    def save(self,filename):
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
            "optimizer_state": self.optimizer.state_dict()
            "optimizer_class": optimizer_class,
            "diff_eqs": self.diff_eqs,
            "generator": self.generator,
            "type": self.__class__
        }

        with open(filename,'wb') as file:
            dill.dump(save_dict,file)
			
    #Loading saved attributes into new solver object   

    # Have to add check and warning/error     
    @classmethod		
    def load(cls, path, config=SolverConfig()):
        with open(path,'rb') as file:
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
            if load_dict['optimizer_class'] == None:
                optimizer = load_dict['optimizer']

            else if config.optimizer_params:
                optimizer = optimizer.load_state_dict(load_dict['opimizer_params'])
            # net is new, save optimizer class
        else if type(config.optimizer) == type:
            #get net and params
            optimizer = config.optimizer(,**config.optimizer_params)
        else :
            optimizer = config.optimizer
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
