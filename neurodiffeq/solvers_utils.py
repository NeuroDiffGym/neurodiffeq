import os
import dill
import numpy as np
import pathlib

class SolverConfig():
    conditions = None
    ode_system = None

class PretrainedSolver():

    #Saving selected attributes of model in dict
    def save(self,filename):
        save_dict = {
            "metrics": self.metrics_fn,
            "criterion": self.criterion,
            "conditions": self.conditions,
            "global_epoch": self.global_epoch, #loss_history
            "nets": self.nets,
            "optimizer": self.optimizer,
            "diff_eqs": self.diff_eqs,
            "generator": self.generator,
            "type": self.__class__
        }

        with open(filename,'wb') as file:
            dill.dump(save_dict,file)
			
    #Loading saved attributes into new solver object        
    @classmethod		
    def load(cls, path, config=SolverConfig()):
        with open(path,'rb') as file:
            load_dict = dill.load(file)	

        t = load_dict['generator']['train'].get_examples()[0].detach().numpy()
        t_min = np.round(min(t))
        t_max = np.round(max(t))

        # For 1D
        # params = {"ode_system":load_dict['diff_eqs'],"key":2}
        # solver = cls(**params)
        if config.ode_system == None:
            ode = load_dict['diff_eqs']
        else:
            ode = config.ode_system
        
        if config.conditions == None:
            cond = load_dict['conditions']
        else:
            cond = config.conditions

        solver = cls(ode_system = ode,
                     conditions = cond,
                     criterion = load_dict['criterion'],
                     metrics = load_dict['metrics'],
                     nets = load_dict['nets'],
                     optimizer = load_dict['optimizer'],
                     train_generator = load_dict['generator']['train'],
                     valid_generator = load_dict['generator']['valid'],
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
