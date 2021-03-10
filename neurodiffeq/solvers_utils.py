import dill
import numpy as np
from abc import ABC, abstractmethod

class PretrainedSolver(ABC):

    
    def save(self,filename):
	#save_keys = ['diff_eqs','metrics','global_epoch','nets','conditions','criterion','optimizer','generator'] #'criterion','optimizer','generator',
        save_dict = {
        "metrics": self.metrics_fn,
        "criterion": self.criterion,
        "conditions": self.conditions,
        "global_epoch": self.global_epoch, #loss_history
        "nets": self.nets,
        "optimizer": self.optimizer,
        "diff_eqs": self.diff_eqs,
        "generator": self.generator
        }
	#print('this is in solver file')
        with open(filename,'wb') as file:
            dill.dump(save_dict,file)
			
    @classmethod		
    def load(cls, path):
        with open(path,'rb') as file:
            load_dict = dill.load(file)	

        t = load_dict['generator']['train'].get_examples()[0].detach().numpy()
        t_min = np.round(min(t))
        t_max = np.round(max(t))
		
        solver = cls(ode_system = load_dict['diff_eqs'],
                     conditions = load_dict['conditions'],
                     criterion = load_dict['criterion'],
                     metrics = load_dict['metrics'],
                     nets = load_dict['nets'],
                     optimizer = load_dict['optimizer'],
                     train_generator = load_dict['generator']['train'],
                     valid_generator = load_dict['generator']['valid'],
                     t_min = t_min,
                     t_max = t_max)
					 
        return solver
		
		

def save_solver(solver,filename):

    with open(filename,'wb') as file:
        dill.dump(solver,file)
		
def load_solver(path):

    with open(path,'rb') as file:
        solver = dill.load(file)
		
    return solver