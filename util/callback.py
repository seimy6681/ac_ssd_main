import torch
from torch.nn.modules import Module

class Callback(object):
    def __init__(self,
                 model: Module,
                 monitor: str,
                 path: str,
                 verbose: bool = False) -> None:
        self.model = model
        self.best_score = None
        self.score = 0
        self.monitor = monitor
        self.verbose = verbose
        self.path = path
        
    def __call__(self, value) -> None:
        if "loss" in self.monitor:
            self.score = -value
        else:
            self.score = value
            
        if self.best_score == None:
            self.best_score = self.score
        
    def _save_model(self):
        if self.verbose:
            print(f"save model to {self.path}")
        torch.save(self.model.state_dict(), self.path)


class EarlyStopping(Callback):
    def __init__(self, 
                 model: Module, 
                 patience: int = 10,
                 delta: float = 0.001,
                 monitor: str = "val_loss", 
                 path: str = "checkpoint.pt",
                 verbose: bool = False) -> None:
        super().__init__(model, monitor, path, verbose)
        self.patience = patience
        self.delta = delta
        self.counter = 0
        
    def __call__(self, value) -> bool:
        super().__call__(value)
        
        if self.best_score == self.score:
            return False
        
        if self.best_score + self.delta < self.score:
            self.best_score = self.score
            self.counter = 0
            return False
        else: # early stopping
            if self.counter >= self.patience:
                return True 
            
            self.counter += 1
            if self.verbose:
                print(f"Early stopping {self.counter}/{self.patience}: " +
                      f"{self.monitor} {abs(self.best_score):06f}(best) {abs(self.score):06f}(current)")
            
            
class SaveBestModel(Callback):
    def __init__(self, 
                 model: Module, 
                 monitor: str, 
                 path: str, 
                 verbose: bool = False) -> None:
        super().__init__(model, monitor, path, verbose)
        
    def __call__(self, value) -> bool:
        super().__call__(value)
        
        if self.score >= self.best_score:
            if self.verbose:
                print(f"Saving best model - {self.monitor} {abs(self.best_score):06f} -> {abs(self.score):06f}")
            self._save_model()
            self.best_score = self.score
        
        return False