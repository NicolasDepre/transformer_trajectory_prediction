
import torch
import wandb
from datetime import datetime


"""
TODO 
    Save model 
"""
class Trainer:

    def __init__(self,model,device,train_data,test_data,val_data,criterion,optimizer,epochs,lr,wandb_config,teacher_forcing=False):

        self.model = model
        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.teacher_forcing = teacher_forcing

        self.loss_evolution = []
        self.test_evolution = []
        self.validation_evolution = []
        self.save_name = f"/waldo/walban/student_datasets/arfranck/model_saves/epochs_{epochs}_lr_{lr}-{datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}.dict"

        wandb_config["save_name"] = self.save_name

        wandb.init(
            project="depren",
            config= wandb_config
        )


    def train(self):

        optimizer = self.optimizer
        criterion = self.criterion
        model = self.model

        for epoch in range(self.epochs):
            loss_evolution = []
            model.train()
            for train_batch in self.train_data:

                # Don't know if batch size is needed

                X_train = train_batch["src"].to(self.device)
                Y_train = train_batch["tgt"].to(self.device)

                optimizer.zero_grad()

                if epoch < self.teacher_forcing:

                    pred = model(X_train,Y_train)

                else: # Training with whole prediction

                    future = None
                    n_next = Y_train.shape[1]
                    for k in range(n_next):
                        pred,future = model(X_train,future,train=False)

                loss = criterion(pred,Y_train)

                loss_evolution.append(loss.item())
                loss.backward()
                optimizer.step()

                wandb.log(
                    {
                        "train_loss": loss
                    }
                )

            self.loss_evolution.append(loss_evolution)
            self.validation()
        self.test()
        torch.save(model.state_dict(), self.save_name,_use_new_zipfile_serialization=False)
        #model_scripted = torch.jit.script(model) # Export to TorchScript
        #model_scripted.save(self.save_name) # Save



    def validation(self):

        print("validate")

        with torch.no_grad():

            model = self.model
            criterion = self.criterion
            model.eval()

            val_loss = []
            for val_batch in self.val_data:

                X_val = val_batch["src"].to(self.device)
                Y_val = val_batch["tgt"].to(self.device)

                future = None
                for k in range(Y_val.shape[1]):
                    pred,future = model(X_val,future,train=False)

                loss = criterion(pred, Y_val)
                val_loss.append(loss)
                wandb.log(
                    {
                        "val_loss": loss
                    }
                )

            self.validation_evolution.append(val_loss)

    def test(self):

        with torch.no_grad():

            model = self.model
            criterion = self.criterion
            model.eval()

            for test_batch in self.test_data:

                X_test = test_batch["src"].to(self.device)
                Y_test = test_batch["tgt"].to(self.device)

                future = None
                for k in range(Y_test.shape[1]):
                    pred, future = model(X_test, future, train=False)

                loss = criterion(pred, Y_test)

                self.test_evolution.append(loss)
                wandb.log(
                    {
                        "test_loss": loss
                    }
                )















