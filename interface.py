# interface.py

#  Link model
from model import ScriptCNN as TheModel

#  Link training function
from train import train_model as the_trainer

#  Link prediction function
from predict import classify_images as the_predictor

#  Link dataset and dataloaders
from dataset import CustomDataset as TheDataset
from dataset import get_dataloader as the_dataloader

