# use torch compile: https://docs.openvino.ai/2023.2/pytorch_2_0_torch_compile.html

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

if torch.cuda.is_available():
    device =  torch.device("cuda")    # set the PyTorch Device to the first CUDA device detected. If this breaks change it from "cuda:0" to "cuda"
    print('Torch CUDA device is: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    print('CUDA Version is', torch.version.cuda) 
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print('CUDA not detected, instead using:', device)

torch.manual_seed(42)  # this is so the model i,s deterministic and gives the same results every time  

################################################################ MODEL HYPERPARAMETERS ###################################################################

Hidden_Size = 533    # Hidden_Size: The number of neurons in the hidden state h.
Batch_Size = 2500     # number of chunks to split all the data into. example 10 means train with 10% of the data at one time. Too low is inefficient. too high is slow and memory intensive

# Time Series Model
Times_Series_Model = 'RNN' # choose RNN or LSTM or BDRNN or BDLSTM as model type (BD means bi-directional)
Number_Layers = 2    # Number_Layers: Number of layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.    

Window_Length = 5  # essentially the length of the moving window of input data

Embedding_Dimensions = 10  # number of dimensions for the embedding
Max_Embedding_Norm = None    # the boundary for the embedding vector

Number_Epochs = 300  # number of times all of the data will be analyzed and used to update the neural network
Learning_Rate = 0.00024   # my model 0.00024  austrian 0.00064

################################################################### PRE-PROCESSING #############################################################################

# Load all data from the file
pd_All_Data = pd.read_csv('all data (2).txt', header=None, delimiter=',', engine='python')
pd_All_Data.pop(0)  # Remove first column

# Convert to numpy array
np_All_Data = pd_All_Data.values

# Convert to tensor
All_Data_Tensor = torch.tensor(np_All_Data, dtype=torch.int64).to(device)

# Split into train and test sets (e.g. 80% train, 20% test)
train_size = int(0.8 * len(All_Data_Tensor))
Train_Data_Tensor = All_Data_Tensor[:train_size]
Test_Data_Tensor = All_Data_Tensor[train_size:]

# Creating 1D Tensors
Train_Data_Tensor_1D = Train_Data_Tensor.reshape(-1).to(device)
Test_Data_Tensor_1D = Test_Data_Tensor.reshape(-1).to(device)

# Extract unique classes and then sorting them
Unique_Classes = torch.cat((Train_Data_Tensor_1D, Test_Data_Tensor_1D)).unique().to(device)
Sorted_Classes, _ = torch.sort(Unique_Classes)

# Create a Class Dictionary, meaning a translation dictionary between input values and usable class values 
Class_Dictionary = {class_val.item(): idx for idx, class_val in enumerate(Sorted_Classes)} # for this line alone (no if statement), if 0:0 then you know zeros are used 
if list(Class_Dictionary)[0] != 0:     # Ensuring we can zero pad successfully
    Class_Dictionary = {class_val.item(): idx + 1 for idx, class_val in enumerate(Sorted_Classes)} # +1 here is zero padding
    Class_Dictionary[0] = 0

# After loading your data, add this remapping code
unique_classes = sorted(list(set(Class_Dictionary.values())))
new_class_mapping = {old_val: new_val for new_val, old_val in enumerate(unique_classes)}

# Update the Class_Dictionary with new consecutive indices
Class_Dictionary = {k: new_class_mapping[v] for k, v in Class_Dictionary.items()}

# Update the Remapping_Tensor
Remapping_Tensor = torch.zeros(max(Class_Dictionary.keys()) + 1, dtype=torch.long).to(device)
for k, v in Class_Dictionary.items():
    Remapping_Tensor[k] = v

# Add debug prints
print(f"Number of unique classes: {len(unique_classes)}")
print(f"Max value in remapped Class_Dictionary: {max(Class_Dictionary.values())}")
print(f"Sample of remapped values: {list(Class_Dictionary.items())[:5]}")

# Remapping the original class labels with the new usable class label values. using a remapping tensor
Train_Data_Remapped = Remapping_Tensor[Train_Data_Tensor].to(device) 
Test_Data_Remapped  = Remapping_Tensor[Test_Data_Tensor].to(device) 

# Setting up the Slicing method
def Slicing(Unsliced_Data, Window_Length):
    Slices = []
    for row in Unsliced_Data:
        for i in range(row.size(0) - Window_Length):
            X_Slice = row[i:i + Window_Length]
            Y_Slice = row[i + Window_Length]
            Slices.append((X_Slice, Y_Slice))
    return Slices

# Running the Slicing Method
Train_Sliced = Slicing(Train_Data_Remapped, Window_Length)
Test_Sliced  = Slicing(Test_Data_Remapped,  Window_Length)

# Preparing the DataLoader the input is a list (tuple) of data (inputs) and labels (ground truth)
Train_Loader = DataLoader (dataset = Train_Sliced, batch_size = Batch_Size, shuffle = True, drop_last = True)   # pin_memory = True is for CUDA only
Test_Loader  = DataLoader (dataset = Test_Sliced,  batch_size = Batch_Size, shuffle = True, drop_last = True)   # pin_memory = True is for CUDA only

##################################################### CALCULATE MODEL TRAINING VARIABLES #################################################################
Number_Classes = int(len(Class_Dictionary)) 
Number_Iterations = int(len(Train_Sliced) // Batch_Size) * int(Number_Epochs)   # the // truncates the division, because of drop_last = True
Five_Percent_Progress = int(Number_Iterations / 20)
if Hidden_Size == None:
    Hidden_Size = int(Number_Classes * 0.25)

############################################################# DEFINE THE MODEL ###########################################################################
class Model_Class(nn.Module):
    def __init__(self): 
        super(Model_Class, self).__init__()   # super function is called to inherit everything from nn.Module   super().__init__()  
        # Ensure embedding size matches the maximum possible index + 1
        max_index = max(Class_Dictionary.values())
        self.embedding = nn.Embedding(max_index + 1, Embedding_Dimensions, padding_idx=0)
        self.Batch_Norm1 = nn.BatchNorm1d(num_features = Window_Length, affine = True)
        self.linear_Output = nn.Linear(Hidden_Size, Number_Classes)
        self.time_series = nn.RNN(Embedding_Dimensions, Hidden_Size, Number_Layers, batch_first = True, nonlinearity = 'tanh', bidirectional = False)  # generally for RNN, TanH is better than ReLu
        self.HiddenState = torch.zeros(Number_Layers, Batch_Size, Hidden_Size).to(device)            
    def forward(self, x):   #  x is Model_Input used in this code. 
        # Add safety check
        if x.max() >= self.embedding.num_embeddings:
            raise ValueError(f"Input contains index {x.max()} which is >= embedding size {self.embedding.num_embeddings}")
        x = self.embedding(x)
        x, h0 = self.time_series(x, self.HiddenState)
        x = self.Batch_Norm1(x) # x.float()    .long()   # h0 is of dimenensions: batch size, layer, feature
        x = self.linear_Output(x[:, -1])  
        return x
######################################## INSTANTIATE THE MODEL, DEFINE LOSS FUNCTION & OPTMIZATION FUNCTION ##############################################
Main_Model = Model_Class().to(device)
criterion = nn.CrossEntropyLoss()     # the cross entropy loss computes softmax as well. cross entropy loss also does not require one-hot-encode of features
optimizer = torch.optim.Adam(Main_Model.parameters(), lr = Learning_Rate)   # good range of learning rate is 0.001 to 0.0001 
########################################################### TRAIN THE MODEL ##############################################################################
Iterations = int(0)
Start_Time = time.time()
Main_Model.train() 
for epoch in range(Number_Epochs):
    Main_Model.train()  # Set the model to training mode
    correct, total = 0, 0  # Initialize correct and total for accuracy calculation
    for (X_Train, Y_Train) in Train_Loader:
        optimizer.zero_grad()  # Clear gradients
        Model_Input_Train = X_Train.view(Batch_Size, Window_Length)  # Reshape input
        Y_Model_Train = Main_Model(Model_Input_Train)  # Forward pass
        Train_Loss = criterion(Y_Model_Train, Y_Train)  # Calculate loss
        Train_Loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Calculate accuracy for the current batch
        predicted = torch.max(Y_Model_Train.data, 1)[1]  # Get predicted classes
        total += Y_Train.size(0)  # Update total count
        correct += (predicted == Y_Train).sum().item()  # Count correct predictions

    # Calculate overall accuracy for the epoch
    accuracy = (100 * correct / total) if total > 0 else 0  # Avoid division by zero

    # Print epoch results
    print(f'Epoch: {epoch + 1}/{Number_Epochs} | Train Loss: {Train_Loss.item():.6f} | Accuracy: {accuracy:.2f}%')

    # Calculate Accuracy
    if (Iterations % Five_Percent_Progress == 0) or (Iterations + 1 == Number_Iterations):
        Main_Model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            correct, total = 0, 0
            for (X_Test, Y_Test) in Test_Loader:
                Model_Input_Test = X_Test.view(Batch_Size, Window_Length)  # Reshape input
                Y_Model_Test = Main_Model(Model_Input_Test)  # Forward pass
                predicted = torch.max(Y_Model_Test.data, 1)[1]  # Get predicted classes
                total += Y_Test.size(0)  # Update total count
                correct += (predicted == Y_Test).sum().item()  # Count correct predictions

            accuracy = (100 * correct / total)  # Calculate accuracy

    Iterations += 1  # Increment Iterations

###################################################### TEST THE MODEL ################################################################################
    if (Iterations % Five_Percent_Progress == 0) or (Iterations + 1 == Number_Iterations):  # Calculate Accuracy 
        Main_Model.eval()     # put the model into evaluation mode
        with torch.no_grad():  # disables backpropagdation which reduces memory usage  can also maybe use torch.interface_mode
            correct, total, accuracy = float(0), float(0), float(0)
            for (X_Test, Y_Test) in Test_Loader:
                    #X_Test, Y_Test = X_Test.to(device), Y_Test.to(device)
                    
                Model_Input_Test = X_Test.view(Batch_Size, Window_Length)   # for Embedding as the primary input              
                Y_Model_Test = Main_Model(Model_Input_Test)         
                Test_Loss = criterion(Y_Model_Test, Y_Test)
                predicted = torch.max(Y_Model_Test.data,1)[1]
                total += Y_Test.size(0)
                correct += (predicted == Y_Test).sum()
            accuracy = (100 * (correct / total))
            if Iterations + 1 == Number_Iterations: # ensures the epoch number and iteration number line up for the very final iteration
                epoch += 1
                Iterations += 1
            Elapsed_Time = (time.time() - Start_Time) / 60  # this is the time the model has been running so far in minutes from when it first started training
            print('Epoch: {:4.0f} | Iteration: {:5.0f} | Train Loss: {:3.6f} | Test Loss: {:3.6f} | Correct: {:2.0f} | Total: {:2.0f} | Accuracy: {:6.2f} % | Elapsed Time: {:3.1f} minutes'.format(epoch, Iterations, Train_Loss, Test_Loss, correct, total, accuracy, Elapsed_Time))
        Main_Model.train() # puts the model back into training mode for the training phase
    Iterations += 1
   
###################################################### MODEL PREDICTIONS ################################################################################

Main_Model.eval() # Set the model to evaluation mode

# Save location
save_dir = r"C:\Users\sanid\OneDrive\Desktop\ai ml"

# Save state dict
torch.save(Main_Model.state_dict(), f"{save_dir}/Model_Prime_ST.torch")

# Save full model
torch.save(Main_Model, f"{save_dir}/Model_Prime_M.torch")

# Load the model state if continuing training
Main_Model.load_state_dict(torch.load(f"{save_dir}/Model_Prime_M.torch"))

# Fix the file path using any of these methods:
# Method 1: Raw string
with open(r'C:\Users\sanid\OneDrive\Desktop\1.0 Model Generation.py', 'w') as file:
    for key, value in Class_Dictionary.items():
        file.write(f'{key},{value}\n')

# OR Method 2: Forward slashes
with open('C:/Users/sanid/OneDrive/Desktop/1.0 Model Generation.py', 'w') as file:
    for key, value in Class_Dictionary.items():
        file.write(f'{key},{value}\n')

# OR Method 3: Double backslashes
with open('C:\\Users\\sanid\\OneDrive\\Desktop\\1.0 Model Generation.py', 'w') as file:
    for key, value in Class_Dictionary.items():
        file.write(f'{key},{value}\n')






'''
for (X_Pred, Y_Pred) in Pred_Loader:
#   1==1
    with torch.no_grad():
    #inputs = inputs.()
    
        
        #prediction = inputs.view(Batch_Size, Window_Length).to(device)
        #outputs = Main_Model(prediction).to(device)                         
        print('inputs are: ', X_Pred)
        print('input shape is: ', X_Pred.shape)
        print('input data type is:', X_Pred.dtype)
        
        # Raw Logits
        Output_Logits = Main_Model(X_Pred).to(device)  
        print('Output_Logits', Output_Logits)
        
        # Softmax
        Output_Softmax = F.softmax(Output_Logits, dim=1)
        print('Output_Softmax', Output_Softmax)

'''
