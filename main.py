from Graph import *
from model import *
from torch.nn import Linear
import pandas as pd
import torch, numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler 
from torch_geometric.data import Data
from torch.nn import Linear, Dropout, Parameter, ModuleList, BatchNorm1d
from torch_geometric.data import DataLoader
import torch.nn.functional as F 
import torch.nn as nn
from torch_geometric.nn.conv  import MessagePassing,TransformerConv,GINEConv
from torch_geometric.utils    import softmax
from torch_geometric.nn       import global_add_pool as gdp
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
torch.cuda.empty_cache()
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")


data = pd.read_csv("data/all/id_prop.csv")
idx_list = list(range(len(data)))

# 加载 npz 文件
loaded_data = np.load('graph_data_data_set.npz')
# 提取数据
num_graphs = len(loaded_data) // 4  # 每个图对应 4 个键
# 创建 Data 对象的列表
data_list = []
# 确定图的数量
num_graphs = sum(1 for key in loaded_data.keys() if key.startswith('x_'))

# 遍历所有图
for i in range(num_graphs):
    data = Data(
        x=torch.tensor(loaded_data[f'x_{i}']),
        edge_index=torch.tensor(loaded_data[f'edge_index_{i}']),
        edge_attr=torch.tensor(loaded_data[f'edge_attr_{i}']),
        y=torch.tensor(loaded_data[f'y_{i}']),
        the_idx=loaded_data[f'the_idx_{i}'].item()
    )
    data_list.append(data)
#print(idx_list)
random_num = 11
batch_size = 64	

# 首先进行训练集和测试+验证集的划分
train_idx, test_val = train_test_split(idx_list, train_size=0.8, random_state=random_num)
test_idx, val_idx = train_test_split(test_val, test_size=0.5, random_state=random_num)

# 创建训练集、测试集和验证集
train_graph_list = [data_list[i] for i in train_idx]
test_graph_list = [data_list[i] for i in test_idx]
val_graph_list = [data_list[i] for i in val_idx]

# 打印集的大小
#print(f'Train set size: {len(train_graph_list)}')
#print(f'Test set size: {len(test_graph_list)}')
#print(f'Validation set size: {len(val_graph_list)}')
#
train_loader = DataLoader(dataset=train_graph_list,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(dataset=test_graph_list,batch_size=batch_size,shuffle=False)
val_loader = DataLoader(dataset=val_graph_list,batch_size=batch_size,shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

gnn = TransformerGNN(heads=3).to(device)
model = REACTION_PREDICTOR(inp_dim=64,module1=gnn).to(device)
print(model)

learning_rate=0.0001
num_epochs = 500
weight_decays = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decays)
scheduler = lr_scheduler.StepLR(optimizer, step_size=450, gamma=0.1)
criterion = nn.MSELoss()

best_val_mae = float('inf')  # 初始化最佳验证集 MAE
train_maes = []
train_r2s = []
val_maes = []
val_r2s = []
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    epoch_loss = 0.0
    train_outputs = []
    train_targets = []
    for i, batch in enumerate(train_loader):
        data = batch.to(device)
        target = data.y.to(device)
        output = model(data)
        train_outputs.append(output[0].cpu().detach().numpy())
        train_targets.append(target.cpu().detach().numpy())
        loss = criterion(output[0], target)
        loss.backward()
        optimizer.step()  
        epoch_loss += loss.item()
    scheduler.step()
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    
    train_outputs = np.concatenate(train_outputs)
    train_targets = np.concatenate(train_targets)
    train_mae = mean_absolute_error(train_targets, train_outputs)
    train_r2 = r2_score(train_targets, train_outputs)
    train_maes.append(train_mae)
    train_r2s.append(train_r2)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train MAE: {train_mae:.4f}')
    
    model.eval()
    val_loss = 0.0
    val_outputs = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            data = batch.to(device)
            target = data.y.to(device)
            output = model(data)
            val_loss += criterion(output[0], target).item()
            val_outputs.append(output[0].cpu().detach().numpy())
            val_targets.append(target.cpu().detach().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    val_outputs = np.concatenate(val_outputs)
    val_targets = np.concatenate(val_targets)
    val_mae = mean_absolute_error(val_targets, val_outputs)
    val_r2 = r2_score(val_targets, val_outputs)
    val_maes.append(val_mae)
    val_r2s.append(val_r2)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation MAE: {val_mae:.4f}')
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), 'model/best_model.pth')


data_loss_mae_r2 = pd.DataFrame({
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_mae": train_maes,
    "val_mae": val_maes
})
data_loss_mae_r2.to_csv("./data/loss_results.csv", index=False)

# Evaluate on test set
model.load_state_dict(torch.load('model/best_model.pth'))  # Load the best model parameters
model.eval()
test_outputs = []
test_targets = []
test_idx_list = []
test_encoding = []

with torch.no_grad():
    for batch in test_loader:
        data = batch.to(device)
        target = data.y.to(device)
        output = model(data)
        test_outputs.append(output[0].cpu().detach().numpy())
        test_encoding.append(output[1].cpu().detach().numpy())
        test_targets.append(target.cpu().detach().numpy())
        test_idx_list.append(batch.the_idx)
test_outputs = np.concatenate(test_outputs)
test_targets = np.concatenate(test_targets)
test_idxs = np.concatenate(test_idx_list)
tests_encoding = np.concatenate(test_encoding)
encoding_tests = pd.DataFrame(tests_encoding, columns=[f'Feature_{i}' for i in range(64)])

# 将索引添加为 DataFrame 的一列
material_test_encoding = pd.DataFrame({'idx': test_idxs,"target":test_targets})
material_test_encoding = pd.concat([material_test_encoding, encoding_tests], axis=1)
material_test_encoding.to_csv("./data/test_material_encoding.csv",index=False)

test_mae = mean_absolute_error(test_targets, test_outputs)
test_r2 = r2_score(test_targets, test_outputs)

print()
print("=========================================================")
print(f'Test MAE: {test_mae:.4f}')
print(f'Test R2: {test_r2:.4f}')

results = pd.DataFrame({"id":test_idxs,"targets": test_targets, "prediction": test_outputs})
results.to_csv("./data/test_prediction_results.csv", index=False)


model.eval()
trains_outputs = []
trains_targets = []
trains_idx_list = []
trains_encoding = []
with torch.no_grad():
    for batch in train_loader:
        data = batch.to(device)
        target = data.y.to(device)
        output = model(data)
        trains_outputs.append(output[0].cpu().detach().numpy())
        trains_encoding.append(output[1].cpu().detach().numpy())
        trains_targets.append(target.cpu().detach().numpy())
        trains_idx_list.append(batch.the_idx)

trains_outputs = np.concatenate(trains_outputs)
trains_targets = np.concatenate(trains_targets)
trains_idxs = np.concatenate(trains_idx_list)
trains_encodings = np.concatenate(trains_encoding)
encoding_train = pd.DataFrame(trains_encodings, columns=[f'Feature_{i}' for i in range(64)])

# 将索引添加为 DataFrame 的一列
material_train_encoding = pd.DataFrame({'idx': trains_idxs,"target":trains_targets})
material_train_encoding = pd.concat([material_train_encoding, encoding_train], axis=1)
material_train_encoding.to_csv("./data/train_material_encoding.csv",index=False)

trains_mae = mean_absolute_error(trains_targets, trains_outputs)
trains_r2 = r2_score(trains_targets, trains_outputs)

print("=========================================================")
print(f'Train MAE: {trains_mae:.4f}')
print(f'Train R2: {trains_r2:.4f}')

results = pd.DataFrame({"id":trains_idxs,"targets": trains_targets, "prediction": trains_outputs})
results.to_csv("./data/train_prediction_results.csv", index=False)


model.eval()
vals_outputs = []
vals_targets = []
vals_idx_list = []
vals_encoding = []

with torch.no_grad():
    for batch in val_loader:
        data = batch.to(device)
        target = data.y.to(device)
        output = model(data)
        vals_outputs.append(output[0].cpu().detach().numpy())
        vals_targets.append(target.cpu().detach().numpy())
        vals_idx_list.append(batch.the_idx)
        vals_encoding.append(output[1].cpu().detach().numpy())

vals_outputs = np.concatenate(vals_outputs)
vals_targets = np.concatenate(vals_targets)
vals_idxs = np.concatenate(vals_idx_list)
vals_encodings = np.concatenate(vals_encoding)
encoding_val = pd.DataFrame(vals_encodings, columns=[f'Feature_{i}' for i in range(64)])

# 将索引添加为 DataFrame 的一列
material_val_encoding = pd.DataFrame({'idx': vals_idxs,"target":vals_targets})
material_val_encoding = pd.concat([material_val_encoding, encoding_val], axis=1)
material_val_encoding.to_csv("./data/val_material_encoding.csv",index=False)

vals_mae = mean_absolute_error(vals_targets, vals_outputs)
vals_r2 = r2_score(vals_targets, vals_outputs)

print("=========================================================")
print(f'Val MAE: {vals_mae:.4f}')
print(f'Val R2: {vals_r2:.4f}')

results = pd.DataFrame({"id":vals_idxs,"targets": vals_targets, "prediction": vals_outputs})
results.to_csv("./data/val_prediction_results.csv", index=False)

dataset_encoding = pd.concat([material_train_encoding,material_test_encoding , material_val_encoding],axis=0)
dataset_encoding.to_csv("./data/dataset_encoding.csv",index=False)