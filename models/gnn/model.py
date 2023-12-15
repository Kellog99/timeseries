import torch
import torch.nn as nn

class pre_processing(torch.nn.Module):
    def __init__(self, 
                 in_feat:int, 
                 emb:dict,
                 device):
        super(pre_processing, self).__init__()
        self.in_feat = in_feat
        self.emb = nn.ModuleDict({})
        out = 0
        for key in emb.keys():
            self.emb[key] = nn.Embedding(num_embeddings = emb[key][0],
                                         embedding_dim = emb[key][1], 
                                         device=device)
            out += emb[key][1]
        
        # sommo le tre variabili non categoriche dell'input
        out += 3
        
        self.norm = nn.BatchNorm1d(265)
        self.linear = nn.Sequential(nn.Linear(in_features = out, 
                                              out_features = 128),
                                    nn.ReLU(), 
                                    nn.Linear(in_features = 128,
                                              out_features = 128),
                                    nn.ReLU(),
                                    nn.Linear(in_features = 128,
                                              out_features = out-1)
        )
        
        
    def forward(self, x):
        out = []
        for i,key in enumerate(self.emb.keys()):
            out.append(self.emb[key](x[:, :,i].int().to(x.device.type)))
        out = torch.cat(out,-1)    
        out = self.norm(out.float())
        out = torch.cat((out, x[:,:,-3:]),-1)
        out = self.linear(out.float())
        out = torch.cat((out, x[:,:,-1:]),-1)
        
        return out.float()

    

class my_gcnn(torch.nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 relu: bool = True):
        super(my_gcnn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, 
                             out_channels, 
                             bias = False)
        self.relu = nn.ReLU()
        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        

    def forward(self,
                x0: tuple) -> torch.tensor:
        x, A = x0
        # Inserisco una threshold
        # se i valori sono più piccoli di un tot significa che non ci è connessione
        D = torch.sign(self.relu(A - 1e-15))
        D = torch.diag((1+torch.sum(D , 1))**(-0.5))
        node_start = A.shape[0]
        node_end = A.shape[1]
        
        if node_start == node_end:
            A_tilde = A + torch.diag(torch.ones(node_start)).to(A.device.type)
            D1 = D2 = D
        else:
            I = torch.diag(torch.ones(node_end))[node_end-node_start:]
            A_tilde = A+I.to(A.device.type)
            D1 = D**2
            D2 = torch.diag(torch.ones(node_end)).to(A.device.type)
            
        x = self.lin(x)
        out = torch.matmul(D1, A_tilde)
        out = torch.matmul(out,D2)
        out = self.act(torch.matmul(out,x))
        
        return (out.float(), A)

    
class GNN(torch.nn.Module):
    def __init__(self, 
                 past_steps:int, 
                 future_steps:int, 
                 future_channel:int, 
                 embs: list, 
                 cat_emb_dim:int, 
                 hidden_GNN:int, 
                in_feat:int, 
                past: int, 
                future: int,
                emb:dict,
                device,
                nfeat_out_gnn:int = 16,
                num_layer1:int = None,
                num_layer2:int = None,
                hid_out_features1:int = None,
                hid_out_features2:int = None):
        """ Custom GNN 
        
        Args:
            past_steps (int):  number of past datapoints used 
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            embs (List): list of the initial dimension of the categorical variables
            cat_emb_dim (int): final dimension of each categorical variable
            hidden_RNN (int): hidden size of the RNN block
            num_layers_RNN (int): number of RNN layers
            sum_emb (bool): if true the contribution of each embedding will be summed-up otherwise stacked
            out_channels (int):  number of output channels
            activation (str, optional): activation fuction function pytorch. Default torch.nn.ReLU
            remove_last (bool, optional): if True the model learns the difference respect to the last seen point
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): we can use quantile loss il len(quantiles) = 0 (usually 0.1,0.5, 0.9) or L1loss in case len(quantiles)==0. Defaults to [].
            dropout_rate (float, optional): dropout rate in Dropout layers
            use_bn (bool, optional): if true BN layers will be added and dropouts will be removed
            use_glu (bool,optional): use GLU for feature selection. Defaults to True.
            glu_percentage (float, optiona): percentage of features to use. Defaults to 1.0.
            n_classes (int): number of classes (0 in regression)
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.

        """
        super(GNN, self).__init__()
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.future_channel = future_channel
        self.embs = embs 
        self.cat_emb_dim = cat_emb_dim
        self.hidden_GNN = hidden_GNN


        self.device = device
        self.out_gnn = nfeat_out_gnn
        self.hid_out_features1 = hid_out_features1
        self.hid_out_features2 = hid_out_features2
        self.num_layer1 = num_layer1
        self.num_layer2 = num_layer2
        
        self.in_feat = in_feat         # numero di features di ogni nodo prima del primo GAT        
        self.past = past 
        self.future = future

        
        ########## PREPROCESSING PART #############
        out = 3
        for key in emb.keys():
            out += emb[key][1]
        self.out_preprocess = out
        self.pre_processing = pre_processing(in_feat = in_feat, 
                                            emb = emb, 
                                            device = device)
        
        N = past + future
        self.M = torch.nn.Parameter(torch.randn(N, N)) # Matrice di adiacenza
        self.mask = torch.tril(torch.ones(N, N))-torch.diag(torch.ones(N))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        
        
        ########## FIRST GAT PART #############
        # B * P, O ---> B * T, O-1
        if num_layer1 in [0,1]:
            self.gnn1 = my_gcnn(in_channels = self.out_preprocess, 
                                out_channels = self.out_gnn, 
                                relu = False)     
        else:
            layers = [my_gcnn(in_channels = self.out_preprocess, 
                                      out_channels = hid_out_features1)]
            
            for _ in range(max(0,num_layer1-2)):
                layers.append(my_gcnn(in_channels = hid_out_features1, 
                                      out_channels = hid_out_features1))
                
            layers.append(my_gcnn(in_channels = hid_out_features1, 
                                  out_channels = self.out_gnn,
                                  relu= False))    
            
            self.gnn1 = nn.Sequential(*layers)
            
        ########## SECOND GAT PART #############
        
        if num_layer2 in [0, 1]:
            self.gnn2 = my_gcnn(in_channels = self.out_preprocess-1, 
                                out_channels = self.out_gnn, 
                                relu = False)
        else:
            layers = [my_gcnn(in_channels = self.out_preprocess-1, 
                                      out_channels = hid_out_features2)]
            
            for _ in range(max(0,num_layer2-2)):
                layers.append(my_gcnn(in_channels = hid_out_features2, 
                                      out_channels = hid_out_features2))
                
            layers.append(my_gcnn(in_channels = hid_out_features2, 
                                  out_channels = self.out_gnn, 
                                  relu = False))    
            
            self.gnn2 = nn.Sequential(*layers)
        
        self.prop_gnn = my_gcnn(in_channels = self.out_gnn, 
                                out_channels = self.out_gnn, 
                                relu = False)
        self.propagation = nn.Sequential(nn.Flatten(-2),
                                         nn.BatchNorm1d(self.out_gnn*self.future),
                                         nn.Linear(in_features = self.out_gnn*self.future, 
                                                   out_features = 128),
                                         nn.ReLU(), 
                                         nn.Linear(in_features = 128,
                                                   out_features = 128),
                                         nn.ReLU(),
                                         nn.Linear(in_features = 128,
                                                   out_features = self.future))
               
    def forward(self, data):

        A = self.relu(torch.matmul(self.M,self.M.T))
        A = self.softmax(A.masked_fill(self.mask.to(A.device.type) == 0, float('-inf')))
        A = A.masked_fill(self.mask.to(A.device.type) == 0, 0)
        
        # pre-processing dei dati
        x = self.pre_processing(data)
        
        # creation of the 2 subgraph
        sg1 = x[:,:self.past,:]
        sg2 = x[:,self.past:,:]

        ########## GNN processing ######################
        
        x1, _ = self.gnn1((sg1, A[:self.past, :self.past]))
        x2, _ = self.gnn2((sg2[:,:,:-1], A[self.past:, self.past:]))
        x = torch.cat((x1,x2),-2)
        
        ########## PROPAGATION PART ####################
        D = torch.sign(self.relu(A - 1e-15))
        D = torch.diag((1+torch.sum(D , 1))**(-0.5))
        
        future, _ = self.prop_gnn((x, A[self.past:,:]))
        future = self.propagation(future)
        
        ########## PREPARE THE DIRICHLET ENERGY ##########
        x = torch.matmul(D,x)
        x = torch.cdist(x,x)        
        
        return future.float(), x, A