from timeseries.utils.typing import Module
from typing import Any
from pydantic import Field
from timeseries.configuration import BaseLoss
from timeseries.configuration import ThreeQuantileConfiguration, PositiveInt, NonNegativeInt, Probability, Quantile
from timeseries.register import ModelFactory
from typing import Any, ClassVar
from torch import  nn
import torch
from timeseries.base import Base
from timeseries.utils.torch import Permute, get_activation
from typing import List
import numpy as np
import logging


class ConfigMyModel(ThreeQuantileConfiguration):
    """Convolutional and recurrent forecasting architecture."""
    past_steps: PositiveInt = Field(..., description='Number of past observations in each input window.')
    future_steps: PositiveInt = Field(..., description='Number of future observations to predict.')
    past_channels: PositiveInt = Field(..., description='Number of observed numerical input features.')
    future_channels: NonNegativeInt = Field(..., description='Number of known future numerical features; zero is allowed.')
    embs: list[PositiveInt] = Field(..., description='Cardinalities of the categorical input features.')
    cat_emb_dim: PositiveInt = Field(..., description='Embedding dimension for each categorical variable.')
    hidden_RNN: PositiveInt = Field(..., description='Hidden recurrent state dimension.')
    num_layers_RNN: PositiveInt = Field(..., description='Number of recurrent layers.')
    kind: str = Field(..., description='Kind.')
    kernel_size: PositiveInt = Field(..., description='Convolution or moving-average kernel width.')
    sum_emb: bool = Field(..., description='Sum categorical embeddings instead of concatenating them.')
    out_channels: PositiveInt = Field(..., description='Number of target variables.')
    activation: str = Field('torch.nn.ReLU', description='Fully qualified torch activation class name.')
    remove_last: bool = Field(False, description='Predict differences from the last observed target value.')
    persistence_weight: float = Field(0.0, description='Weight applied to the persistence-based loss penalty.', ge=0)
    loss_type: BaseLoss = Field('l1', description='Training loss selection.')
    quantiles: list[Quantile] | None = Field(None, description='Three increasing output quantiles, or None/empty for point forecasts.')
    dropout_rate: Probability = Field(0.1, description='Dropout rate.')
    use_bn: bool = Field(False, description='Use batch normalization in place of dropout.')
    use_glu: bool = Field(True, description='Enable gated feature selection.')
    glu_percentage: float = Field(1.0, description='Target fraction of features retained by gated selection.', ge=0, le=1)
    n_classes: NonNegativeInt = Field(0, description='Number of classes; zero selects regression.')
    optim: str | None = Field(None, description='Torch optimizer class name; None selects Adam.')
    optim_config: dict[str, Any] | None = Field(None, description='Keyword arguments for the optimizer.')
    scheduler_config: dict[str, Any] | None = Field(None, description='Keyword arguments for the StepLR scheduler.')




torch.autograd.set_detect_anomaly(True)

class GLU(Module):
    def __init__(self, d_model: int):
        """Gated Linear Unit, 'Gate' block in TFT paper 
        Sub net of GRN: linear(x) * sigmoid(linear(x))
        No dimension changes

        Args:
            d_model (int): model dimension
        """
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated Linear Unit
        Sub net of GRN: linear(x) * sigmoid(linear(x))
        No dimension changes: [bs, seq_len, d_model]

        Args:
            x (torch.Tensor)

        Returns:
            torch.Tensor
        """

        ##here comes something like BSxL
        x1 = (self.activation(self.linear(x.unsqueeze(2)))/6.0).squeeze()
        out = x1*x #element-wise multiplication
        
        ##get the score
        score = torch.sign(x1).mean()
        return out,score

class Block(Module):
    def __init__(self,input_channels:int,kernel_size:int,output_channels:int,input_size:int,sum_layers:bool ):

        super(Block, self).__init__()

        self.dilations = nn.ModuleList()
        self.steps = int(np.floor(np.log2(input_size)))-1

        if self.steps <=1:
            self.steps = 1
       
        for i in range(self.steps):
            #dilation
            self.dilations.append(nn.Conv1d(input_channels, output_channels, kernel_size, stride=1,padding='same',dilation=2**i))
            s = max(2**i-1,1)
            k = 2**(i+1)+1
            p = int(((s-1)*input_size + k - 1)/2)
            self.dilations.append(nn.Conv1d(input_channels, output_channels, k, stride=s,padding=p))

        self.sum_layers = sum_layers
        mul = 1 if sum_layers else self.steps*2 
        self.conv_final = nn.Conv1d(output_channels*mul, output_channels*mul, kernel_size, stride=1,padding='same')
        self.out_channels = output_channels*mul
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = Permute()(x)
        outputs = []
        for i in range(self.steps):

            outputs.append(self.dilations[i](x))

        if self.sum_layers:
            tmp = torch.stack(outputs).sum(dim=0)
        else:
            tmp = torch.cat(outputs,1)
        
        return Permute()(tmp)

@ModelFactory.register(
    name='MyModel',
    description='Convolutional and recurrent forecasting architecture.',
    id='mymodel', domain='forecasting',
)
class MyModel(Base):
    CONFIG_T: ClassVar[type[ConfigMyModel]] = ConfigMyModel

    def __init__(
        self,
        past_steps: int,
        future_steps: int,
        past_channels: int,
        future_channels: int,
        embs: list[int],
        cat_emb_dim: int,
        hidden_RNN: int,
        num_layers_RNN: int,
        kind: str,
        kernel_size: int,
        sum_emb: bool,
        out_channels: int,
        activation: str = 'torch.nn.ReLU',
        remove_last: bool = False,
        persistence_weight: float = 0.0,
        loss_type: BaseLoss = 'l1',
        quantiles: list[float] | None = None,
        dropout_rate: float = 0.1,
        use_bn: bool = False,
        use_glu: bool = True,
        glu_percentage: float = 1.0,
        n_classes: int = 0,
        optim: str | None = None,
        optim_config: dict[str, Any] | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        """ Custom encoder-decoder 
        
        Args:
            past_steps (int):  number of past datapoints used 
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            embs (List): list of the initial dimension of the categorical variables
            cat_emb_dim (int): final dimension of each categorical variable
            hidden_RNN (int): hidden size of the RNN block
            num_layers_RNN (int): number of RNN layers
            kind (str): one among GRU or LSTM
            kernel_size (int): kernel size in the encoder convolutional block
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
        quantiles = [] if quantiles is None else quantiles
        if activation == 'torch.nn.SELU':
            logging.info('SELU do not require BN')
            use_bn = False
        activation_factory = get_activation(activation)
        
        super().__init__(
            past_steps=past_steps,
            future_steps=future_steps,
            past_channels=past_channels,
            future_channels=future_channels,
            out_channels=out_channels,
            quantiles=quantiles,
            loss_type=loss_type,
            persistence_weight=persistence_weight,
            n_classes=n_classes,
            optim=optim,
            optim_config=optim_config,
            scheduler_config=scheduler_config,
        )
        #self.device = get_device()
        self.num_layers_RNN = num_layers_RNN
        self.hidden_RNN = hidden_RNN
        self.embs = nn.ModuleList()
        self.sum_emb = sum_emb
        self.kind = kind
        self.use_glu = use_glu
        self.glu_percentage = torch.tensor(glu_percentage).to(self.device)
        self.remove_last = remove_last
            #assert out_channels==1, "Classification require only one channel"
        
        emb_channels = 0

        for k in embs:
            self.embs.append(nn.Embedding(k+1,cat_emb_dim))
            emb_channels+=cat_emb_dim

        if sum_emb and (emb_channels>0):
            emb_channels = cat_emb_dim
            logging.info('Using sum')
        else:
            logging.info('Using stacked')

        if self.use_glu:
            self.past_glu = nn.ModuleList()
            self.future_glu = nn.ModuleList()
            for i in range(past_channels):
                self.past_glu.append(GLU(1))
            
            for i in range(future_channels):
                self.future_glu.append(GLU(1))
    
        self.initial_linear_encoder =  nn.Sequential(Permute(),
                                                    nn.Conv1d(past_channels, (past_channels+hidden_RNN//4)//2, kernel_size, stride=1,padding='same'),
                                                    activation_factory(),
                                                    nn.BatchNorm1d(  (past_channels+hidden_RNN//4)//2) if use_bn else nn.Dropout(dropout_rate) ,
                                                    nn.Conv1d( (past_channels+hidden_RNN//4)//2, hidden_RNN//4, kernel_size, stride=1,padding='same'),
                                                    Permute())

        self.initial_linear_decoder =   nn.Sequential(Permute(),
                                                    nn.Conv1d(future_channels, (future_channels+hidden_RNN//4)//2, kernel_size, stride=1,padding='same'),
                                                    activation_factory(),
                                                    nn.BatchNorm1d(  (future_channels+hidden_RNN//4)//2) if use_bn else nn.Dropout(dropout_rate) ,
                                                    nn.Conv1d( (future_channels+hidden_RNN//4)//2, hidden_RNN//4, kernel_size, stride=1,padding='same'),
                                                    Permute())
        self.conv_encoder = Block(emb_channels+hidden_RNN//4,kernel_size,hidden_RNN//2,self.past_steps,sum_emb)
        
        #nn.Sequential(Permute(), nn.Conv1d(emb_channels+hidden_RNN//8, hidden_RNN//8, kernel_size, stride=1,padding='same'),Permute(),nn.Dropout(0.3))
        #import pdb
        #pdb.set_trace()
        if future_channels+emb_channels==0:
            ## occhio che vuol dire che non ho passato , per ora ci metto una pezza e uso hidden dell'encoder
            self.conv_decoder = Block(hidden_RNN,kernel_size,hidden_RNN//2,self.future_steps,sum_emb) 
        else:
            self.conv_decoder = Block(future_channels+emb_channels,kernel_size,hidden_RNN//2,self.future_steps,sum_emb) 
            #nn.Sequential(Permute(),nn.Linear(past_steps,past_steps*2),  nn.PReLU(),nn.Dropout(0.2),nn.Linear(past_steps*2, future_steps),nn.Dropout(0.3),nn.Conv1d(hidden_RNN, hidden_RNN//8, 3, stride=1,padding='same'),   Permute())
        if self.kind=='lstm':
            self.Encoder = nn.LSTM(input_size= self.conv_encoder.out_channels,#, hidden_RNN//4,
                                   hidden_size=hidden_RNN//2,
                                   num_layers = num_layers_RNN,
                                   batch_first=True,bidirectional=True)
            self.Decoder = nn.LSTM(input_size= self.conv_decoder.out_channels,#, hidden_RNN//4,
                                   hidden_size=hidden_RNN//2,
                                   num_layers = num_layers_RNN,
                                   batch_first=True,bidirectional=True)
        elif self.kind=='gru':
            self.Encoder = nn.GRU(input_size=self.conv_encoder.out_channels,#, hidden_RNN//4,
                                  hidden_size=hidden_RNN//2,
                                  num_layers = num_layers_RNN,
                                  batch_first=True,bidirectional=True)
            self.Decoder = nn.GRU(input_size= self.conv_decoder.out_channels,#, hidden_RNN//4,
                                  hidden_size=hidden_RNN//2,
                                  num_layers = num_layers_RNN,
                                  batch_first=True,bidirectional=True)
        else:
            print('Specify kind= lstm or gru please')
        self.final_linear = nn.ModuleList()
        for _ in range(out_channels*self.mul):
            self.final_linear.append(nn.Sequential(nn.Linear(hidden_RNN+emb_channels+future_channels,hidden_RNN*2), 
                                            activation_factory(),
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.BatchNorm1d(hidden_RNN*2) if use_bn else nn.Dropout(dropout_rate) ,
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.Linear(hidden_RNN*2,hidden_RNN),
                                            activation_factory(),
                                             Permute() if use_bn else nn.Identity() ,
                                            nn.BatchNorm1d(hidden_RNN) if use_bn else nn.Dropout(dropout_rate) ,
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.Linear(hidden_RNN,hidden_RNN//2),
                                            activation_factory(),
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.BatchNorm1d(hidden_RNN//2) if use_bn else nn.Dropout(dropout_rate) ,
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.Linear(hidden_RNN//2,hidden_RNN//4),
                                            activation_factory(),
                                            nn.Linear(hidden_RNN//4,1)))

    def training_step(self, batch, batch_idx: int = 0):
        """
        
        :meta private:
        """
        y_hat,score = self(batch)
        return self.compute_loss(batch, y_hat)
    
    def validation_step(self, batch, batch_idx: int = 0):
        """
        
        :meta private:
        """
        y_hat,score = self(batch)
        return self.compute_loss(batch, y_hat)

    def forward(self, batch):
        """It is mandatory to implement this method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        x =  batch['x_num_past'].to(self.device)
        if 'x_cat_future' in batch.keys():
            cat_future = batch['x_cat_future'].to(self.device)
        if 'x_cat_past' in batch.keys():
            cat_past = batch['x_cat_past'].to(self.device)
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_future'].to(self.device)
            xf = torch.clone(x_future)
        else:
            x_future = None     
            
        if self.remove_last:
            idx_target = batch['idx_target'][0]

            x_start = x[:,-1,idx_target].unsqueeze(1)
            ##BxC
            x[:,:,idx_target]-=x_start        

        ## first GLU
        score = 0
        xp =  torch.clone(x)
        
        if self.use_glu:
            score_past_tot = 0
            score_future_tot = 0
            
            for i in range(len(self.past_glu)):
                x[:,:,i],score = self.past_glu[i](xp[:,:,i])
                score_past_tot+=score
            score_past_tot/=len(self.past_glu)
            
            if x_future is not None:
                for i in range(len(self.future_glu)):
                    x_future[:,:,i],score = self.future_glu[i](xf[:,:,i])
                    score_future_tot+=score
                score_future_tot/=len(self.future_glu)
            score = 0.5*(score_past_tot+score_future_tot)
        tmp = [self.initial_linear_encoder(x)]
        
        for i in range(len(self.embs)):
            if self.sum_emb:
                if i>0:
                    tmp_emb+=self.embs[i](cat_past[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_past[:,:,i])
            else:
                tmp.append(self.embs[i](cat_past[:,:,i]))
        if self.sum_emb and (len(self.embs)>0):
            tmp.append(tmp_emb)
        #import pdb
        #pdb.set_trace()
        tot = torch.cat(tmp,2)

        out, hidden = self.Encoder(self.conv_encoder(tot))      
        
        tmp = []
        for i in range(len(self.embs)):
            if self.sum_emb:
                if i>0:
                    tmp_emb+=self.embs[i](cat_future[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_future[:,:,i])
            else:
                tmp.append(self.embs[i](cat_future[:,:,i]))   
        if self.sum_emb and (len(self.embs)):
            tmp.append(tmp_emb)
            
        if x_future is not None:
            tmp.append(x_future)

        if len(tmp)>0:
            tot = torch.cat(tmp,2)
            out, _ = self.Decoder(self.conv_decoder(tot),hidden)  
            has_future = True
        else:
            out, _ = self.Decoder(self.conv_decoder(out),hidden)  
            has_future = False
        res = []

        if has_future:
            decoder_features = torch.cat([tot,out],dim=2)
        else:
            decoder_features = out
        #import pdb
        #pdb.set_trace()
        for j in range(self.out_channels*self.mul):
            res.append(self.final_linear[j](decoder_features))

        res = torch.cat(res,2)
        ##BxLxC
        B = res.shape[0]
        res = res.reshape(B,self.future_steps,-1,self.mul)
        if self.remove_last:
            res+=x_start.unsqueeze(1)

        return res, score

    def inference(self, batch:dict)->torch.Tensor:
        
        res, score = self(batch)
        #logging.info(score)  ##????
        return res
