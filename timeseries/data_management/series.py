import numpy as np
import plotly.express as px
import pandas as pd
from typing import List, Literal, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
pd.options.mode.chained_assignment = None 
from .utils import extend_df, MyDataset, ActionEnum
from datetime import datetime
import logging 
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())      
      
      
class Categorical():
    
    def __init__(self,name:str, frequency: int,duration: List[int], classes: int, action: ActionEnum | Literal["multiplicative", "additive"],  level: List[float]):
        """Class for generating toy categorical data

        Args:
            name (str): name of the categorical signal
            frequency (int): frequency of the signal
            duration (List[int]): duration of each class
            classes (int): number of classes
            action (str): one between additive or multiplicative
            level (List[float]): intensity of each class

        """

        self.name = name 
        self.frequency = frequency
        self.duration = duration
        self.classes = classes
        self.action = ActionEnum(action)
        self.level = level

        self.validate()
    
    def validate(self):
        """Validate, maybe there will be other checks in the future
        
        :meta private:
        """
        if len(self.level) == self.classes:
            pass
        else:
            raise ValueError("Length must match")
        

            

    
    def generate_signal(self,length:int)->None:
        """Generate the resposne signal

        Args:
            length (int): length of the signal
        """
        if self.action == ActionEnum.multiplicative:
            signal = np.ones(length)
        elif self.action == ActionEnum.additive:
            signal = np.zeros(length)
        else:
            raise ValueError("Unsupported categorical action.")
        classes = []    
        _class = 0
        _level = self.level[0]
        _duration = self.duration[0]

        count = 0
        count_freq = 0
        for i in range(length):
            if count_freq%self.frequency == 0:
                signal[i] = _level
                classes.append(_class)
                count+=1
                if count == _duration:
                    #change class
                    count = 0
                    _class+=1
                    count_freq+= _duration
                    _class = _class%self.classes
                    _level = self.level[_class]
                    _duration = self.duration[_class]

                   
            else:
                classes.append(-1)
                count_freq+=1
        
        self.classes_array = classes
        self.signal_array = signal

    def plot(self)->None:
        """Plot the series
        """
        tmp = pd.DataFrame({'time':range(len(self.classes_array)),'signal':self.signal_array,'class':self.classes_array})
        fig = px.scatter(tmp,x='time',y='signal',color='class',title=self.name)
        fig.show()

 
class TimeSeries():
    
    def __init__(self,name:str):
        """Class for generating time series object. If you don't have any time series you can build one fake timeseries using some helping classes (Categorical for instance).


        Args:
            name (str): name of the series
                
                
        Usage:
            For example we can generate a toy timeseries:\n
            - add a multiplicative categorical feature (weekly)\n
            >>> settimana = Categorical('settimanale',1,[1,1,1,1,1,1,1],7,'multiplicative',[0.9,0.8,0.7,0.6,0.5,0.99,0.99])\n
            - an additive montly feature (here a year is composed by 5 months)\n
            >>> mese = Categorical('mensile',1,[31,28,20,10,33],5,'additive',[10,20,-10,20,0])\n
            - a spotted categorical variable that happens every 100 days and lasts 1 day\n
            >>> spot = Categorical('spot',100,[7],1,'additive',[10])\n
            >>> ts = TimeSeries('prova')\n
            >>> ts.generate_signal(length = 5000,categorical_variables = [settimana,mese,spot],noise_mean=1,type=0) ##we can add also noise\n
            >>> ts.plot()\n
        """
        
        self.name = name
        self.scalers_fitted = False

    def __str__(self) -> str:
        return f"Timeseries named {self.name} of length {self.dataset.shape[0]}.\n Categorical variable: {self.cat_var},\n Future variables: {self.future_variables},\n Past variables: {self.past_variables},\n Target variables: {self.target_variables}"
    def __repr__(self) -> str:
        return f"Timeseries named {self.name} of length {self.dataset.shape[0]}.\n Categorical variable: {self.cat_var},\n Future variables: {self.future_variables},\n Past variables: {self.past_variables},\n Target variables: {self.target_variables}"
    
    def _generate_base(self,length:int,type:int=0)-> None:
        """Generate a basic timeseries 

        Args:
            length (int): length
            type (int, optional): Type of the generated timeseries. Defaults to 0.
        """
        if type==0:
            self.base_signal = 10*np.cos(np.arange(length)/(2*np.pi*length/100))
            self.out_vars = 1
        else:
            logging.error('Please implement your own method')
        """
        
        """    
    def generate_signal(self,length:int=5000,categorical_variables:List[Categorical]=[],noise_mean:int=1,type:int=0)->None:
        """This will generate a syntetic signal with a selected length, a noise level and some categorical variables. The additive series are added at the end while the multiplicative series acts on the original signal
        The TS structure will be populated

        Args:
            length (int, optional): length of the signal. Defaults to 5000.
            categorical_variables (List[Categorical], optional): list of Categorical variables. Defaults to [].
            noise_mean (int, optional): variance of the noise to add at the end. Defaults to 1.
            type (int, optional): type of the timeseries (only type=0 available right now). Defaults to 0.
        """
        
       
        dataset = pd.DataFrame({'time':range(length)})
        self._generate_base(length,type)
        signal = self.base_signal.copy()
        additive = np.zeros(length)
        self.cat_var = []
        for c in categorical_variables:
            c.generate_signal(length)
            _signal = c.signal_array
            classes = c.classes_array
            dataset[c.name] = classes
            self.cat_var.append(c.name)
            if c.action == ActionEnum.multiplicative:
                signal*=_signal
            else:
                additive += _signal
        signal+=additive
        dataset['signal'] = signal + noise_mean*np.random.randn(len(signal))
        self.scalers_fitted = False
        self.dataset = dataset
        
        
        self.past_variables = ['signal']
        self.future_variables = []
        self.target_variables = ['signal']
        self.num_var = list(set(self.past_variables).union(set(self.future_variables)).union(set(self.target_variables)))
        
        
    def enrich(self,dataset,columns):
        if columns =='hour':
            dataset[columns] = dataset.time.dt.hour
        elif columns=='dow':
            dataset[columns] = dataset.time.dt.weekday
        elif columns=='month':
            dataset[columns] = dataset.time.dt.month
        elif columns=='minute':
            dataset[columns] = dataset.time.dt.minute
        else:
            logging.info('I can not automatically add column {c} plase update this function accordlyng')

    def load_signal(self,data:pd.DataFrame,enrich_cat:List[str] = [],past_variables:List[str]=[],future_variables:List[str]=[],target_variables:List[str]=[],cat_var:List[str]=[],check_past:bool=True)->None:
        """ This is a crucial point in the data structure. We expect here to have a dataset with time as timestamp.
            There are some checks:
                1- the duplicates will tbe removed taking the first instance
                2- the frequency will the inferred taking the minumum time distance between samples
                3- the dataset will be filled completing the missing timestamps

        Args:
            data (pd.DataFrame): input dataset the column indicating the time must be called `time`
            enrich_cat (List[str], optional): it is possible to let this function enrich the dataset for example adding the standard columns: hour, dow, month and minute. Defaults to [].
            past_variables (List[str], optional): list of column names of past variables not available for future times . Defaults to [].
            future_variables (List[str], optional): list of future variables available for tuture times. Defaults to [].
            target_variables (List[str], optional): list of the target variables. They will added to past_variables by default unless `check_past` is false. Defaults to [].
            cat_var (List[str], optional): list of the categortial variables (same for past and future). Defaults to [].
            check_past (bool, optional): see `target_variables`. Defaults to True.
        """
        
        
        
        dataset = data.copy()
        logging.info('################I will drop duplicates, I dont like them###################')
        dataset.drop_duplicates(subset=['time'],  keep='first', inplace=True, ignore_index=True)

        if dataset.time.diff()[1:].nunique()>1:
            logging.info("#########There are holes in the dataset i will try to extend the dataframe inserting NAN#############3")
            freq = pd.to_timedelta(np.diff(dataset.time).min())
            logging.info(f'#############Detected minumum frequency: {freq}#############')
            dataset = extend_df(dataset.time,freq).merge(dataset,how='left')
            

            
        assert len(target_variables)>0, f'Provide at least one column for target'
        assert 'time'  in dataset.columns, f'The temporal column must be called time'
        if set(target_variables).intersection(set(past_variables))!= set(target_variables): 
            if check_past:
                logging.info('##########I will update past column adding all target columns, if you want to avoid this beahviour please use check_pass as false############')
                past_variables = list(set(past_variables).union(set(target_variables)))
        
        self.cat_var = cat_var
        self.enrich_cat = enrich_cat
        for c in enrich_cat:
            self.cat_var = list(set(self.cat_var+[c]))
                
            if c in dataset.columns:
                logging.info('#########Categorical {c} already present, it will be added to categorical variable but not recomputed#########') 
            else:
                self.enrich(dataset,c)
        self.scalers_fitted = False
        self.dataset = dataset
        self.past_variables =past_variables
        self.future_variables = future_variables
        self.target_variables = target_variables
        self.out_vars = len(target_variables)
        self.num_var = list(set(self.past_variables).union(set(self.future_variables)).union(set(self.target_variables)))
        
    def plot(self):
        """  
        Easy way to control the loaded data
        Returns:
            plotly.graph_objects._figure.Figure: figure of the target variables
        """
      
        logging.info('plotting only target variables')
        tmp = self.dataset[['time']+self.target_variables].melt(id_vars=['time'])
        fig = px.line(tmp,x='time',y='value',color='variable',title=self.name)
        fig.show()
        return fig
    
        
    def create_data_loader(self,data:pd.DataFrame,past_steps:int,future_steps:int,shift:int=0,keep_entire_seq_while_shifting:bool=False,starting_point:Union[None,dict]=None,skip_step:int=1)->MyDataset:
        """ Create the dataset for the training/inference step

        Args:
            data (pd.DataFrame): input dataset, usually a subset of self.data
            past_steps (int): past context length
            future_steps (int): future lags to predict
            shift (int, optional): if >0 the future input variables will be shifted (categorical and numerical). For example for attention model it is better to start with a know value of y and use it during the process. Defaults to 0.
            keep_entire_seq_while_shifting (bool, optional): if the dataset is shifted, you may want the future data be of length future_step+shift (like informer), default false
            starting_point (Union[None,dict], optional): a dictionary indicating if a sample must be considered. It is checked for the first lag in the future (useful in the case your model has to predict only starting from hour 12). Defaults to None.
            skip_step (int, optional): list of the categortial variables (same for past and future). Usual there is a skip of one between two saples but for debugging  or training time purposes you can skip some samples. Defaults to 1.

        Returns:
            MyDataset: class thath extends torch.utils.data.Dataset (see utils)
                keys of a batch:
                y : the target variable(s)
                x_num_past: the numerical past variables
                x_num_future: the numerical future variables
                x_cat_past: the categorical past variables
                x_cat_future: the categorical future variables
                idx_target: index of target features in the past array
        """

        
        x_num_past_samples = []
        x_num_future_samples = []
        x_cat_past_samples = []
        x_cat_future_samples = []
        y_samples = []
        t_samples = []
        
        ##overwrite categorical columns
        for c in self.cat_var:
            self.enrich(data,c)

        for c in self.cat_var:
            data[c] = self.scaler_cat[c].transform(data[c].values.ravel()).flatten()
        for c in self.num_var: 
            data[c] = self.scaler_num[c].transform(data[c].to_numpy().reshape(-1,1)).flatten()
        
        idx_target = []
        for c in self.target_variables:
            idx_target.append(self.past_variables.index(c))
         
        
        x_num_past = data[self.past_variables].values
        if len(self.future_variables)>0:
            x_num_future = data[self.future_variables].values
        if len(self.cat_var)>0:
            x_cat = data[self.cat_var].values
        y_target = data[self.target_variables].values
        t = data["time"].to_numpy()

        ##questo serve a forzare di iniziare i samples alla stessa ora per esempio (controllo sul primo indice della y)
        if starting_point is not None:
            check = data[list(starting_point.keys())[0]].values == starting_point[list(starting_point.keys())[0]]
        else:
            check = [True]*len(y_target)
        
        for i in range(past_steps,data.shape[0]-future_steps,skip_step):
            if check[i]:

                if len(self.future_variables)>0:
                    if keep_entire_seq_while_shifting:
                        xx = x_num_future[i-shift:i+future_steps].mean()
                    else:
                        xx = x_num_future[i-shift:i+future_steps-shift].mean()
                else:
                    xx = 0.0
                if np.isfinite(x_num_past[i-past_steps:i].min() + y_target[i:i+future_steps].min() + xx):
                    
                    x_num_past_samples.append(x_num_past[i-past_steps:i])
                    if len(self.future_variables)>0:
                        if keep_entire_seq_while_shifting:
                            x_num_future_samples.append(x_num_future[i-shift:i+future_steps])
                        else:
                            x_num_future_samples.append(x_num_future[i-shift:i+future_steps-shift])
                    if len(self.cat_var)>0:
                        x_cat_past_samples.append(x_cat[i-past_steps:i])
                        if keep_entire_seq_while_shifting:
                            x_cat_future_samples.append(x_cat[i-shift:i+future_steps])
                        else:
                            x_cat_future_samples.append(x_cat[i-shift:i+future_steps-shift])
                    y_samples.append(y_target[i:i+future_steps])
                    t_samples.append(t[i:i+future_steps])


        if len(self.future_variables)>0:
            try:
                x_num_future_samples = np.stack(list(x_num_future_samples))
            except:
                logging.error(f'WARNING!!!!!!!!!!!!x_num_future_samples is empty and it should not')
        targets = np.stack(y_samples)

        times = np.stack(t_samples)
        if len(self.cat_var)>0:
            x_cat_past_samples = np.stack(x_cat_past_samples)
            x_cat_future_samples = np.stack(x_cat_future_samples)
        x_num_past_samples = np.stack(x_num_past_samples)
        
        dd = {'y':targets.astype(np.float32),

              'x_num_past':x_num_past_samples.astype(np.float32)}
        if len(self.cat_var)>0:
            dd['x_cat_past'] = x_cat_past_samples
            dd['x_cat_future'] = x_cat_future_samples
        if len(self.future_variables)>0:
            dd['x_num_future'] = np.asarray(x_num_future_samples).astype(np.float32)
        
        return MyDataset(dd,times,idx_target)
    
          
    
    def split_for_train(self,
                        perc_train:Union[float,None]=0.6,
                        perc_valid:Union[float,None]=0.2,
                        range_train:Union[List[Union[datetime, str]],None]=None,
                        range_validation:Union[List[Union[datetime, str]],None]=None,
                        range_test:Union[List[Union[datetime, str]],None]=None,
                        past_steps:int = 100,
                        future_steps:int=20,
                        shift:int = 0,
                        keep_entire_seq_while_shifting:bool=False,
                        starting_point:Union[None, dict]=None,
                        skip_step:int=1
                        )->tuple[MyDataset, MyDataset, MyDataset]:
        """Split the data and create the datasets.

        Args:
            perc_train (Union[float,None], optional): fraction of the training set. Defaults to 0.6.
            perc_valid (Union[float,None], optional): fraction of the test set. Defaults to 0.2.
            range_train (Union[List[Union[datetime, str]],None], optional): a list of two elements indicating the starting point and end point of the training set (string date style or datetime). Defaults to None.
            range_validation (Union[List[Union[datetime, str]],None], optional):a list of two elements indicating the starting point and end point of the validation set (string date style or datetime). Defaults to None.
            range_test (Union[List[Union[datetime, str]],None], optional): a list of two elements indicating the starting point and end point of the test set (string date style or datetime). Defaults to None.
            past_steps (int, optional): past step to consider for making the prediction. Defaults to 100.
            future_steps (int, optional): future step to predict. Defaults to 20.
            shift (int, optional): see `create_data_loader`. Defaults to 0.
            keep_entire_seq_while_shifting (bool, optional): if the dataset is shifted, you may want the future data be of length future_step+shift (like informer), default false

            starting_point (Union[None, dict], optional):  see `create_data_loader`. Defaults to None.
            skip_step (int, optional):  see `create_data_loader`. Defaults to 1.

        Returns:
            List[DataLoader,DataLoader,DataLoadtrainer]: three dataloader used for training or inference
        """

        
        
        try:
            l = self.dataset.shape[0]
        except:
            logging.error('Empty dataset')
            raise ValueError("Empty dataset")
        

        if range_train is None:
            if perc_train is None or perc_valid is None:
                raise ValueError("Training and validation fractions are required.")
            logging.info(f'Split temporally using perc_train: {perc_train} and perc_valid:{perc_valid}')
        
            train = self.dataset.iloc[0:int(perc_train*l)]
            validation = self.dataset.iloc[int(perc_train*l):int(perc_train*l+perc_valid*l)]
            test = self.dataset.iloc[int(perc_train*l+perc_valid*l):]
        else:
            if range_validation is None or range_test is None:
                raise ValueError("Provide training, validation, and test intervals together.")
            logging.info('Split temporally using the time intervals provided')

            train = self.dataset[self.dataset.time.between(range_train[0],range_train[1])]
            validation =  self.dataset[self.dataset.time.between(range_validation[0],range_validation[1])]
            test =  self.dataset[self.dataset.time.between(range_test[0],range_test[1])]
                                      

        logging.info('######################################################################################################')
        logging.info('######Scaling numerical (standard scaler) and categorical (label encorer) on the training data! ######')
        logging.info('######################################################################################################')
        if self.scalers_fitted:
            pass
        else:
            self.scaler_cat = {}
            self.scaler_num = {}
            for c in self.num_var:
                self.scaler_num[c] =  StandardScaler()
                self.scaler_num[c].fit(train[c].to_numpy().reshape(-1,1))
            for c in self.cat_var:                               
                self.scaler_cat[c] =  LabelEncoder()
                self.scaler_cat[c].fit(train[c].to_numpy().reshape(-1,1))
                                        
        
        self.scalers_fitted = True
        dl_train = self.create_data_loader(train,past_steps,future_steps,shift,keep_entire_seq_while_shifting,starting_point,skip_step)
        dl_validation = self.create_data_loader(validation,past_steps,future_steps,shift,keep_entire_seq_while_shifting,starting_point,skip_step)
        dl_test = self.create_data_loader(test,past_steps,future_steps,shift,keep_entire_seq_while_shifting,starting_point,skip_step)

        return dl_train,dl_validation,dl_test
