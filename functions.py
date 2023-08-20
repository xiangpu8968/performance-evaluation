# Containing all the basic functions needed to evaluate funds
# Treynor Ratio
def treynor(Rp:list,Rb:list,Rf:list) -> float:
    rp, rf = mean(Rp), mean(Rf)
    Rp_rf = [rp-rf for rp in Rp]
    Rb_rf = [rb-rf for rb in Rb]
    alpha, beta = univar_lnr(Rb_rf,Rp_rf)
    return (rp - rf) / beta

# Sharpe Ratio
def sharpe(Rp:list,Rb:list,Rf:list) -> float:
    return mean(sub(Rp,Rf)) / stdv(Rp)

# M-squared Ratio
def m_squared(Rp:list,Rb:list,Rf:list) -> float:
    return (sharpe(Rp,Rb,Rf) - sharpe(Rb,Rb,Rf)) / stdv(Rb)

# Sortino Ratio
def sortino(Rp:list,Rb:list,Rf:list) -> float:
    return (mean(Rp) - mean(Rf)) / semi_stdv(Rp,Rb,Rf)

# Semi-deviation
def semi_stdv(Rp:list,Rb:list,Rf:list) -> float:
    rp = mean(Rp)
    temp = [r - rp for r in Rp]
    for i in range(len(temp)):
        if temp[i] >= 0:
            temp[i]
    sum_square = 0.0
    for r in temp:
        sum_square += r**2
    return (sum_square / len(temp)) ** (1/2)

# Jensen's Alpha
def jensen_alpha(Rp:list,Rb:list,Rf:list) -> float:
    Rp_rf = sub(Rp,Rf)
    Rb_rf = sub(Rb,Rf)
    alpha, beta = univar_lnr(Rb_rf,Rp_rf)
    return alpha

def jensen_beta(Rp:list,Rb:list,Rf:list) -> float:
    Rp_rf = sub(Rp,Rf)
    Rb_rf = sub(Rb,Rf)
    alpha, beta = univar_lnr(Rb_rf,Rp_rf)
    return beta

# Information Ratio
def information_ratio(Rp:list,Rb:list,Rf:list,mode:int=0) -> float:  # mode 0 for diff_return, mode 1 for alpha
    try:
        rp, rb = mean(Rp), mean(Rb)
        alpha, beta = univar_lnr(Rb,Rp)
        sigma = stdv([Rp[i]-Rb[i] for i in range(min(len(Rp),len(Rb)))])
        if mode == 0:
            numerator = rp - rb
        elif mode == 1:
            numerator = alpha
        return numerator / sigma
    except:
        return None

# R_square
def r_square(y_hat:list,y:list,*agrs) -> float:
    rss = sum_square(y_hat,y)
    ess = sum_square(y,[mean(y) for i in range(len(y))])
    return ess / (rss + ess)

# Mean Abosolute Error (MAE)
def mae(y_hat:list,y:list,*args) -> float:
    return mean([abs(a-b) for a,b in zip(y_hat,y)])    

# Mean Squared Error (MSE)
def mse(y_hat:list,y:list,*args) -> float:
    return sum_square(y_hat,y) / len(y_hat)

# Root Mean Squared Error (RMSE)
def rmse(y_hat:list,y:list,*args) -> float:
    return sum_square(y_hat,y) ** (1/2)

# Directional Accuracy
def direction_acc(y_hat:list,y:list,*args) -> float:
    direct_counter = 0
    for a,b in zip(y_hat,y):
        if a * b > 0:
            direct_counter += 1
    return direct_counter / len(y_hat)
# Compound Return
def compound_return(X:list,*args) -> float:
    acc_value = back_test(X)
    return (acc_value[-1]/acc_value[0]) ** (1/len(X)) - 1

# Kendall's Tau and Spearman's Ranking Correlation
# Quantile Loss
# Backtesting
def get_inv_multiple(X:list,*args) -> float:
    acc_value = back_test(X)
    return acc_value[-1]/acc_value[0]

# Basic functions
def mean(X:list) -> float:
    return sum(X) / len(X)

def add(X:list,Y:list) -> list:
    temp = []
    for x,y in zip(X,Y):
        temp.append(x+y)
    return temp

def sub(X:list,Y:list) -> list:
    temp = []
    for x,y in zip(X,Y):
        temp.append(x-y)
    return temp

def divide(X:list,divider:float) -> list:
    return [x/divider for x in X]

def multi(X:list,multiplier:float) -> list:
    return [x*multiplier for x in X]

def sum(X:list) -> float:
    temp = 0.0
    for x in X:
        temp += x
    return temp

def sum_square(x:list,y:list) -> float:
    temp_sum = 0.0
    for a,b in zip(x,y):
        temp_sum += (a - b) ** 2
    return temp_sum

def stdv(X:list,mode:int=0) -> float:  # 0 for population stdv, 1 for sample stdv
    x_mean = mean(X)
    sum_square = 0
    for x in X:
        sum_square += (x - x_mean) ** 2
    num = len(X)
    if mode == 1:
        num -= 1
    return (sum_square / num) ** (1/2)

def cov(X:list,Y:list) -> float:
    temp_sum = 0.0
    x_mean, y_mean = mean(X), mean(Y)
    for x,y in zip(X,Y):
        sum += (x - x_mean) * (y - y_mean)
    return sum / (len(X) - 1)

def corr(X:list,Y:list) -> float:
    return cov(X,Y) / (stdv(X) * stdv(Y))

def  transpose(X:list=[[]]) -> list:
    row = len(X)
    col = len(X[0])
    temp = [[X[j][i] for j in range(row)] for i in range(col)]
    return(temp)

def univar_lnr(X:list,Y:list) -> (float):
    n = len(X)
    sum_x,sum_y = sum(X), sum(Y)
    sum_xy = sum([x * y for x, y in zip(X,Y)])
    sum_x2 = sum([x**2 for x in X])
    beta = ((sum_y * sum_x) / n - sum_xy) / ((sum_x * sum_x) / n - sum_x2)
    alpha = (sum_y - beta * sum_x) / n
    return alpha, beta

def year_to_month(X:list) -> list:
    return [(1+x)**(1/12)-1 for x in X]

def month_to_year(X:list) -> list:
    return [(1+x)**12-1 for x in X]

def get_extreme(X:list,num:int=1,mode:str='max') -> list:
    if mode == 'max':
        temp = sorted(X,reverse=True)
    else:
        temp = sorted(X)
    return temp[:num]

def get_index(X:list,value_l:list) -> list:
    return [X.index(value) for value in value_l]

def get_value(X:list,index_l:list) -> list:
    return [X[i] for i in index_l]

# back test methods
def back_test(X:list) -> list:
    value = 100
    temp = []
    temp.append(100)
    for x in X:
        value = value * (1+x)
        temp.append(value)
    return temp

def simple_strategy(Y_hat:list,Y:list,Rf:list) -> list:
    Y_port = []
    for y_hat,y,rf in zip(Y_hat,Y,Rf):
        if y_hat <= 0:
            Y_port.append(rf)
        else:
            Y_port.append(y)
    return Y_port

def max_drawdown(X:list,*args) -> float:
    acc_value = back_test(X)
    drawdown = 0.0
    for i in range(1,len(acc_value)):
        max_value = max(acc_value[:i])
        if drawdown > (acc_value[i] - max_value)/max_value:
            drawdown = (acc_value[i] - max_value)/max_value
    return -drawdown

# File I/O operations
def read_file(file,read_title=True,read_index=True,list_as_var=False):
    file_sep = {'csv':',','xls':'\t'}
    file_type = file.split('.')[-1]
    data, title, index = [], [], []
    f = open(file,'r')
    if read_title == True:
        title = f.readline()
        title = [s.strip() for s in title.split(file_sep[file_type])]
    temp = f.readlines()
    for row in temp:
        row = [s.strip() for s in row.split(file_sep[file_type])]
        if read_index == True:
            index.append(row[0])
            data.append([float(x) for x in row[1:]])
        else:
            data.append([float(x) for x in row[1:]])
    f.close()
    if list_as_var == True:
        data = transpose(data)
    return data,title,index

def write_file(file:str,data:list,title:bool=None,index:bool=None,list_as_var:bool=False):
    file_sep = {'csv':',','xls':'\t'}
    file_type = file.split('.')[-1]
    f = open(file,'w')
    if title != None:
        f.write(file_sep[file_type].join(title)+'\n')
    if list_as_var == False:
        data = transpose(data)
    for i in range(len(data[0])):
        if index != None:
            f.write(index[i]+file_sep[file_type]+file_sep[file_type].join([str(var[i]) for var in data])+'\n')
        else:
            f.write(file_sep[file_type].join([str(var[i]) for var in data])+'\n')
    f.close()

# OOP modules
class Evaluator:
    def __init__(self,Rp,Rb,Rf:list=[]):
        self.rp, self.rb= Rp, Rb
        if Rf == []:
            self.rf = [0.0] * len(self.rb)
        else:
            self.rf = Rf
        self.measures = {'Treynor':treynor,
                         'Sharpe':sharpe,
                         'M-squared':m_squared,
                         'Sortino':sortino,
                         'J-Alpha':jensen_alpha,
                         'Beta':jensen_beta,
                         'Information Ratio':information_ratio,
                         'R_square':r_square,
                         'MAE':mae,
                         'MSE':mse,
                         'RMSE':rmse,
                         'Directional Accuracy':direction_acc,
                         'Max-drawdown':max_drawdown,
                         'Inv Multiple':get_inv_multiple}
        self.descriptive = {'Mean Return':mean,'Volatility':stdv,'Max':max,'Min':min}
        self.evaluate()

    def evaluate(self):
        self.performance = {}
        for stat in self.descriptive:
            self.performance[stat] = self.descriptive[stat](self.rp)
        for measure in self.measures:
            self.performance[measure] = self.measures[measure](self.rp,self.rb,self.rf)

    def display(self):
        for measure in self.performance.keys():
            try:
                print(f'{measure:<20s}: {self.performance[measure]:>7.4f}')
            except:
                print(f'{measure:<20s}: {self.performance[measure]}')

    def back_test(self):
        self.port_acc_value = back_test(self.rp)
        self.bench_acc_value = back_test(self.rb)

    def write_perform(self,file):
        title = ['measure','value']
        index = [key for key in self.performance.keys()]
        data = [self.performance[key] for key in index]
        write_file(file=file,title=title,index=index,data=[data],list_as_var=True)

if __name__ == '__main__':
    # Test file I/O operations
    file = 'test.csv'
    perform_file = 'perform.csv'
    data,title,index = read_file(file,read_title=True,read_index=True,list_as_var=True)
    Rp = data[0]
    Rb = data[1]
    Rf = data[2]
    
    # Test basic functions and measurements
    eval = Evaluator(Rp,Rb,Rf)
    eval.display()
    eval.write_perform(perform_file)
