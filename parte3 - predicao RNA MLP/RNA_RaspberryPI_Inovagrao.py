#!/usr/bin/env python
# coding: utf-8

# Importando as bibliotecas para RNA
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Bibliotecas para acesso aos arquivos
import pandas as pd
import numpy as np

# Biblioteca para plotar o gráfico
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Biblioteca para GUI
import tkinter as tk
from tkinter import ttk as tkk
matplotlib.use('qtagg')

# Threads
#import threading as th

# Obtendo o arquivo com os Metaparâmetros da RNA
arq = open('../datasets/mlp_param.txt')
args = arq.read()
args = eval(args)
arq.close()

# Verificando a possibilidade de executar na GPU (Cudas)
if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

# Carregando os valores máximo e mínimo para normalização MIN-MAX
max_min = pd.read_csv('../datasets/min_max.csv', index_col=0)

# Funções de Normalização
def minmax_norm(df_input, df_input_min, df_input_max):
    return(df_input - df_input_min) / (df_input_max - df_input_min)

def minmax_unorm(df_input, df_input_min, df_input_max):
    return (df_input * (df_input_max - df_input_min) + df_input_min)

# Implementando a MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, out_size):
        super(MLP, self).__init__()

        self.features = nn.Sequential(
                                nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size2),
                                nn.ReLU(),
                               # nn.Linear(hidden_size2, hidden_size3),
                               # nn.ReLU(),
                        )
        self.out     = nn.Linear(hidden_size2, out_size)
    def forward(self, X):
        feature = self.features(X)
        output  = self.out(feature)
        return output

input_size  = 11 #Qtd de atributos de Entrada na RNA - MLP
hidden_size = args['size_shape']
hidden_size2 = args['size_shape2']
hidden_size3 = args['size_shape3']
out_size    = 1 # nº de variáveis preditas

net = MLP(input_size, hidden_size, hidden_size2, out_size).to(args['device']) # Se tiver GPU

# Carregando a RNA - Parâmetros da Arquitetura (Pesos do Treinamento)
net=torch.load('../datasets/net.pt', map_location=torch.device('cpu'))
print('Estrutura da RNA MLP:')
print(net.eval())

# Acessando a Base de Dados onde os valores de treinamento forma obtidos
dataset = pd.read_csv('../datasets/dataset_cafe_estatistica.csv',index_col=0)
# Obtendo a lista de grãos treinados
lista_graos = list(dataset['grao'].unique())

# Preparando a Interface com o usuário
cat_grao=''
def seleciona_grao():
    indice_item = lsbox_graos.curselection()[0]
    global cat_grao
    cat_grao = lsbox_graos.get(indice_item)
    # Selecionando o dataset com o filtro pela categoria do grão
    cont = dataset['grao'] == cat_grao
    # Alterando o Label da Temp.Ambiente
    lb_temp_amb['text'] = 'Temp. do Ambiente: [%.2f .. %.2f]' %(dataset[cont]['temp_amb'].min(),
            dataset[cont]['temp_amb'].max())
    # Alterando o Text da Temp.Ambiente
    edt_temp_amb.delete(1.0,'end')
    edt_temp_amb.insert(1.0,dataset[cont]['temp_amb'].mode()[0])
    # Alterando o Label da Umid.Ambiente
    lb_umi_amb['text']  = 'Umid. do Ambiente: [%.2f .. %.2f]' %(dataset[cont]['umid_amb'].min(),
            dataset[cont]['umid_amb'].max())
    # Alterando o Text da Umid.Ambiente
    edt_umi_amb.delete(1.0,'end')
    edt_umi_amb.insert(1.0,dataset[cont]['umid_amb'].mode()[0])
    # Alterando o Label da Temp.Carga
    lb_temp_carga['text'] = 'Temp. de Carga: [%.2f .. %.2f]' %(dataset[cont]['charge_bt_modificado'].min(),
            dataset[cont]['charge_bt_modificado'].max())
    # Alterando o Text da Temp.Carga
    edt_temp_carga.delete(1.0,'end')
    edt_temp_carga.insert(1.0,dataset[cont]['charge_bt_modificado'].mode()[0])
    # Alterando o Label da Massa de Carga
    lb_mas_carga['text'] = 'Massa de Carga: [%.2f .. %.2f]' %(dataset[cont]['massa_grao'].min(),
            dataset[cont]['massa_grao'].max())
    # Alterando o Text da Massa de Grãos
    edt_mas_carga.delete(1.0,'end')
    edt_mas_carga.insert(1.0,dataset[cont]['massa_grao'].mode()[0])
    # Alterando o Label do Tempo de Torra
    lb_tp_torra['text'] = 'Tempo de Torra: [%.2f .. %.2f]' %(dataset[cont]['timex'].min(),
            dataset[cont]['timex'].max())
    # Alterando o Text do Tempo de Torra
    edt_tp_torra.delete(1.0,'end')
    edt_tp_torra.insert(1.0,dataset[cont]['timex'].max())

    #print(cat_grao)

# Classe para a criação do DataSet
class Dataset_amostras(Dataset):
    def __init__(self, df):
        self.dados = df.to_numpy()

    def __getitem__(self, idx):
        sample = self.dados[idx][[0,1,2,3,4,5,6,7,9,10,11]]
        label  = self.dados[idx][8:9]

        # Converter para Tensor
        sample = torch.from_numpy(sample.astype(np.float32))
        label  = torch.from_numpy(label.astype(np.float32))
        return sample, label

    def __len__(self):
        return len(self.dados)

def gerar_curva_torra():
    global df_torra
    # Reecebendo os valores das features para gerar a saida
    temp_amb = float(edt_temp_amb.get(1.0,'end-1c'))
    umid_amb = float(edt_umi_amb.get(1.0,'end-1c'))
    temp_car = float(edt_temp_carga.get(1.0,'end-1c'))
    massa    = float(edt_mas_carga.get(1.0,'end-1c'))
    #tempo_i  = float(input('Tempo de carga: '))
    #tempo_i  = dataset[cont]['timex'].min()
    tempo_i = 0
    tempo_o  = float(edt_tp_torra.get(1.0,'end-1c'))

    # Ajustando a categoria de grãos
    cafe_do_mario       = 0
    vo_mira_conilon     = 0
    vo_mira_arabica     = 0
    cafe_saulo_gourmet  = 0
    cafe_saulo_especial = 0
    if cat_grao == 1:
        cafe_do_mario = 1
    elif cat_grao == 2:
        vo_mira_conilon = 1
    elif cat_grao == 3:
        vo_mira_arabica = 1
    elif cat_grao == 4:
        cafe_saulo_gourmet = 1
    elif cat_grao == 5:
        cafe_saulo_especial = 1

    # Ajustando a masa de grãos
    acima_2000  = 0
    abaixo_2000 = 0
    if massa >= 2000:
        acima_2000  = 1
        abaixo_2000 = 0
    else:
        acima_2000  = 0
        abaixo_2000 = 1

    # Ajustando a masa de grãos
    acima_2000  = 0
    abaixo_2000 = 0
    if massa >= 2000:
        acima_2000  = 1
        abaixo_2000 = 0
    else:
        acima_2000  = 0
        abaixo_2000 = 1

    # Criando os pontos de tmatplotlib.use('TkAgg')empo de torra
    timex = list(range(int(tempo_i), int(tempo_o), 2))

    # Montando o Dataframe
    colunas = list(max_min.columns)
    df_torra = pd.DataFrame(columns=colunas)
    df_torra['timex'] = timex
    df_torra['temp2'] = 0
    df_torra['cafe_do_mario']   = cafe_do_mario
    df_torra['vo_mira_conilon'] = vo_mira_conilon
    df_torra['vo_mira_conilon'] = vo_mira_conilon
    df_torra['vo_mira_arabica'] = vo_mira_arabica
    df_torra['cafe_saulo_gourmet'] = cafe_saulo_gourmet
    df_torra['cafe_saulo_especial'] = cafe_saulo_especial
    df_torra['temp_amb'] = temp_amb
    df_torra['umid_amb'] = umid_amb
    df_torra['charge_bt_modificado'] = temp_car
    df_torra['Acima de 2000'] = acima_2000
    df_torra['Abaixo de 2000'] = abaixo_2000

    # Normalizando o Dataframe
    df_torra_norm = df_torra.copy()
    for coluna in colunas:
        df_torra_norm[coluna] = minmax_norm(df_torra[coluna], max_min[coluna].loc['MIN'], max_min[coluna].loc['MAX'])

    # Criando o nosso DataSet
    test_set = Dataset_amostras(df_torra_norm)
    dado, rotulo = test_set[0]
    print(rotulo)
    print(dado)

    # Rodando os valores na RNA e gerando a predição
    pred = []
    for i in range(df_torra_norm.shape[0]):
        dado = test_set[i][0]
        pred.append(net(dado).cpu().data.numpy()[0])

    # Incluindo os valores da predição no DataFrame normalizado
    # (Não precisa, mas já que existe...)
    df_torra_norm['temp2'] = pred
    df_torra['temp2'] = pred
    #df_torra_norm.head()

    # "Desnormalizando" os valores
    df_torra['temp2'] = minmax_unorm(df_torra['temp2'], max_min['temp2'].loc['MIN'], max_min['temp2'].loc['MAX'])
    #df_torra

    # Removendo os valores inicias até a temperatura de entrada dos grãos selecionada
    for i in list(df_torra.index):
        if df_torra.loc[i, 'temp2'] > temp_car:
            df_torra.loc[i, 'temp2'] = temp_car
        else:
            break
    df_torra.loc[0, 'temp2'] = temp_car
    #df_torra

    # Plotando o Gráfico de Torra
    #g1.start()
    plotar_grafico_torra()

    # Testando o Combobox
    cbox['values'] = list(df_torra['timex'])
    cbox['state'] = 'readonly'

def preenche_temp_edt(event):
    global df_indice
    edt_temp_torra.delete(1.0,'end')
    pt_tempo = str_var.get()
    df_indice = df_torra.index[df_torra['timex']==int(pt_tempo)].to_list()
    temperatura = df_torra.iloc[df_indice]['temp2'].to_list()[0]
    edt_temp_torra.insert(1.0,temperatura)

def salvar_pontos_torra():
    # Recebendo os valores dos campos edit
    temperatura = float(edt_temp_torra.get(1.0,'end-1c'))
    suavidade   = int(edt_suavidade.get(1.0,'end-1c'))
    difer = temperatura - df_torra.at[df_indice[0], 'temp2']
    inter = difer / suavidade
    alter = []
    for i in range(suavidade):
        alter.append(difer - i * inter)
        #print(difer - i * inter)

    # Alterando os valores da Curva de Torra
    pos = df_indice[0]
    #df_torra.at[pos, 'temp2']=temperatura # Ponto Central

    #Alterando os pontos posteriores ao ponto central
    alter = []
    for i in range(suavidade):
        alter.append(difer - i * inter)
    for i in range(pos,pos+suavidade): # Pontos posteriores
        if i > 0 and i < df_torra.shape[0]-1:
            df_torra.at[i, 'temp2'] = df_torra.at[i, 'temp2'] + alter[i-pos]
    #Alterando os pontos anteriores ao ponto central
    for i in range(pos-suavidade, pos):
        if i > 0 and i < df_torra.shape[0]-1:
            pt_relat = i - (pos-suavidade) + 1
            #print(suavidade-pt_relat)
            df_torra.at[i, 'temp2'] = df_torra.at[i, 'temp2'] + alter[suavidade-pt_relat]

    # Plotando o Gráfico de Torra
    #g2.start()
    plotar_grafico_torra()

def plotar_grafico_torra():
    # Plotando o Gráfico de Torra
    plt.rc('figure', figsize=(7,4))
    plt.plot(df_torra['timex'], df_torra['temp2'])
    plt.scatter(df_torra['timex'], df_torra['temp2'])
    plt.title('Gráfico de Torra')
    plt.xlabel('Tempo em Segundos')
    plt.ylabel('Temperatura em ºC')
    plt.grid()
    plt.show()

def salvar_arquivo_torra():
    print()

# Criando as Threads
#g1 = th.Thread(target=plotar_grafico_torra)
#g2 = th.Thread(target=plotar_grafico_torra)

# Criando a GUI
janela = tk.Tk()
janela.title('RNA - Torra Grãos')

#Label de apresentação da ListBox
lbl1 = tk.Label(janela, text='Escolha a categoria do grão:')
lbl1.grid(column=0, row=0, padx=10, pady=5)

# ListBox de escolha dos grãos
lsbox_graos = tk.Listbox(janela)
for item in lista_graos: # Inserindo os ítens na ListaBox
    lsbox_graos.insert(tk.END, item)
lsbox_graos.grid(column=0, row=1, padx=5, pady=5)

# Botão para escolha do grão
bt1 = tk.Button(janela, text='Grão', command=seleciona_grao)
bt1.grid(column=1, row=1, padx=5, pady=5)

# Selecionando o dataset com o filtro pela categoria do grão
cont = dataset['grao'] == cat_grao

# Parâmetro de Temperatura do Ambiente
lb_temp_amb = tk.Label(janela, text='Temp. do Ambiente: [%.2f .. %.2f]'
        %(dataset[cont]['temp_amb'].min(), dataset[cont]['temp_amb'].max()))
lb_temp_amb.grid(column=0, row=2, sticky='w')
edt_temp_amb = tk.Text(janela, height=1, width=6)
edt_temp_amb.grid(column=1, row=2, stick='w')

# Parâmetro de Umidade do Ambiente
lb_umi_amb = tk.Label(janela, text='Umid. do Ambiente: [%.2f .. %.2f]'
        %(dataset[cont]['umid_amb'].min(), dataset[cont]['umid_amb'].max()))
lb_umi_amb.grid(column=0, row=3, sticky='w')
edt_umi_amb = tk.Text(janela, height=1, width=6)
edt_umi_amb.grid(column=1, row=3, stick='w')

# Parâmetros de Temperatura de Carga
lb_temp_carga = tk.Label(janela, text='Temp. de Carga: [%.2f .. %.2f]'
        %(dataset[cont]['charge_bt_modificado'].min(), dataset[cont]['charge_bt_modificado'].max()))
lb_temp_carga.grid(column=0, row=4, sticky='w')
edt_temp_carga = tk.Text(janela, height=1, width=6)
edt_temp_carga.grid(column=1, row=4, stick='w')

# Parâmetro da Massa de Carga
lb_mas_carga = tk.Label(janela, text='Massa de Carga: [%.2f .. %.2f]'
        %(dataset[cont]['massa_grao'].min(), dataset[cont]['massa_grao'].max()))
lb_mas_carga.grid(column=0, row=5, sticky='w')
edt_mas_carga = tk.Text(janela, height=1, width=6)
edt_mas_carga.grid(column=1, row=5, stick='w')

# Parâmetro do Tempo de Torra
lb_tp_torra = tk.Label(janela, text='Tempo de Torra: [%.2f .. %.2f]'
        %(dataset[cont]['timex'].min(), dataset[cont]['timex'].max()))
lb_tp_torra.grid(column=0, row=6, sticky='w')
edt_tp_torra = tk.Text(janela, height=1, width=6)
edt_tp_torra.grid(column=1, row=6, stick='w')

# Botão para gerar a Curva de Torra
bt2 = tk.Button(janela, text='Curva de Torra', command=gerar_curva_torra)
bt2.grid(column=0, row=7, padx=5, pady=5)

# Label de apresentação do ComboBox
lbl1 = tk.Label(janela, text='Pontos da Curva\n(Segundos): ')
lbl1.grid(column=0, row=8, padx=5, pady=5)
lbl2 = tk.Label(janela, text='Temperatura:')
lbl2.grid(column=1, row=8, padx=5, pady=5, stick='w')

# ComboBox com os pontos da Curva de Torra
str_var = tk.StringVar()
cbox = tkk.Combobox(janela, textvariable=str_var)
cbox.bind('<<ComboboxSelected>>', preenche_temp_edt)
cbox.grid(column=0, row=9, padx=5, pady=5)

# Campo de temperatura no ponto selecionado no ComboBox
edt_temp_torra = tk.Text(janela, height=1, width=6)
edt_temp_torra.grid(column=1, row=9, stick='w')

# Label do campo de suavização
lbl3 = tk.Label(janela, text='Suavização:')
lbl3.grid(column=1, row=10, padx=5, pady=5, stick='w')

# Campo para suavizar a curva alterada
edt_suavidade = tk.Text(janela, height=1, width=6)
edt_suavidade.grid(column=1, row=11, stick='w')
edt_suavidade.insert(0.0, '50')

# Botão para alterar os pontos editados da Curva de Torra
bt3 = tk.Button(janela, text='Salvar pontos', command=salvar_pontos_torra)
bt3.grid(column=0, row=12, padx=5, pady=5)

# Label para o nome do arquivo
lbl4 = tk.Label(janela, text='Nome do arquivo:')
lbl4.grid(column=0, row=13, stick='w')

# Campo para o nome do arquivo de torra
edt_nome_arq_torra = tk.Text(janela, height=1, width=30)
edt_nome_arq_torra.grid(column=0,row=14,padx=5,pady=5,stick='w')

# Botão para salvar a curva de torra em arquivo
bt4 = tk.Button(janela, text='Salvar arquivo', command=salvar_arquivo_torra)
bt4.grid(column=1, row=14, padx=5, pady=5)

# Gráfico de Torra
#fig_torra = plt.Figure(figsize=(8,4), dpi=60)
#graf_torra = fig_torra.add_subplot(111)
#canva_torra = FigureCanvasTkAgg(fig_torra, janela)
#canva_torra.get_tk_widget().grid(column=0, row=8)

janela.mainloop()

# Lendo o arquivo exemplo RNA de torra
arq = '../datasets/saida_RNA.alog'
arq_torra = open(arq, 'r+')
torra = eval(arq_torra.read())


# In[20]:


#print(torra)


# In[21]:


torra['timex'] = list(df_torra['timex'].astype(float))
torra['temp2'] = list(df_torra['temp2'])
torra['temp1'] = list(df_torra['temp2'])
torra = str(torra)
#arq_torra.write(repr(torra))
#torra


# In[22]:


torra1 = repr(torra)
torra1 = eval(torra1)
print(len(torra1))


# In[23]:


print(torra1)


# In[ ]:
