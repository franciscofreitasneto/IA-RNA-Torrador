import torch
from torch import nn
import pandas as pd
import numpy as np
import decimal

def float_range(start, stop, step):
    # gerar um range de numeros float
    while start < stop:
        yield float(start)
        start += decimal.Decimal(step)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, out_size):
        super(MLP, self).__init__()

        self.features = nn.Sequential(nn.Linear(input_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size2),
                                      nn.ReLU(),
                                      #nn.Linear(hidden_size2, hidden_size3),
                                      #nn.ReLU(),
                                      )
        self.out      = nn.Linear(hidden_size2, out_size)

    def forward(self, X):
        feature = self.features(X)
        output  = self.out(feature)
        return output

class GettingMlpDataset:
    def __init__(self, cat_grao, temp_amb, umid_amb, temp_car, massa, tempo_o, csv_path):
        self.cat_grao = cat_grao
        self.temp_amb = temp_amb
        self.umid_amb = umid_amb
        self.temp_car = temp_car
        self.massa    = massa
        self.tempo_o  = tempo_o
        self.csv_path  = csv_path

    def run(self):
        tempo_i = 0
        time_x = list(range(tempo_i, int(self.tempo_o), 2))
        # cafe_do_mario
        # vo_mira_conilon
        # vo_mira_arabica
        # cafe_saulo_gourmet
        # cafe_saulo_especial
        categorias = [0, 0, 0, 0, 0]
        categorias[self.cat_grao] = 1

        massa = [0, 0] # acima_2000, abaixo_2000
        if self.massa >= 2000: massa[0] = 1
        else: massa[1] = 1

        csv = pd.read_csv(self.csv_path, index_col=0)

        colunas = list(csv.columns)
        df_torra = pd.DataFrame(columns=colunas)

        df_torra['timex'] = time_x
        df_torra['temp2'] = 0
        df_torra['cafe_do_mario'] = categorias[0]
        df_torra['vo_mira_conilon'] = categorias[1]
        df_torra['vo_mira_arabica'] = categorias[2]
        df_torra['cafe_saulo_gourmet'] = categorias[3]
        df_torra['cafe_saulo_especial'] = categorias[4]
        df_torra['temp_amb'] = self.temp_amb
        df_torra['umid_amb'] = self.umid_amb
        df_torra['charge_bt_modificado'] = self.temp_car
        df_torra['Acima de 2000'] = massa[0]
        df_torra['Abaixo de 2000'] = massa[1]

        self.df_torra = df_torra
        return df_torra

    def normalized(self):
        df = self.run()
        csv = pd.read_csv(self.csv_path, index_col=0)

        df_normalized = df.copy()
        colunas = list(csv.columns)

        for coluna in colunas:
            df_normalized[coluna] = (df[coluna] - csv[coluna].loc['MIN']) / (csv[coluna].loc['MAX'] - csv[coluna].loc['MIN'])

        self.df_torra_normalized = df_normalized
        return df_normalized

class RunningMlp:
    def __init__(self, nn_path, df):
        self.net = torch.load(nn_path, map_location=torch.device('cpu'))
        self.df = df

    def run(self):
        self.net.eval()

        df_copy = self.df.copy()
        del df_copy['temp2']
        np_array = df_copy.to_numpy().astype(np.float32)

        pred = []
        for sample in np_array:
            tensor_sample = torch.from_numpy(sample)
            pred.append(self.net(tensor_sample).cpu().data.numpy()[0])
        df_result = pd.DataFrame()
        df_result['temp2'] = pred
        return df_result

class DesnormalizingMlpDataSet:
    def __init__(self, df, csv_path):
        self.csv_path = csv_path
        self.df = df

    def run(self):
        csv_file = pd.read_csv(self.csv_path, index_col=0)
        self.df['temp2'] = self.df['temp2'] * (csv_file['temp2'].loc['MAX'] - csv_file['temp2'].loc['MIN']) + csv_file['temp2'].loc['MIN']
        #self.df['temp2'] = [float(f'{n: .8f}') for n in self.df['temp2'].values.tolist()]

        return self.df

class CreatingAlogFile:
    original_alog_path = '/home/pi/Documents/artisan/inovagrao/base_curve/new_profile.alog'
    new_alog_path = '/home/pi/Documents/curvas_de_torra/nova_curva.alog'

    def __init__(self, df, time_x):
        self.df = df
        self.time_x = time_x

    def run(self):
        alog_fp = open(self.original_alog_path, 'r')
        alog_contents = eval(alog_fp.read())
        alog_fp.close()

        tp_bt = self.df['temp2'].min()
        temp_2_ls = self.df['temp2'].values.tolist()
        index_tp = temp_2_ls.index(tp_bt)

        dry_end              = round(self.time_x[-1] * 0.4 / 2)
        first_cracker_start  = round(self.time_x[-1] * 0.7 / 2)
        first_cracker_end    = round((self.time_x[-1] * 5) / 6 / 2)
        second_cracker_start = round(self.time_x[-1] * 0.875 / 2)
        second_cracker_end   = round(self.time_x[-1] * 0.958 / 2)
        drop                 = round(self.time_x[-1] / 2)

        alog_contents['timeindex'][1] = dry_end
        alog_contents['timeindex'][2] = first_cracker_start
        alog_contents['timeindex'][3] = first_cracker_end
        alog_contents['timeindex'][4] = second_cracker_start
        alog_contents['timeindex'][5] = second_cracker_end
        alog_contents['timeindex'][6] = drop
        alog_contents['timeindex'][7] = 0

        alog_contents['timex'] = self.time_x
        alog_contents['temp2'] = list(self.df['temp2'])
        alog_contents['temp1'] = list(self.df['temp2'])
        alog_contents['extratimex'] = [self.time_x]
        alog_contents['extratemp1'] = [[-1 for i in self.time_x]]
        alog_contents['extratemp2'] = [[-1 for i in self.time_x]]

        alog_contents['computed']['CHARGE_BT'] = self.df['temp2'][0]
        alog_contents['computed']['CHARGE_ET'] = self.time_x[0]

        alog_contents['computed']['TP_time'] = self.time_x[index_tp]
        alog_contents['computed']['TP_BT'] = tp_bt
        alog_contents['computed']['totaltime'] = self.time_x[-1] + 2 # soma 2 devido ao sample time

        # eixo X da posição das anotações
        alog_contents['anno_positions'][1][1] = alog_contents['anno_positions'][1][3] = self.time_x[index_tp]
        alog_contents['anno_positions'][2][1] = alog_contents['anno_positions'][2][3] = dry_end * 2
        alog_contents['anno_positions'][3][1] = alog_contents['anno_positions'][3][3] = first_cracker_start * 2
        alog_contents['anno_positions'][4][1] = alog_contents['anno_positions'][4][3] = first_cracker_end * 2
        alog_contents['anno_positions'][5][1] = alog_contents['anno_positions'][5][3] = second_cracker_start * 2
        alog_contents['anno_positions'][6][1] = alog_contents['anno_positions'][6][3] = second_cracker_end * 2
        alog_contents['anno_positions'][7][1] = alog_contents['anno_positions'][7][3] = drop * 2

        # eixo Y da posição das anotações
        offset_temp = 18
        alog_contents['anno_positions'][0][2] = temp_2_ls[0] + offset_temp
        alog_contents['anno_positions'][0][4] = temp_2_ls[0] - offset_temp
        alog_contents['anno_positions'][1][2] = temp_2_ls[index_tp] + offset_temp
        alog_contents['anno_positions'][1][4] = temp_2_ls[index_tp] - offset_temp
        alog_contents['anno_positions'][2][2] = temp_2_ls[dry_end] + offset_temp
        alog_contents['anno_positions'][2][4] = temp_2_ls[dry_end] - offset_temp
        alog_contents['anno_positions'][3][2] = temp_2_ls[first_cracker_start] + offset_temp
        alog_contents['anno_positions'][3][4] = temp_2_ls[first_cracker_start] - offset_temp
        alog_contents['anno_positions'][4][2] = temp_2_ls[first_cracker_end] + offset_temp
        alog_contents['anno_positions'][4][4] = temp_2_ls[first_cracker_end] - offset_temp
        alog_contents['anno_positions'][5][2] = temp_2_ls[second_cracker_start] + offset_temp
        alog_contents['anno_positions'][5][4] = temp_2_ls[second_cracker_start] - offset_temp
        alog_contents['anno_positions'][6][2] = temp_2_ls[second_cracker_end] + offset_temp
        alog_contents['anno_positions'][6][4] = temp_2_ls[second_cracker_end] - offset_temp
        alog_contents['anno_positions'][7][2] = temp_2_ls[drop] + offset_temp
        alog_contents['anno_positions'][7][4] = temp_2_ls[drop] - offset_temp

        alog_contents = str(alog_contents)

        with open(self.new_alog_path, 'w') as new_alog_fp:
            new_alog_fp.write(alog_contents)


class GeneratingRoastProfile:
    categoria_grao = {'café do Mario': 0, 'Vó Mira Conilon': 1, 'Vó Mira Arabica': 2, 'Café do Saulo Gourmet': 3, 'café do Saulo Especial': 4}
    categoria_massa = {'Acima de 2000': 2001, 'Abaixo de 2000': 100}
    # torch nao consegue carregar o caminho apartir ~/
    nn_path = '/home/pi/Documents/artisan/inovagrao/neural_net/net.pt'
    min_max_path = '~/Documents/artisan/inovagrao/datasets/min_max.csv'

    def __init__(self, *args):
        self.cat_grao, self.massa, self.temp_amb, self.umid_amb, self.temp_car = args
        self.cat_grao = self.categoria_grao[self.cat_grao]
        self.massa = self.categoria_massa[self.massa]
        self.temp_amb = float(self.temp_amb)
        self.umid_amb = float(self.umid_amb) / 100
        self.temp_car = float(self.temp_car)

    def run(self):
        gettingMlpDataset = GettingMlpDataset(self.cat_grao, self.temp_amb, self.umid_amb,
                                              self.temp_car, self.massa, 550, self.min_max_path)
        mlp_dataset = gettingMlpDataset.normalized()

        running_mlp = RunningMlp(self.nn_path, mlp_dataset)
        df_predicted = running_mlp.run()

        desnormalizing_mlp_dataSet = DesnormalizingMlpDataSet(df_predicted, self.min_max_path)
        dataset_desnorm = desnormalizing_mlp_dataSet.run()

        # conversar com o daniel sobre esse for:
        for i in list(dataset_desnorm.index):
            if dataset_desnorm.loc[i]['temp2'] > self.temp_car: # temperatura de carga dos graos
                dataset_desnorm.loc[i, 'temp2'] = self.temp_car
            else: break

        #time_x = list(range(0, 550, 2)) # 0 e 550 de acordo com os valores de entrada do teste_get_dataset
        time_x = [i for i in float_range(0, 550, '2')]
        creating_alog_file = CreatingAlogFile(dataset_desnorm, time_x)
        creating_alog_file.run()
