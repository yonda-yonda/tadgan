import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy import stats
from typing import Optional

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class Encoder(torch.nn.Module):
    def __init__(self, window_size, latent_dim, hidden_size=100):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size=window_size, hidden_size=hidden_size,
                                  num_layers=1, bidirectional=True, batch_first=True)
        self.dense = torch.nn.Linear(
            in_features=hidden_size*2, out_features=latent_dim)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        z = self.dense(x)
        return (z)


class Decoder(torch.nn.Module):
    def __init__(self, window_size, latent_dim, hidden_size=64):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=latent_dim, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.dense = torch.nn.Linear(
            in_features=hidden_size*2, out_features=window_size)

    def forward(self, z):
        z, (hn, cn) = self.lstm(z)
        x = self.dense(z)
        return (x)


class Cx(torch.nn.Module):
    def __init__(self, window_size, state_dim=1, cnn_blocks=4):
        super().__init__()
        layers = [['Conv1d_0', torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=state_dim,
                            out_channels=64, kernel_size=5, padding=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.25)
        )]]

        if cnn_blocks > 1:
            for i in range(1, cnn_blocks):
                layers.append(['Conv1d_{}'.format(i), torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels=64,
                                    out_channels=64, kernel_size=5, padding=2),
                    torch.nn.LeakyReLU(negative_slope=0.2),
                    torch.nn.Dropout(0.25)
                )])
        layers.append(['Flatten', torch.nn.Flatten()])
        layers.append(['Dense', torch.nn.Linear(
            in_features=64*window_size, out_features=1)])
        self.layers = torch.nn.ModuleDict(layers)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x


class Cz(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        layers = [['dense_0', torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_dim, out_features=100),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2)
        )], ['dense_1', torch.nn.Sequential(
            torch.nn.Linear(in_features=100, out_features=100),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2)
        )], ['dense_2', torch.nn.Sequential(
            torch.nn.Linear(in_features=100, out_features=1)
        )]]
        self.layers = torch.nn.ModuleDict(layers)

    def forward(self, z):
        for layer in self.layers.values():
            z = layer(z)
        return z


class ProcessedDataset(Dataset, metaclass=ABCMeta):
    """
    Implement __getitem__ to return (start_index, np.array(state_size, window_size))
    """

    @property
    @abstractmethod
    def window_size(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def state_size(self):
        raise NotImplementedError()


def _euclideandistance(a, b):
    return np.sqrt(np.sum((a - b)**2))


def _minVal(values):
    return sorted(values, key=lambda x: x['cost'])[0]


def _dtw(ts1, ts2):
    ret = [[None for _ in range(len(ts2))] for _ in range(len(ts1))]

    np.zeros((len(ts1), len(ts2)))

    ret[0][0] = {'cost': _euclideandistance(ts1[0], ts2[0]), 'path': [-1, -1]}

    for i in range(1, len(ts1)):
        ret[i][0] = {'cost': ret[i - 1][0]['cost'] +
                     _euclideandistance(ts1[i], ts2[0]), 'path': [i - 1, 0]}

    for j in range(1, len(ts2)):
        ret[0][j] = {'cost': ret[0][j - 1]['cost'] +
                     _euclideandistance(ts1[0], ts2[j]), 'path': [0, j - 1]}

    for i in range(1, len(ts1)):
        for j in range(1, len(ts2)):
            mv = _minVal([
                {
                    'cost': ret[i - 1][j]['cost'],
                    'path': [i - 1, j]
                },
                {
                    'cost': ret[i][j - 1]['cost'],
                    'path': [i, j - 1]
                },
                {
                    'cost': ret[i - 1][j - 1]['cost'],
                    'path': [i - 1, j - 1]
                }]
            )
            ret[i][j] = {
                'cost': mv['cost'] + _euclideandistance(ts1[i], ts2[j]),
                'path': mv['path']
            }

    return ret[-1][-1]


class TadGanError(Exception):
    def __init__(self, message):
        super().__init__(message)


class TadGAN:
    def __init__(self, dataset: Optional[ProcessedDataset]=None,
                 model_path: Optional[str]=None,
                 batch_size=64,
                 lr=0.0005,
                 num_critics=5,
                 latent_dim=20,
                 loss_rate_critics=[1, 1, 10],
                 loss_rate_generator=[1, 10],
                 encoder_hidden_size=100,
                 decoder_hidden_size=64,
                 cx_cnn_blocks=4):
        self.dataset = dataset
        model = torch.load(model_path) if model_path is not None else None
        if self.dataset is None and model is None:
            raise TadGanError('invalid: need either dataset or model_path.')

        window_size = model['window_size'] if model else dataset.window_size
        state_size = model['state_size'] if model else dataset.state_size
        step_size = model['step_size'] if model else int(dataset[1][0] - dataset[0][0])

        if dataset is not None and window_size != dataset.window_size:
            raise TadGanError('invalid: window size of dataset')
        if dataset is not None and state_size != dataset.state_size:
            raise TadGanError('invalid: state size of dataset')
        if step_size < 1:
            raise TadGanError('invalid: step size of dataset')

        self.window_size = window_size
        self.state_size = state_size
        self.step_size = step_size

        self.batch_size = model['batch_size'] if model else batch_size
        self.num_epoch = model['num_epoch'] if model else 0
        self.lr = model['lr'] if model else lr
        self.num_critics = model['num_critics'] if model else num_critics
        self.latent_dim = model['latent_dim'] if model else latent_dim
        self.loss_rate_critics = model['loss_rate_critics'] if model else loss_rate_critics
        self.loss_rate_generator = model['loss_rate_generator'] if model else loss_rate_generator

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.encoder = Encoder(self.window_size, latent_dim, hidden_size=encoder_hidden_size)
        if model:
            self.encoder.load_state_dict(model['encoder'])
        self.encoder = self.encoder.to(self.device)
        self.decoder = Decoder(self.window_size, latent_dim, hidden_size=decoder_hidden_size)
        if model:
            self.decoder.load_state_dict(model['decoder'])
        self.decoder = self.decoder.to(self.device)
        self.cx = Cx(self.window_size, state_dim=self.state_size, cnn_blocks=cx_cnn_blocks)
        if model:
            self.cx.load_state_dict(model['cx'])
        self.cx = self.cx.to(self.device)
        self.cz = Cz(latent_dim)
        if model:
            self.cz.load_state_dict(model['cz'])
        self.cz = self.cz.to(self.device)
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters(), lr=lr)
        if model:
            self.optimizer_encoder.load_state_dict(model['optimizer_encoder'])
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters(), lr=lr)
        if model:
            self.optimizer_decoder.load_state_dict(model['optimizer_decoder'])
        self.optimizer_cx = torch.optim.Adam(self.cx.parameters(), lr=lr)
        if model:
            self.optimizer_cx.load_state_dict(model['optimizer_cx'])
        self.optimizer_cz = torch.optim.Adam(self.cz.parameters(), lr=lr)
        if model:
            self.optimizer_cz.load_state_dict(model['optimizer_cz'])

    @staticmethod
    def gradient_penalty(model, real, fake, device):
        alpha = torch.randn((real.size(0), 1, 1), device=device)
        interpolates = (alpha * real + ((1 - alpha) * fake)
                        ).requires_grad_(True)

        model_interpolates = model(interpolates)
        grad_outputs = torch.ones(
            model_interpolates.size(), device=device, requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        norm = gradients.norm(2, dim=2)
        gradient_penalty = torch.mean((norm - 1) ** 2)
        return gradient_penalty

    def __critic_x_iteration(self, x, z):
        self.cx.train(True)
        self.decoder.train(True)
        self.optimizer_cx.zero_grad()

        fake_x = self.decoder(z)
        real_output = torch.squeeze(self.cx(x))
        fake_output = torch.squeeze(self.cx(fake_x))

        real_loss = torch.mean(real_output)
        fake_loss = torch.mean(fake_output)

        gp_loss = self.gradient_penalty(self.cx, x, fake_x, self.device)

        loss = self.loss_rate_critics[0]*fake_loss - \
            self.loss_rate_critics[1]*real_loss + \
            self.loss_rate_critics[2]*gp_loss
        loss.backward()

        self.optimizer_cx.step()

        return loss.item()

    def __critic_z_iteration(self, z, x):
        self.cz.train(True)
        self.encoder.train(True)
        self.optimizer_cz.zero_grad()

        fake_z = self.encoder(x)
        real_output = torch.squeeze(self.cz(z))
        fake_output = torch.squeeze(self.cz(fake_z))

        real_loss = torch.mean(real_output)
        fake_loss = torch.mean(fake_output)

        gp_loss = self.gradient_penalty(self.cz, z, fake_z, self.device)

        loss = self.loss_rate_critics[0]*fake_loss - \
            self.loss_rate_critics[1]*real_loss + \
            self.loss_rate_critics[2]*gp_loss
        loss.backward()

        self.optimizer_cz.step()

        return loss.item()

    def __encoder_decoder_iteration(self, x, z):
        self.encoder.train(True)
        self.decoder.train(True)
        self.cx.train(False)
        self.cz.train(False)
        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()

        fake_x = self.decoder(z)
        fake_z = self.encoder(x)
        reconstructed_x = self.decoder(fake_z)

        fake_x_output = torch.squeeze(self.cx(fake_x))
        fake_z_output = torch.squeeze(self.cz(fake_z))
        fake_x_loss = torch.mean(fake_x_output)
        fake_z_loss = torch.mean(fake_z_output)

        loss = - self.loss_rate_generator[0]*(fake_x_loss + fake_z_loss) + self.loss_rate_generator[1] * \
            torch.sqrt(torch.sum(torch.square(x - reconstructed_x)))

        loss.backward()

        self.optimizer_encoder.step()
        self.optimizer_decoder.step()

        return loss.item()

    @property
    def raw(self):
        if self.dataset is None:
            return []
        length = self.window_size + self.step_size * (len(self.dataset) - 1)
        ret = np.zeros((length, self.state_size))

        for row in self.dataset:
            start = row[0]
            values = row[1]
            for i in range(self.window_size):
                index = int(i + start)
                ret[index, :] = values[:, i]

        return ret

    def save(self, path='./', file_name='model.pth'):
        torch.save(
            {
                'window_size': self.window_size,
                'state_size': self.state_size,
                'step_size': self.step_size,
                'batch_size': self.batch_size,
                'num_epoch': self.num_epoch,
                'lr': self.lr,
                'num_critics': self.num_critics,
                'latent_dim': self.latent_dim,
                'loss_rate_critics': self.loss_rate_critics,
                'loss_rate_generator': self.loss_rate_generator,
                'encoder': self.encoder.to('cpu').state_dict(),
                'decoder': self.decoder.to('cpu').state_dict(),
                'cx': self.cx.to('cpu').state_dict(),
                'cz': self.cz.to('cpu').state_dict(),
                'optimizer_encoder': self.optimizer_encoder.state_dict(),
                'optimizer_decoder': self.optimizer_decoder.state_dict(),
                'optimizer_cx': self.optimizer_cx.state_dict(),
                'optimizer_cz': self.optimizer_cz.state_dict(),
            },
            f'{path}/{file_name}',
        )

    def train(self, num_epoch=100, debug=False):
        if self.dataset is None:
            raise TadGanError('invalid: none dataset')
        cx_loss = []
        cz_loss = []
        g_loss = []
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                  drop_last=True, shuffle=True, num_workers=os.cpu_count())
        for epoch in range(1, num_epoch+1):
            self.num_epoch += 1
            epoch_cx_loss = []
            epoch_cz_loss = []
            epoch_g_loss = []
            for batch, data in enumerate(train_loader):
                x = data[1].float().to(self.device)
                z = torch.empty(self.batch_size, 1, self.latent_dim).uniform_(
                    0, 1).to(self.device)

                loss = self.__critic_x_iteration(x, z)
                epoch_cx_loss.append(loss)

                loss = self.__critic_z_iteration(z, x)
                epoch_cz_loss.append(loss)

                if (batch + 1) % self.num_critics == 0:
                    epoch_g_loss.append(self.__encoder_decoder_iteration(x, z))

            cx_loss.append(np.mean(np.array(epoch_cx_loss)))
            cz_loss.append(np.mean(np.array(epoch_cz_loss)))
            g_loss.append(np.mean(np.array(epoch_g_loss)))

            if debug:
                print('Epoch: {}/{}, [Cx loss: {}] [Cz loss: {}] [G loss: {}]'.format(
                    epoch, num_epoch, cx_loss[-1], cz_loss[-1], g_loss[-1]))
        return (np.array(cx_loss), np.array(cz_loss),  np.array(g_loss))

    
    def reconstruct(self, values):
        if values.shape[1] != self.state_size:
            raise TadGanError('invalid: values unmatch state_size')
            
        self.decoder.train(False)
        self.encoder.train(False)
        reconstructed = []

        start = 0
        end = self.window_size
        while end <= len(values):
            fake = self.encoder(torch.tensor(values[start:end], device=self.device).view(
                1, self.state_size, self.window_size).float())
            reconstructed.append(self.decoder(fake).detach()[
                                  0].transpose(1, 0).tolist())
            start+=self.step_size
            end+=self.step_size
            

        num_row = len(reconstructed)
        length = self.window_size + self.step_size * (num_row - 1)
        ret = []
        for i in range(length):
            diagonal = []

            for j in range(max(0, i - length + self.window_size), min(i + 1, self.window_size)):
                if i - j < num_row:
                    diagonal.append(reconstructed[i - j][j])
            if len(diagonal) > 0:
                ret.append(np.median(np.array(diagonal), axis=0).tolist())
        return np.array(ret)


    def anomaly_score(self, raw, reconstructed, distance='point', combination='add', alpha=None, score_window=10):
        """
        refer to https://github.com/sintel-dev/Orion/blob/6548161e240a413ba011a699d109eb8a1e50c148/orion/primitives/tadgan.py#L345
        """
        if raw.shape != reconstructed.shape:
            raise TadGanError('invalid: raw and reconstructed are not same size.')
            
        self.cx.train(False)


        if distance == 'dtw':
            if len(raw) < score_window:
                score_window = len(raw)
            if score_window < 0:
                score_window = 1
            dtw_length = (score_window // 2) * 2 + 1
            dtw_half_length = dtw_length // 2

            distance = []
            for i in range(len(raw) - dtw_length):
                distance.append(
                    _dtw(raw[i:i + dtw_length], reconstructed[i:i + dtw_length])['cost'])
            reconstructed_score = np.clip(stats.zscore(np.array(distance)), a_min=0, a_max=None)
            reconstructed_score = np.pad(reconstructed_score, (dtw_half_length, len(raw) - len(distance) - dtw_half_length),
                   'constant', constant_values=(0, 0))
        else:
            reconstructed_score = np.sqrt(
                np.sum((raw - reconstructed)**2, axis=1))
            reconstructed_score = np.clip(stats.zscore(reconstructed_score), a_min=0, a_max=None)

        if combination is None:
            return reconstructed_score

        if alpha is None:
            alpha = 0.5 if combination == 'add' else 1
        if alpha > 1:
            alpha = 1
        if alpha < 0:
            alpha = 0

        cxr_rect = []
        start = 0
        end = self.window_size
        while end <= len(reconstructed):
           cxr_rect.append(np.repeat(self.cx(torch.tensor(reconstructed[start:end].transpose(1, 0)).view(
                1, self.state_size, self.window_size).to(self.device).float()).detach().tolist()[0][0], self.window_size).tolist())
           start+=self.step_size
           end+=self.step_size

        length = self.window_size + self.step_size * (len(cxr_rect) - 1)
        critic_score = []
        for i in range(length):
            cxr_inter = []
            for j in range(max(0, i - length + self.window_size), min(i + 1, self.window_size)):
                cxr_inter.append(cxr_rect[i - j][j])

            if len(cxr_inter) > 1:
                cxr_inter = np.array(cxr_inter)
                try:
                    critic_score.append(cxr_inter[np.argmax(
                        stats.gaussian_kde(cxr_inter)(cxr_inter))])
                except np.linalg.LinAlgError:
                    critic_score.append(np.median(cxr_inter))
            else:
                critic_score.append(cxr_inter[0])

        critic_score = np.clip(-stats.zscore(critic_score), a_min=0, a_max=None)

        if combination == 'mult':
            return alpha * reconstructed_score * critic_score

        return alpha * reconstructed_score + (1 - alpha) * critic_score
