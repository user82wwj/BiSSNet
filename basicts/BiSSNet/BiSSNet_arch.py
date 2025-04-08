import torch
from torch import nn

from .mlp import MultiLayerPerceptron, GraphMLP, FusionMLP
from .transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from .mlp_mod import FusionMLP_mod

device = 'cuda'


class BiSSNet(nn.Module):

    def __init__(self, adj_mx, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.output_len = model_args["output_len"]

        self.his_len = model_args["his_len"]
        self.if_enhance = model_args["if_enhance"]
        self.enhance_dim = model_args["enhance_dim"]
        self.if_en = model_args["if_en"]
        self.if_de = model_args["if_de"]

        self.fusion_num_step = model_args["fusion_num_step"]
        self.fusion_num_layer = model_args["fusion_num_layer"]
        self.fusion_dim = model_args["fusion_dim"]
        self.fusion_out_dim = model_args["fusion_out_dim"]
        self.fusion_out_dim_half = self.fusion_out_dim // 2
        self.fusion_dropout = model_args["fusion_dropout"]

        self.if_forward = model_args["if_forward"]
        self.if_backward = model_args["if_backward"]
        self.if_ada = model_args["if_ada"]
        self.adj_mx = adj_mx
        self.node_dim = model_args["node_dim"]
        self.tensor_trans = self.node_dim // 2
        self.nhead = model_args["nhead"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]

        self.graph_num = 1 * self.if_forward + 1 * self.if_backward + 1 * self.if_ada

        self.st_dim = (self.graph_num > 0) * self.node_dim + \
                      self.if_time_in_day * self.temp_dim_tid + \
                      self.if_day_in_week * self.temp_dim_diw

        self.st_dim_half = (self.graph_num > 0) * self.tensor_trans + \
                           self.if_time_in_day * self.temp_dim_tid + \
                           self.if_day_in_week * self.temp_dim_diw

        self.output_dim = self.fusion_num_step * self.fusion_out_dim


        if self.if_forward:
            self.adj_mx_forward_encoder = nn.Sequential(
                GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
            )
            self.adj_mx_forward_encoder_trans = nn.Sequential(
                GraphMLP(input_dim=self.tensor_trans, hidden_dim=self.node_dim)
            )
            self.forward_trans = nn.Parameter(torch.randn(self.num_nodes, self.tensor_trans), requires_grad=True)
            ###mod
            self.adj_mx_forward_encoder_half = nn.Sequential(
                GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim // 2)
            )

        if self.if_backward:
            self.adj_mx_backward_encoder = nn.Sequential(
                GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
            )
            self.adj_mx_backward_encoder_trans = nn.Sequential(
                GraphMLP(input_dim=self.tensor_trans, hidden_dim=self.node_dim)
            )
            self.backward_trans = nn.Parameter(torch.randn(self.num_nodes, self.tensor_trans), requires_grad=True)
            ###mod
            self.adj_mx_backward_encoder_half = nn.Sequential(
                GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim // 2)
            )
        if self.if_ada:
            self.adj_mx_ada = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.adj_mx_ada)

        self.fusion_layers = nn.ModuleList([
            FusionMLP(
                input_dim=self.st_dim + self.input_len + self.if_de * self.input_len + self.if_enhance * self.enhance_dim,
                hidden_dim=self.st_dim + self.input_len + self.if_de * self.input_len + self.if_enhance * self.enhance_dim,
                out_dim=self.fusion_out_dim,
                graph_num=self.graph_num,
                first=True, **model_args)
        ])
        for _ in range(self.fusion_num_step - 1):
            self.fusion_layers.append(
                FusionMLP(input_dim=self.st_dim + self.fusion_out_dim,
                          hidden_dim=self.st_dim + self.fusion_out_dim,
                          out_dim=self.fusion_out_dim,
                          graph_num=self.graph_num,
                          first=False, **model_args)
            )

        # half---------------
        self.fusion_layers_half = nn.ModuleList([
            FusionMLP_mod(
                input_dim=self.st_dim_half + self.input_len + self.if_de * self.input_len + self.if_enhance * self.enhance_dim,
                hidden_dim=self.st_dim_half + self.input_len + self.if_de * self.input_len + self.if_enhance * self.enhance_dim,
                out_dim=self.fusion_out_dim_half,
                graph_num=self.graph_num,
                first=True, **model_args)
        ])
        for _ in range(self.fusion_num_step - 1):
            self.fusion_layers_half.append(
                FusionMLP_mod(input_dim=self.st_dim_half + self.fusion_out_dim_half,
                              hidden_dim=self.st_dim_half + self.fusion_out_dim_half,
                              out_dim=self.fusion_out_dim_half,
                              graph_num=self.graph_num,
                              first=False, **model_args)
            )

        if self.fusion_num_step > 1:
            self.regression_layer = nn.Sequential(
                *[MultiLayerPerceptron(input_dim=self.output_dim,
                                       hidden_dim=self.output_dim,
                                       dropout=self.fusion_dropout)
                  for _ in range(self.fusion_num_layer)],
                nn.Linear(in_features=self.output_dim, out_features=self.output_len, bias=True),
            )

            self.regression_layer_half = nn.Sequential(
                *[MultiLayerPerceptron(input_dim=self.output_dim // 2,
                                       hidden_dim=self.output_dim // 2,
                                       dropout=self.fusion_dropout)
                  for _ in range(self.fusion_num_layer)],
                nn.Linear(in_features=self.output_dim // 2, out_features=self.output_len, bias=True),
            )

        if self.if_enhance:
            self.long_linear = nn.Sequential(
                nn.Linear(in_features=self.his_len, out_features=self.enhance_dim, bias=True),
            )

        if self.if_en:
            self.encoder = TransformerEncoder(
                TransformerEncoderLayer(d_model=self.input_len, nhead=self.nhead, dim_feedforward=4 * self.input_len,
                                        batch_first=True), num_layers=self.nhead)
        if self.if_de:
            self.decoder = TransformerDecoder(
                TransformerDecoderLayer(d_model=self.input_len, nhead=self.nhead, dim_feedforward=4 * self.input_len,
                                        batch_first=True), num_layers=self.nhead)
        if self.if_en:
            self.encoder_longtime = TransformerEncoder(
                TransformerEncoderLayer(d_model=self.input_len, nhead=self.nhead, dim_feedforward=4 * self.input_len,
                                        batch_first=True), num_layers=self.nhead)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:

        long_input_data_emb = []

        if self.if_enhance:
            long_input_data = long_history_data[..., 0].transpose(1, 2)
            long_input_data = self.long_linear(long_input_data)
            long_input_data_emb.append(long_input_data)

        input_data = history_data[..., 0].transpose(1, 2)
        batch_size, num_nodes, length = input_data.shape
        input_data_en = []
        input_data_de = []
        if self.if_en:
            input_data_en.append(self.encoder(input_data))
        else:
            input_data_en.append(input_data)
        if self.if_de:
            input_data_de.append(self.decoder(input_data, input_data_en[0]))

        long_batch_size, long_num_nodes, _ = long_input_data.shape
        long_input_data_en = []
        if self.if_en:
            long_input_data_en.append(self.encoder_longtime(long_input_data))
        else:
            long_input_data_en.append(long_input_data)

        time_series_emb = [torch.cat(long_input_data_en + input_data_en + input_data_de, dim=2)]

        node_forward_emb_half = []
        node_backward_emb_half = []
        node_ada_emb = []

        node_forward_emb = []
        node_backward_emb = []

        if self.if_forward:
            node_forward_graph = self.adj_mx[0].to(device)
            node_forward_graph = node_forward_graph.expand(batch_size, -1, -1)
            node_forward = self.adj_mx_forward_encoder(node_forward_graph)
            node_forward_emb.append(node_forward)

            node_forward = self.adj_mx_forward_encoder_half(node_forward_graph)
            node_forward_emb_half.append(node_forward)

        if self.if_backward:
            node_backward_graph = self.adj_mx[1].to(device)
            node_backward_graph = node_backward_graph.expand(batch_size, -1, -1)
            node_backward = self.adj_mx_backward_encoder(node_backward_graph)
            node_backward_emb.append(node_backward)

            node_backward = self.adj_mx_backward_encoder_half(node_backward_graph)
            node_backward_emb_half.append(node_backward)

        if self.if_ada:
            node_ada_emb.append(self.adj_mx_ada.unsqueeze(0).expand(batch_size, -1, -1))

        predicts = []
        predict_emb = []
        hidden_forward_emb = []
        hidden_backward_emb = []
        hidden_ada_emb = []
        for index, layer in enumerate(self.fusion_layers):
            predict, \
            hidden_forward, hidden_backward, hidden_ada, \
            node_forward_emb_out, node_backward_emb_out, node_ada_emb_out = layer(history_data,
                                                                                  time_series_emb, predict_emb,
                                                                                  node_forward_emb, node_backward_emb,
                                                                                  node_ada_emb,
                                                                                  hidden_forward_emb,
                                                                                  hidden_backward_emb, hidden_ada_emb)
            predicts.append(predict)
            predict_emb = [predict]
            time_series_emb = []
            hidden_forward_emb = hidden_forward
            hidden_backward_emb = hidden_backward
            hidden_ada_emb = hidden_ada

            node_forward_emb = node_forward_emb_out
            node_backward_emb = node_backward_emb_out
            node_ada_emb = node_ada_emb_out
        predicts = torch.cat(predicts, dim=2)

        # half------------------------------------
        predicts_half = []
        predict_emb_half = []
        hidden_forward_emb_half = []
        hidden_backward_emb_half = []
        time_series_emb = [torch.cat(long_input_data_emb + input_data_en + input_data_de, dim=2)]
        for index, layer in enumerate(self.fusion_layers_half):
            predict, \
            hidden_forward, hidden_backward, hidden_ada, \
            node_forward_emb_out, node_backward_emb_out, node_ada_emb_out = layer(history_data,
                                                                                  time_series_emb, predict_emb_half,
                                                                                  node_forward_emb_half,
                                                                                  node_backward_emb_half,
                                                                                  node_ada_emb,
                                                                                  hidden_forward_emb_half,
                                                                                  hidden_backward_emb_half,
                                                                                  hidden_ada_emb)
            predicts_half.append(predict)
            predict_emb_half = [predict]
            time_series_emb = []
            hidden_forward_emb_half = hidden_forward
            hidden_backward_emb_half = hidden_backward
            hidden_ada_emb = hidden_ada

            node_forward_emb_half = node_forward_emb_out
            node_backward_emb_half = node_backward_emb_out
            node_ada_emb = node_ada_emb_out
        #
        predicts_half = torch.cat(predicts_half, dim=2)

        if self.fusion_num_step > 1:
            predicts = self.regression_layer(predicts)
            predicts_half = self.regression_layer_half(predicts_half)
            predicts = predicts + predicts_half
        return predicts.transpose(1, 2).unsqueeze(-1)
