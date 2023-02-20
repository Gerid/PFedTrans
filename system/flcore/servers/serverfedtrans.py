import torch
from torch import nn
import copy
import time
import numpy as np
import math
import random

from flcore.clients.clienttrans import clientTrans,Cluster, weight_flatten
from flcore.servers.serverbase import Server
from utils.kmeans import kmeans
from threading import Thread


#git clone https://github.com/subhadarship/kmeans_pytorch
#cd kmeans_pytorch
#pip install --editable .
class FedTrans(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientTrans)


        #cluster_number
        self.every_recluster_eps = args.every_recluster_eps
        self.num_cluster = args.num_cluster

        # model embedding layer
        self.emb_dim = args.emb_dim

        #head_params =  torch.cat([p.flatten() for p in self.global_model.head.parameters()])
        #self.emb_layer = nn.Linear(len(head_params), self.emb_dim).to(self.device)
        self.emb_layer = nn.Linear(len(nn.utils.parameters_to_vector(self.global_model.head.parameters())),self.emb_dim).to(self.device)
        self.alpha_layer = nn.Linear(self.emb_dim, 1).to(self.device)

        # attn init
        self.attn_learning_rate = args.attn_learning_rate
        self.attn_init()
        # TK -- ratio of cur head update (1-TK,TK)(agg_head, cur_head)
        self.tk = args.tk_ratio


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def attn_init(self):
        self.intra_attn_model = Attn_Model(emb_dim=self.emb_dim*2, attn_dim=128*2).to(self.device) # emb client model + grad
        self.inter_attn_model = Attn_Model(emb_dim=self.emb_dim).to(self.device) # emb cluster 
        self.attn_optimizer = torch.optim.SGD([
                {'params': self.emb_layer.parameters()},
                {'params': self.inter_attn_model.parameters()},
                {'params': self.intra_attn_model.parameters()},
                 {'params': self.alpha_layer.parameters()},           ], lr=self.attn_learning_rate, momentum=0.9)
        self.attn_loss = nn.MSELoss().to(self.device)

    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            if(i==0):
                self.send_models()
            else:
                self.send_models(init_head=False)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                if i != 0:
                    client.prev_head = weight_flatten(client.model.head)
                client.train()
                client.cur_head = copy.deepcopy(client.model.head)
                client.emb(self.emb_layer)

            if i != 0:
                self.attn_optimize()
            else:
                self.form_cluster(reform=True)
                # log cluster results
                for idx, clus in enumerate(self.clusters):
                    pass
            
            self.recluster = True
            if self.recluster == True and i%self.every_recluster_eps == 0 and i!=0:
                self.form_cluster(self.recluster)
                #log recluster info
                #self.logger()
            
            for cluster in self.active_clusters:
                cluster.avg_update()
                cluster.emb(self.emb_layer)
            
            res = {}
            res['intra_clusters_res'] = []
            for cluster in self.active_clusters:
                res['intra_clusters_res'].append(self.intra_cluster_agg(cluster))
            res['inter_clusters_res'] = self.inter_cluster_agg()
            torch.save(res, "res.pt")
            
            for cluster in self.clusters:
                for client in cluster.clients:
                    alpha = self.alpha_layer(client.emb_vec)
                    alpha = torch.sigmoid(alpha)
                    client.model.head = self.w_add_params([self.tk*alpha,self.tk*(1-alpha),1-self.tk],[cluster.model.head, client.model.head, client.cur_head])

            # threads = [Thread(target=client.train)
            #            for client in self.clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()

    def send_models(self, init_head=True):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model,init_head)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def form_cluster(self, reform=False):
        #compute similarity k-means
        client_emb_list = [client.emb_vec.data.clone().reshape(1, -1) for client in self.clients]
        client_emb_list = torch.cat(client_emb_list, dim=0)
        
        cluster_res = kmeans(X=client_emb_list, num_clusters=self.num_cluster, distance='euclidean',iter_limit=200, device=self.device)

        if reform:
            self.clusters = [Cluster(c_id, self.global_model) for c_id in range(self.num_cluster)]

        for client_id, cluster_id  in enumerate(cluster_res[0]):
           self.clusters[cluster_id].clients.append(self.clients[client_id])

        self.active_clusters = []
        for cluster in self.clusters:
            if(len(cluster.clients)>0):
                self.active_clusters.append(cluster)
        
            #cluster.per_layer is the centroid per_model for clients within cluster

    def attn_optimize(self):
        loss = 0
        total_train = 0
        self.attn_optimizer.zero_grad()
        for client in self.clients:
            total_train += client.train_samples
        for i, client in enumerate(self.selected_clients):
            ratio = client.train_samples / total_train
            loss += ratio * torch.linalg.norm(nn.utils.parameters_to_vector(client.model.head.parameters()) - nn.utils.parameters_to_vector(client.prev_head))
        loss.backward()
        self.attn_optimizer.step()

    def inter_cluster_agg(self):
        cluster_emb_list = [cluster.emb_vec.clone().reshape(1, -1) for cluster in self.active_clusters]
        x = torch.cat(cluster_emb_list, dim=0).squeeze(1)
        weights = self.inter_attn_model(x)
        res = [cluster_emb_list,weights]
        cluster_model_list = [copy.deepcopy(cluster.model.head) for cluster in self.clusters]
        for i in range(weights.size()[0]):
            w = [weights[i][j] for j in range(weights[i].size()[0])]
            head = self.w_add_params(w, cluster_model_list)
            self.active_clusters[i].model.head = head
        return res
        

    def intra_cluster_agg(self, cluster):
        client_emb_list = [client.emb_vec.clone().reshape(1, -1) for client in cluster.clients]

        x = torch.cat(client_emb_list, dim=0).squeeze(1)
        if len(cluster.clients) == 1:
            return
        weights = self.intra_attn_model(x)
        print("weights:{}".format(weights))
        client_model_list = [] 
        for i in range(len(cluster.clients)):
            head = copy.deepcopy(cluster.clients[i].model.head)
            client_model_list.append(head)
        weights = weights.squeeze(0)
        res = [client_emb_list,weights]
        for i in range(weights.size()[0]):
            w = [weights[i][j] for j in range(weights[i].size()[0])]
            head = self.w_add_params(w, client_model_list)
            cluster.clients[i].model.head = head
            #client.head_temp = heads
        return res

    def cluster_update(self):
        for cluster in self.clusters:
            cluster.update_model()
            cluster.emb_vec = self.emb_layer(cluster.model.head)

    def w_add_params(self, m_weights, models):
        res = copy.deepcopy(models[0])
        for param in res.parameters():
            param.data.zero_()
            
        for w, model in zip(m_weights, models):
            for res_param, model_param in zip(res.parameters(), model.parameters()):
                res_param.data += model_param.data.clone() * w
        return res

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model.base = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.base.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
    
    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.base.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

class Attn_Model(nn.Module):
    def __init__(self, emb_dim=128, attn_dim=128, num_heads=8):
        super(Attn_Model, self).__init__()
        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.query = nn.Linear(emb_dim, attn_dim)
        self.key = nn.Linear(emb_dim, attn_dim)
        #self.inter_LN = nn.LayerNorm(attn_dim)

        # 1-layer attention for simple verify

    def forward(self, x, models=None, prev_models=None):
        #x = self.inter_LN(x) 
        q = self.query(x)
        k = self.key(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) 

        #scaled coef removed since we want to diff weight matrix entries
        #scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attn_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights