import torch
import numpy as np

EMBED_DIM = 15
EARLY_STOP_VAL = 10**-2
EARLY_STOP_ACC = 0.99

class SentRNN(torch.nn.Module):
    def __init__(self, inp_size=EMBED_DIM, h_size=EMBED_DIM, out_size=1, embeddings=None):
        #if embeddings == None:
        ##    raise NotImplementedError
        super(SentRNN, self).__init__()
        self.inp_size = inp_size
        self.h_size = h_size
        self.out_size = out_size

        self.encoder = torch.nn.Embedding(inp_size, h_size, sparse=False)
        self.recurrent = torch.nn.GRU(EMBED_DIM, 84, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True)
        self.fconnected = torch.nn.Linear(84, out_size)

        if embeddings:
            self.encoder.weight.data.copy_(embeddings)

        self.hidden_state = torch.zeros(2, 32, 84).to('cuda')#('cuda')
        self.device = 'cpu'
        if torch.cuda.is_available():
            self = self.to('cuda')
            self.device = 'cuda'

    def forward(self, x, label):
        #self.hidden_state = torch.zeros(2, 32, 84).long().to('cuda')#('cuda')
        #print(x.shape)
        x = self.encoder(x.long())
        #print(x.shape)
        x, hidden_state = self.recurrent(x.float(), self.hidden_state.float()) #self.hidden_state) #, self.hidden_state)
        self.hidden_state = hidden_state
        x = x[:, :, -1]
        #x, _ = torch.max(x, dim=2)  
        x = self.fconnected(x) #torch.transpose(x, 0, 1))
        #x = torch.sigmoid(x)
        label[label < 0] = 0
        #label = torch.nn.functional.one_hot(label, num_classes=2)
        #x = torch.nn.functional.sigmoid(x.squeeze(0))
        #loss = torch.nn.binary_cross_entropy
        x = torch.sigmoid(x)
        #sgn = torch.sign(x)
        #bin = torch.relu(sgn)
        #AA
        #loss = dice_loss(bin.squeeze(1), label.float())
        loss = torch.nn.functional.binary_cross_entropy(x.squeeze(1), label.float(), reduction="mean")
        #lp = torch.nn.functional.softmax(x, dim=1)
        self.hidden_state = self.hidden_state.detach()
        #self.recurrent.zero_grad()
        #x is not prob.., 
        return loss, x

    def train(self, train_x, train_y, lr=0.001):

        #optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam([
                {"params": self.encoder.parameters(), "lr": 0.001},
                {"params": self.recurrent.parameters(), "lr": 0.005},
                {"params": self.fconnected.parameters(), "lr": 0.005},
                ], lr=0.01, weight_decay=0.005)
        
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True)
        #train_x = train_x.unsqueeze(1)
        #train_y = train_y.unsqueeze(1) 
        #train_x = train_x.reshape(32, 1, -1)
        #train_y = train_y.reshape(32, 1, -1)
        train_x, val_x = train_x[:int(train_x.shape[0]*0.8)], train_x[int(0.8*train_x.shape[0]):]
        train_y, val_y = train_y[:int(train_y.shape[0]*0.8)], train_y[int(0.8*train_y.shape[0]):]
        zipped = list(zip(train_x, train_y))
        valzip = list(zip(val_x, val_y))
        dl = torch.utils.data.DataLoader(zipped,batch_size=32,shuffle=True, drop_last=True)
        vl = torch.utils.data.DataLoader(valzip,batch_size=32,shuffle=False, drop_last=True)
        iter = 0
        max_ep = 200

        checkpoint_name = "best_val_model.pth"
        checkpoint_accu = 0.0
        for epoch in range(max_ep):
            eploss = 0
            epcorr = 0
            epall  = 0
            for (target, label) in dl:
                optimizer.zero_grad()
                #target = torch.nn.functional.one_hot(target.long(), EMBED_DIM).to(self.device)
                target = target.to(self.device)
                label  = label.to(self.device)
                loss, x = self.forward(target, label)
                loss.backward()
                optimizer.step()
                eploss += loss.item()
                x = x.squeeze(1)
                x[x < 0.5]  = 0
                x[x >= 0.5] = 1
                epall += len(x)
                corr_t = (x == label)
                corr_t = corr_t[corr_t == True]
                epcorr += len(corr_t)
                if iter % 5 == 0:
                    pass
                    #print(target.shape)
                    #print(loss)
                    #print("inp shape", target.shape)
                iter += 1
                torch.nn.utils.clip_grad_norm(self.parameters(), 5)
            if epoch % 5 == 0:
                print(epoch, "/", max_ep, "eploss", eploss/len(dl))
                print("epacc:", epcorr/epall)
            if eploss/len(dl) < EARLY_STOP_VAL or epcorr/epall > EARLY_STOP_ACC:
                break

            eploss, epall, epcorr = 0, 0, 0
            for (target, label) in vl:
                #target = torch.nn.functional.one_hot(target.long(), EMBED_DIM).to(self.device)
                target = target.to(self.device)
                label  = label.to(self.device)
                with torch.no_grad():
                    _, x = self.forward(target, label)
                #eploss += loss.item()

                #print(x)
                x = x.squeeze(1)
                x[x < 0.5]  = 0
                x[x >= 0.5] = 1
                epall += len(x)
                corr_t = (x == label)
                corr_t = corr_t[corr_t == True]
                epcorr += len(corr_t)
                #print("this should be constant", len(dl))
            #print(epoch, "/", max_ep, "val-loss", eploss/len(dl))
            valaccu = epcorr/epall
            if valaccu > checkpoint_accu:
                torch.save(self, checkpoint_name)
                checkpoint_accu = valaccu
            if epoch % 5 == 0:
                print("valaccu:", epcorr/epall)
        self = torch.load(checkpoint_name)


    def test(self, test_x, test_y=None):
        #train_y = train_y.reshape(32, 1, -1)
        if test_y == None:
            test_y = torch.zeros_like(test_x) #dummy
        zipped = list(zip(test_x, test_y))
        dl = torch.utils.data.DataLoader(zipped,batch_size=32,shuffle=False, drop_last=True)
        #preds = np.zeros(shape=(list(test_y.shape)))
        preds = []
        iter = 0
        for (target, label) in dl:
            #target = torch.nn.functional.one_hot(target.long(), EMBED_DIM).to(self.device)
            target = target.to(self.device)
            label  = label.to(self.device)
            with torch.no_grad():
                loss, x = self.forward(target, label)
                x = x.squeeze(1)
                #preds[iter*32:(iter + 1)*32] = x.cpu().numpy()
                x_test = x
                #reference, not copy 
                #x_test[x < 0.5] = -1
                #x_test[x >= 0.5] = 1
                #print(x_test)
                #print(label)
                #print(x_test == label)
                preds += x.cpu().tolist()
                #print("loss", loss)
            iter += 1
        return np.array(preds)
