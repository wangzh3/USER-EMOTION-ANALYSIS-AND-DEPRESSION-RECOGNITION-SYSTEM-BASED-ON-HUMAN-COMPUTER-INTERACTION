import torch
import torch.utils.data
import numpy as np

root="/Users/adminadmin/Documents/mywork/master/"
path1=root+"AVEC2014_AudioVisual/Train.txt"
path2=root+"AVEC2014_AudioVisual/Test.txt"
savepath=root+"code/nn/advnnloss.txt"
BS=256
class mydataset(torch.utils.data.Dataset):
    def __init__(self,txtpath):
        super(mydataset, self).__init__()
        data=np.loadtxt(txtpath,dtype=np.str)
        self.data=data

    def __getitem__(self, index):
        matrixpath=self.data[index][0]
        matrix=np.loadtxt(matrixpath,dtype=np.float32)
        vec = matrix.reshape((1,2560))
        vec1=torch.tensor(vec, dtype=torch.float32)
        label=np.zeros(3,np.float)
        label_A=float(self.data[index][1])
        label_D = float(self.data[index][2])
        lavel_V=float(self.data[index][3])
        label[0]=label_A
        label[1]=label_D
        label[2]=lavel_V
        label1=torch.tensor(label, dtype=torch.float32)
        return vec1,label1
    def __len__(self):
        return self.data.shape[0]

traindataset = mydataset(txtpath=path1)
train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=BS, shuffle=True)

testdataset = mydataset(txtpath=path2)
test_loader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=BS, shuffle=True)
class advnn(torch.nn.Module):
    def __init__(self):
        super(advnn,self).__init__()
        self.layer1=torch.nn.Sequential(
            torch.nn.Linear(2560,20480),
            torch.nn.ReLU()
        )
        self.layer2=torch.nn.Sequential(
            torch.nn.Linear(20480,20480),
            torch.nn.ReLU(),

        )
        self.layer3=torch.nn.Sequential(
            torch.nn.Linear(20480,1024),
            torch.nn.ReLU()

        )
        self.layer4=torch.nn.Sequential(
            torch.nn.Linear(1024,3)
        )


    def forward(self, x):
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

adv=advnn()
print (adv)

optimizer=torch.optim.Adam(adv.parameters(),lr=0.0001)
loss_func=torch.nn.MSELoss()


times=int(traindataset.__len__()/BS)
print (times)
times1=int(testdataset.__len__()/BS)
print (times1)
file=open(savepath,"w")
print("epoch i trainloss trainA trainD trainV testloss testA testD testV")
file.write("trainA trainD trainV testA testD testV\n")

def myloss(output,label):
    number=output.shape[0]
    lossmatrix=(output-label)**2
    sum=torch.sum(lossmatrix, dim=0)
    return sum

loss1_train_list=[]
loss1_test_list=[]
for epoch in range(10):
    count_train=0
    count_test = 0
    loss1_train=0
    loss1_test=0
    for batch_idx, (input, label) in enumerate(train_loader):
        try:
            output = adv(input.view(-1,2560))
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss1_n = myloss(output, label)
            loss1_train+=loss1_n.detach().cpu()
            count_train+=input.size(0)

            s = str(epoch) + " " + str(batch_idx) + "/" + str(len(train_loader)) + " " + \
                str(loss.detach().numpy()) + " " + str(loss1_n.detach().numpy()[0]/input.size(0)) + " " + str(
                loss1_n.detach().numpy()[1]/input.size(0)) + " " + str(loss1_n.detach().numpy()[2]/input.size(0))
            print(s)
        except:
            pass



    loss1_train/=count_train

    for testinput, testlabel in test_loader:
        try:
            testoutput = adv(testinput.view(-1, 2560))
            mytestloss = myloss(testoutput, testlabel)
            loss1_test += mytestloss.detach().cpu()
            count_test+=testinput.size(0)
        except:
               pass
    loss1_test/=count_test

    print(loss1_train.numpy(), loss1_test.numpy())
    s1=str(loss1_train.numpy()[0])+" "+str(loss1_train.numpy()[1])+" "+str(loss1_train.numpy()[2])+" "+\
       str(loss1_test.numpy()[0])+" "+str(loss1_test.numpy()[1])+" "+str(loss1_test.numpy()[2])

    file.write(s1+"\n")

    loss1_train_list.append(loss1_train)
    loss1_test_list.append(loss1_test)
    torch.save({'model_state_dict':adv.state_dict(),
                'train': loss1_train_list, 'test': loss1_test_list}, 'adv'+str(epoch)+'.pt')

file.close()
torch.save(adv,'net.pkl')