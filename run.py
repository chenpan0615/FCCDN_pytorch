from networks import GenerateNet
from loss import FCCDN_loss

class Config():
    def __init__(self):
        self.MODEL_NAME = 'FCCDN'
        self.MODEL_OUTPUT_STRIDE = 16
        self.BAND_NUM = 3
        self.USE_SE = True

if __name__ == '__main__':
    import torch
    toy_data1 = torch.rand(2,3,256,256).cuda()
    toy_data2 = torch.rand(2,3,256,256).cuda()
    input = [toy_data1, toy_data2]
    toy_label1 = torch.randint(0, 2, (2,1,256,256)).cuda()
    toy_label2 = torch.randint(0, 2, (2,1,128,128)).cuda()
    label = [toy_label1.float(), toy_label2.float()]
    cfg = Config()
    model = GenerateNet(cfg)
    model = model.cuda()
    output = model(input)
    loss = FCCDN_loss(output, label)
    print(loss)
