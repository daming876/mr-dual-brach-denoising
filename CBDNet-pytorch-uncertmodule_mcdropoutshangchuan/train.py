import os, time, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter
from dataset.loader import Real, Syn
from model.cbdnet import Network, fixed_loss
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('--bs', default=1, type=int, help='batch size')	#读入数据这里，batch_size的设置一定要小于子文件夹个数，不然训练时损失函数一直为0,我们子文件夹为1，batchsize也设置为1.迭代5次后完成一个epoch
parser.add_argument('--ps', default=128, type=int, help='patch size')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=3000, type=int, help='sum of epochs')#原来默认是5000
args = parser.parse_args()


def train(train_loader, model, criterion, optimizer):
	losses = AverageMeter()
	model.train()

	for (noise_img, clean_img, sigma_img, flag) in train_loader:
		input_var = noise_img.cuda()
		target_var = clean_img.cuda()
		sigma_var = sigma_img.cuda()
		flag_var = flag.cuda()

		noise_level_est, output, score_map, variance_map = model(input_var)	#整体的class Network模型输出：noise_level（整体的预估噪声）, out（去噪后的图像）

		loss = criterion(output, target_var, noise_level_est, sigma_var,variance_map, flag_var)
		losses.update(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	return losses.avg


if __name__ == '__main__':
	save_dir = './save_model/'

	model = Network(dropout_rate=0.1).to(device)
	model.cuda()
	model = nn.DataParallel(model)

	if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
		# load existing model
		model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
		print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())	#重新初始化优化器
		optimizer.load_state_dict(model_info['optimizer'])	#并加载刚才初始化优化器的状态
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)	#初始化学习率调度器
		scheduler.load_state_dict(model_info['scheduler'])	#并加载刚才的初始化学习率调度器的状态
		cur_epoch = model_info['epoch']	#设置当前轮次
	else:
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		# create model
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

		# 初始化学习率调度器，这里使用了余弦退火调度策略。T_max是最大迭代次数，从命令行参数args.epochs获取。
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
		cur_epoch = 0
		
	criterion = fixed_loss()
	criterion.cuda()

	#train_dataset = Real('./data/SIDD_train/', 15, args.ps) + Syn('./data/Syn_train/', 15, args.ps)#原代码
	train_dataset = Real('/mnt/gemlab_data_2/jiang/HWformer/data/images/train/openneuro-clean-noisyhun/', 1, args.ps)+Syn('/mnt/gemlab_data_2/jiang/HWformer/data/images/train/openneuro-clean-noisy-sigmahun/', 1, args.ps) #使用openneuro进行的训练22*155*4=13640
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

	for epoch in range(cur_epoch, args.epochs + 1):
		loss = train(train_loader, model, criterion, optimizer)
		scheduler.step()

		torch.save({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'scheduler' : scheduler.state_dict()}, 
			os.path.join(save_dir, 'checkpoint.pth.tar'))

		print('Epoch [{0}]\t'
			'lr: {lr:.6f}\t'
			'Loss: {loss:.5f}'
			.format(
			epoch,
			lr=optimizer.param_groups[-1]['lr'],
			loss=loss))
