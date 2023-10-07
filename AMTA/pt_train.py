# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from amta3D_A import AMTA_Net3D_A
from dataset import create_data_loader
from loss_function import loss_dice

print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
folder_name = '../../autodl-tmp/data'
RESULT_PATH = '../../autodl-tmp/result'


def model_train(train_loader, target_label, auxiliary_label, total_steps):
    ## training

    model = AMTA_Net3D_A()
    if use_cuda:
        model.cuda()
    step = 0
    loss_record = []
    target_record = []
    aux_record = []
    step_record = []
    min_target = 1000
    min_name = ''
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=42, T_mult=3)
    while step < total_steps:
        for ii, (images, labels, _, _) in enumerate(train_loader):
            step += 1
            if step > total_steps:
                break
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            preds, mid_output = model(images)  # 3D

            loss_pred = loss_dice(preds, labels, target_label)  
            loss_mid = loss_dice(mid_output, labels, auxiliary_label)

            loss = loss_pred + loss_mid
            # loss = loss_pred # no auxiliary task 

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Compute and print loss
            print('Step %d loss: %.5f, target: %.5f, aux: %.5f; lr: %.8f' % (step, loss.item(), loss_pred.item(), loss_mid.item(), optimizer.state_dict()['param_groups'][0]['lr']))
            if step % 1000 == 1:
                model_name = 'saved_model%d' % step
                torch.save(model, os.path.join(RESULT_PATH, model_name))
                print('model saved')
            if loss_pred < min_target:
                min_target = loss_pred
                if min_target < 0.1:
                    model_name = 'saved_model_target_loss_%.5f_step_%d' % (loss_pred, step)
                    min_name = model_name
                    torch.save(model, os.path.join(RESULT_PATH, model_name))
                    print('model saved')
            
            loss_record.append(loss.item())
            target_record.append(loss_pred.item())
            aux_record.append(loss_mid.item())
            step_record.append(step)

    plt.figure(figsize=(10, 6))
    record = open(os.path.join(RESULT_PATH, 'record.txt'), 'w')
    print(step_record, loss_record, target_record, aux_record, file=record)
    plt.plot(step_record, loss_record, label='Total Dice loss')
    plt.plot(step_record, target_record, label='Main task Dice loss')
    plt.plot(step_record, aux_record, label='Auxiliary task Dice loss')
    plt.legend()
    loss_img_name = 'loss_curve'
    plt.savefig(loss_img_name)
    print('Training done.')

    # save trained model
    model_name = 'saved_model_final_loss_%.5f_step_%d' % (loss_pred, step)
    torch.save(model, os.path.join(RESULT_PATH, model_name))
    if min_name == '':
        min_name = model_name
    print('Model saved.')
    return model, min_name


def get_pred_img(pred):
    return torch.argmax(pred, dim=1)

def model_test(model, test_loader, target_label):
    model.eval()
    results = []
    for id, (image, label, fn, _) in enumerate(test_loader):
        with torch.no_grad():
            pred, _ = model(image.cuda())
            dice_score = loss_dice(pred, label.cuda(), target_label)
            img = image[0].cuda()
            lab = label[0].cuda()
            pre_fore = pred[:, 0]
            pre_back = pred[:, 1]
            torch.save(torch.cat((img, lab, pre_fore, pre_back), dim=0).detach().cpu(), 'result/result%d_%s' % (id, fn[0]))
            dice_np = dice_score.detach().cpu() 
            results.append(1 - dice_np)

    mean_dice = np.mean(results)
    print('Final mean dice is:' + str(mean_dice))
    return mean_dice


if __name__ == '__main__':
    target_label=[7]
    auxiliary_label=[4,5,6,7]
    train_loader, test_loader = create_data_loader(folder_name, target_label, slice_dataset=True)
    model, min_name = model_train(train_loader, target_label=target_label, auxiliary_label=auxiliary_label, total_steps=5000)
    model = torch.load('../../autodl-tmp/result/%s' % min_name)
    model_test(model, test_loader, target_label)
