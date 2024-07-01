import torch
import cv2
import os
from tqdm import tqdm

from img_visualizer import ImageVisualizer
from metrics import DetectionMetrics
from utils import save_best_metrics_to_csv
import matplotlib.pyplot as plt



def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized_tensor

def diffusion_inference(model, image, num_classes, device, num_iterations=10, draw_class=1):
    model.eval()#########################
    with torch.no_grad():
        
        # image = min_max_normalize(image)
        
        batch_size, _, height, width = image.shape
        combined = torch.empty(batch_size, 1 + num_classes, height, width, device=image.device)
        
        # Генерируем случайный шум и объединяем с изображением
        noise = torch.rand(batch_size, num_classes, height, width, device=image.device)
        print("noise", noise)
        
        # нули сделаю 
        image = torch.zeros(batch_size, 1, height, width)
        noise = torch.zeros(batch_size, num_classes, height, width)
        ###
        
        combined[:, 0, :, :] = image[:, 0, :, :]
        combined[:, 1:, :, :] = noise

        # Путь для сохранения промежуточных изображений шума
        noisy_path = "noisy_masks"
        if not os.path.exists(noisy_path):
            os.makedirs(noisy_path)

        # # Сохраняем начальный шум и изображение
        # cv2.imwrite(f"{noisy_path}/noisy_mask_before.jpg", (noise[0][0].detach().cpu().numpy() * 255).astype('uint8'))
        # cv2.imwrite(f"{noisy_path}/image.jpg", (combined[0, 0, :, :].detach().cpu().numpy() * 255).astype('uint8'))

        plt.imsave(f"{noisy_path}/noisy_mask_before.jpg", noise[0][0].detach().cpu().numpy(), cmap='gray')
        plt.imsave(f"{noisy_path}/image.jpg", combined[0, 0, :, :].detach().cpu().numpy(), cmap='gray')

        
        for _ in tqdm(range(num_iterations), desc="Diffusion Iterations"):
            outputs = torch.tanh(model(combined))
            print("outputs", outputs)
            combined[:, 1:, :, :] = ((combined[:, 1:, :, :] - outputs + 1) / 3.0).clamp(0, 1)

        # Сохраняем конечный шум
        # cv2.imwrite(f"{noisy_path}/noisy_mask_after.jpg", (combined[0, 1 + draw_class].detach().cpu().numpy() * 255).astype('uint8'))
        plt.imsave(f"{noisy_path}/noisy_mask_after.jpg", combined[0, 1 + draw_class].detach().cpu().numpy(), cmap='gray')

        
        
    final_mask = combined[:, 1:, :, :]
    return final_mask

def old_diffusion_inference(model, image, num_classes, device, num_iterations=10):
    model.eval()
    with torch.no_grad():
        noisy_mask = torch.randn(image.size(0), num_classes, image.size(2), image.size(3)).to(device)  # Генерируем случайный шум
        print("noisy_mask shape", noisy_mask.shape)
        print("image,shape", image.shape)
        
        # посмотрю на шум 
        noisy_path = "noisy_masks"
        if not os.path.exists(noisy_path):
            os.makedirs(noisy_path)

        cv2.imwrite(f"{noisy_path}/noisy_mask_before.jpg", (noisy_mask[0][0].detach().cpu().numpy()* 255).astype('uint8'))
        cv2.imwrite(f"{noisy_path}/image.jpg", (image[0][0].detach().cpu().numpy()* 255).astype('uint8'))
        
        for _ in tqdm(range(num_iterations)):
            inputs = torch.cat((image, noisy_mask), dim=-3)
            print("inputs shape", inputs.shape)
            predicted_noise = model(inputs)
            predicted_noise = torch.tanh(predicted_noise)
            print("predicted_noise shape", predicted_noise.shape)
            
            print("[:, 1:, :, :].shape", inputs[:, 1:, :, :].shape)
            noisy_mask[:, 1:, :, :] = (inputs[:, 1:, :, :] - predicted_noise + 1) / 3
            noisy_mask = torch.clamp(noisy_mask, 0, 1)
            
            # # print("predicted_noise", predicted_noise)
            # # print("noisy_mask after", noisy_mask)
            # noisy_mask = (noisy_mask - predicted_noise + 1) / 3
            # # print("noisy_mask shape", noisy_mask.shape)
            # # print("predicted_noise shape", predicted_noise.shape)
            # noisy_mask = torch.clamp(noisy_mask, 0, 1)
            # # print("noisy_mask", noisy_mask)

    cv2.imwrite(f"{noisy_path}/noisy_mask_after.jpg", (noisy_mask[0][1].detach().cpu().numpy()* 255).astype('uint8'))
        
    final_mask = noisy_mask
    return final_mask

def predict(net, image, num_classes, draw_class):
    

    
    combined = torch.empty(1,
                           image.size(0) + num_classes,
                           image.size(1),
                           image.size(2),
                           device=image.device)

    noise = torch.rand(num_classes, image.size(1), image.size(2), device=image.device)
    combined[0] = torch.cat([image, noise], dim=0)*0 ##################################### нули сделал
    
    # outputs = torch.tanh(net(combined)) # Прямой проход
    # print("outputs", outputs)
    
    noisy_path = "noisy_masks"
    
    with torch.no_grad():
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(3):
            for j in range(3):
                
                net_out = net(combined)
                # print("net_out", net_out)
                outputs = torch.tanh(net_out) # Прямой проход
                # print("outputs", outputs)
                combined[:,1:,:, :] = ((combined[:,1:,:, :] - outputs + 1)/3.0)
                print(i+j, combined[0, 1].max(), combined[0, 2].max())
                
                axs[i, j].imshow(combined[0, 1+draw_class].cpu().detach().numpy())
                axs[i, j].axis('off')

    
    # plt.tight_layout()
    # plt.show()
    
    plt.tight_layout()
    fig.savefig(f"{noisy_path}/image_predict.jpg", bbox_inches='tight')
    plt.close(fig) 
    
  


def test_model(
    model,
    model_weight,
    criterion,
    train_loader,
    train_predict_path,
    val_loader,
    val_predict_path,
    device,
    num_classes,
    epoch=None,
):
    train_image_visualizer = ImageVisualizer(train_predict_path)
    val_image_visualizer = ImageVisualizer(val_predict_path)

    # writer = SummaryWriter(log_dir="weak_logs")
    metrics_calculator = DetectionMetrics(mode="ML", num_classes=num_classes)

    # model.load_state_dict(torch.load(model_weight))
    # model.to(device)
    # model.eval()  # Перевод модели в режим оценки

    # class_names_dict = {class_info['id']: class_info['name'] for class_info in SCT_out_classes}
    # class_names_dict = {
    #     1: "Внутримозговое кровозлияние",
    #     2: "Субарахноидальное кровозлияние",
    #     3: "Cубдуральное кровозлияние",
    #     4: "Эпидуральное кровозлияние",
    # }

    class_names_dict = {
        1: "Снижение пневматизации околоносовых пазух",
        2: "Горизонтальный уровень жидкость-воздух",
    }

    # Инициализация переменных для метрик
    val_loss_sum = 0.0
    val_iou_sum = torch.zeros(num_classes)

    colors = [
        ((251, 206, 177), "Абрикосовым"),
        ((127, 255, 212), "Аквамариновым"),
        ((255, 36, 0), "Алым"),
        ((153, 102, 204), "Аметистовым"),
        ((153, 0, 102), "Баклажановым"),
        ((48, 213, 200), "Бирюзовым"),
        ((152, 251, 152), "Бледно зеленым"),
        ((213, 113, 63), "Ванильным"),
        ((100, 149, 237), "Васильковым"),
        ((34, 139, 34), "Зелёный лесной"),
        ((0, 0, 255), "Синий"),
        ((75, 0, 130), "Индиго"),
        ((255, 0, 255), "Чёрный"),
        ((0, 51, 153), "Маджента"),
        ((65, 105, 225), "Королевский синий"),
        ((255, 255, 0), "Жёлтый"),
        ((255, 69, 0), "Оранжево-красный"),
        ((255, 0, 0), "Темно синим"),
        ((0, 51, 153), "Красный"),
        ((255, 215, 0), "Золотой"),
        ((250, 128, 114), "Лососевый"),
        ((255, 99, 71), "Томатный"),
        ((255, 215, 0), "Золотой"),
        ((0, 139, 139), "Тёмный циан"),
        ((0, 255, 255), "Морская волна"),
    ]

    # было так
    # with torch.no_grad():
    #     for train_batch in train_loader:
    #         # optimizer.zero_grad()
    #         images = train_batch["images"].to(device)
    #         masks = train_batch["masks"][:, 1:, :, :].to(device)
    #         outputs = model(images)
    #         outputs = torch.sigmoid(outputs)

    #         # loss = criterion(outputs, masks)
    #         # loss.backward()
    #         # optimizer.step()
    #         # train_loss_sum += loss.item()

    #         # train_iou_batch = iou_metric(outputs, masks, num_classes)
    #         # # train_iou_sum += torch.sum(train_iou_batch, dim=0)  # Суммирование IoU для каждого класса по всем батчам
    #         # # вроде так
    #         # train_iou_sum += train_iou_batch

    #         # # для трейна метрики тоже посчитаю
    #         # metrics_calculator.update_counter(masks, outputs)

    #         # уберу пока
    #         train_image_visualizer.visualize(
    #             images, masks, outputs, class_names_dict, colors, epoch
    #         )

    #     for k, val_batch in enumerate(val_loader):
    #         images_val = val_batch["images"].to(device)
    #         masks_val = val_batch["masks"][:, 1:].to(device)
    #         outputs_val = model(images_val)

    #         outputs_val = torch.sigmoid(outputs_val)

    #         # лоссы я убрал
    #         # val_loss_sum += criterion(outputs_val, masks_val).item()
    #         # val_iou_batch = iou_metric(outputs_val, masks_val, num_classes)
    #         # val_iou_sum += val_iou_batch
    #         # metrics_calculator.update_counter(masks_val, outputs_val)

    #         # уберу пока
    #         val_image_visualizer.visualize(
    #             images_val, masks_val, outputs_val, class_names_dict, colors, epoch
    #         )

    #     # тут все норм но я в даталоудеры хочу 32 элемента передать и тогда тут ошибка
    #     try:
    #         val_loss_avg = val_loss_sum / len(val_loader)
    #         val_iou_avg = val_iou_sum / len(val_loader)

    #         # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_avg}, Val Loss: {val_loss_avg},  Val IoU: {val_iou_avg}")
    #         print(f"Val Loss: {val_loss_avg},  Val IoU: {val_iou_avg}")
            
    #     except:
    #         pass
        
    #     # val_metrics = metrics_calculator.calc_metrics()

    #     # base_name = model_weight.split("/")[1]
    #     # parts = base_name.split('_')
    #     # experiment_name = parts[1]
    #     # print(f"Experiment name: {experiment_name}")
        
    #     # best_metrics = {
    #     #     "experiment": experiment_name,
    #     #     "epoch": 119, # вот тут посто пока 3 эпоху напишу, я ее никак не сохранял
    #     #     "train_loss": 0.2,  # не пишу пока
    #     #     "val_loss": 0.23, # и тут
    #     #     "val_metrics": {
    #     #         "IOU": val_metrics["IOU"],
    #     #         "F1": val_metrics["F1"],
    #     #         "area_probs_F1": val_metrics["area_probs_F1"],
    #     #     },
    #     # }
        
    #     # best_model_path = "sinusite_last_models"
    #     # csv_file = f"{best_model_path}/last_metrics.csv"
    #     # save_best_metrics_to_csv(best_metrics, csv_file)
        

    # # для диффузии делаю 
    # num_iterations=1
    
    # with torch.no_grad():
    #     # for train_batch in train_loader:
    #     #     images = train_batch["images"].to(device)
    #     #     initial_noise = torch.rand_like(images[:, :1, :, :]).to(device)

    #     #     outputs = diffusion_inference(model, images, num_classes, device, num_iterations)

    #     #     train_image_visualizer.visualize_diffusion(
    #     #         images, outputs, class_names_dict, colors, num_iterations
    #     #     )

    #     for val_batch in val_loader:
    #         images_val = val_batch["images"].to(device)

    #         outputs_val = diffusion_inference(model, images_val, num_classes, device, num_iterations)

    #         val_image_visualizer.visualize_diffusion(
    #             images_val, outputs_val, class_names_dict, colors, num_iterations=9
    #         )


    model.load_state_dict(torch.load('/home/imran-nasyrov/best_diffusion_model.pth'))

    model.to(device)


    num_epochs = 300  # Количество эпох для тренировки

    for epoch in range(num_epochs):
        # model.eval() 
        model.train()
        running_loss = 0.0
        loop = tqdm(val_loader, leave=True)
        for result in loop: 
            images, masks = result["images"].to(device, dtype=torch.float32), result["masks"][:, 1:, :, :].to(device, dtype=torch.float32)
            predict(model, images[0], 2, 1)