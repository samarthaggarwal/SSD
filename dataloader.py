import cv2
import pickle
import glob 
import os 
import numpy as np 
import ipdb

IM_EXTENSIONS = ['png', 'jpg', 'bmp']


def read_img(img_path, img_shape=(128,128)):
    """
    load image file and divide by 255.
    """
    img = cv2.imread(img_path)
    orig_shape = img.shape
    img = cv2.resize(img, img_shape)
    img = img/255.0

    return img, orig_shape[:2]


def dataloader(dataset_dir, label_path, num_classes, batch_size=32, img_shape=(128, 128)):

    """
    data loader
    return image, [class_label, class_and_location_label]
    """
    
    if dataset_dir[-1]!="/":
        dataset_dir+="/"
    # img_files = glob.glob(dataset_dir)
    img_files = os.listdir(dataset_dir)
    img_files = [f for f in img_files if f[-3:] in IM_EXTENSIONS]

    with open(label_path, "rb") as f:
        labels = pickle.load(f)
    
    numofData = len(img_files)# endwiths(png,jpg ...)
    data_idx = np.arange(numofData)
    
    while True:
        batch_idx = np.random.choice(data_idx, size=batch_size, replace=False)
        
        batch_img = []
        batch_label = []
        # batch_label_cls = []
        
        for i in batch_idx:
            
            img, orig_shape = read_img(dataset_dir + img_files[i], img_shape=img_shape)
            label = labels[i]
            # transformation of labels
            # format : (class_id, b_x, b_y, b_w, b_h)
            for i in range(len(label)):
                temp = label[i]
                new_x = int((temp[1] * img_shape[0]) / orig_shape[0])
                new_y = int((temp[2] * img_shape[1]) / orig_shape[1])
                new_w = int((temp[3] * img_shape[0]) / orig_shape[0])
                new_h = int((temp[4] * img_shape[1]) / orig_shape[1])
                label[i] = (temp[0], new_x, new_y, new_w, new_h)

            # Transforming labels for outputs at different scales
            """
            for feature map at each scale, find where will the bounding box centre lie and recaliberate the coordinates wrt that grid box.
            output : 16x16x15, 8x8x15, 4x4x15, 2x2x15
            15-dim : 10+1 classes, 4 boundary params
            """
            # 16x16 feature maps, shape = (16,16,15)
            label_16 = np.zeros((16, 16, num_classes+4), dtype=float)
            for i in range(len(label)):
                temp = label[i]
                x = (temp[1] * 16) / img_shape[0]
                y = (temp[2] * 16) / img_shape[1]
                grid_x = int(x)
                grid_y = int(y)
                label_16[grid_x, grid_y, temp[0]+1]=1
                label_16[grid_x, grid_y, num_classes] = x-grid_x
                label_16[grid_x, grid_y, num_classes+1] = y-grid_y
                label_16[grid_x, grid_y, num_classes+2] = temp[3] / (img_shape[0] / 16)
                label_16[grid_x, grid_y, num_classes+3] = temp[4] / (img_shape[1] / 16)
            label_16_reshaped = label_16.reshape(16**2, num_classes+4)

            # 8x8 feature maps, shape = (8,8,15)
            label_8 = np.zeros((8, 8, num_classes+4), dtype=float)
            for i in range(len(label)):
                temp = label[i]
                x = (temp[1] * 8) / img_shape[0]
                y = (temp[2] * 8) / img_shape[1]
                grid_x = int(x)
                grid_y = int(y)
                label_8[grid_x, grid_y, temp[0]+1]=1
                label_8[grid_x, grid_y, num_classes] = x-grid_x
                label_8[grid_x, grid_y, num_classes+1] = y-grid_y
                label_8[grid_x, grid_y, num_classes+2] = temp[3] / (img_shape[0] / 8)
                label_8[grid_x, grid_y, num_classes+3] = temp[4] / (img_shape[1] / 8)
            label_8_reshaped = label_8.reshape(8**2, num_classes+4)

            # 4x4 feature maps, shape = (4,4,15)
            label_4 = np.zeros((4, 4, num_classes+4), dtype=float)
            for i in range(len(label)):
                temp = label[i]
                x = (temp[1] * 4) / img_shape[0]
                y = (temp[2] * 4) / img_shape[1]
                grid_x = int(x)
                grid_y = int(y)
                label_4[grid_x, grid_y, temp[0]+1]=1
                label_4[grid_x, grid_y, num_classes] = x-grid_x
                label_4[grid_x, grid_y, num_classes+1] = y-grid_y
                label_4[grid_x, grid_y, num_classes+2] = temp[3] / (img_shape[0] / 4)
                label_4[grid_x, grid_y, num_classes+3] = temp[4] / (img_shape[1] / 4)
            label_4_reshaped = label_4.reshape(4**2, num_classes+4)

            # 2x2 feature maps, shape = (2,2,15)
            label_2 = np.zeros((2, 2, num_classes+4), dtype=float)
            for i in range(len(label)):
                temp = label[i]
                x = (temp[1] * 2) / img_shape[0]
                y = (temp[2] * 2) / img_shape[1]
                grid_x = int(x)
                grid_y = int(y)
                label_2[grid_x, grid_y, temp[0]+1]=1
                label_2[grid_x, grid_y, num_classes] = x-grid_x
                label_2[grid_x, grid_y, num_classes+1] = y-grid_y
                label_2[grid_x, grid_y, num_classes+2] = temp[3] / (img_shape[0] / 2)
                label_2[grid_x, grid_y, num_classes+3] = temp[4] / (img_shape[1] / 2)
            label_2_reshaped = label_2.reshape(2**2, num_classes+4)

            # shape = (16**2 + 8**2 + 4**2 + 2**2 = 340, num_classes+4=15)
            label_concat = np.concatenate((label_16_reshaped, label_8_reshaped, label_4_reshaped, label_2_reshaped), axis=0)

            # ipdb.set_trace()
            batch_img.append(img)
            batch_label.append(label_concat.tolist())
            # batch_label_cls.append(label_concat[0:1])
            
        # yield np.array(batch_img, dtype=np.float32), 
        # [np.array(batch_label_cls, dtype=np.float32), np.array(batch_label, dtype=np.float32)]
        yield np.array(batch_img, dtype=np.float32), np.array(batch_label, dtype=np.float32)


if __name__ == "__main__":
    pass
