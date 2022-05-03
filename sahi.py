from PIL import Image
import os
import cv2
import torch
import numpy as np

def image_slice(img_path, cache_path, h, w):
    # img_path: Path of the image
    # cache_path: Path of cache folder
    # h: number of slices in height
    # w: number of slices in weidth
    im = Image.open(img_path)
    im_name = os.path.basename(img_path)
    im_name = os.path.splitext(im_name)[0]
    width, height = im.size
    block_width = int(width / w)
    block_height = int(height / h)

    for i in range(0, h):
        for j in range(0, w):
            if(j == w):
                left = 0 + j * block_width
                top = 0 + i * block_height
                right = width
                if(i == h):
                    bottom = height
                else:
                    bottom = (i + 1 ) * block_height
            else:
                left = 0 + j * block_width
                top = 0 + i * block_height
                right = ( j + 1 ) * block_width
                bottom = (i + 1 ) * block_height        
            # print("left:" + str(left) + " top: " + str(top) + " right: " + str(right) + " bottom: " + str(bottom))
            cropped_im = im.crop((left, top, right, bottom))
            cropped_im_name = im_name + "_h" + str(i) + "_w" + str(j) + ".jpg"
            cache_folder_path = os.path.join(cache_path, im_name)
            if(not os.path.exists(cache_folder_path)):
                os.makedirs(cache_folder_path)
            cache_name = os.path.join(cache_path, im_name,cropped_im_name)
            cropped_im.save(cache_name)
    # Returns the folder of cropped images
    return cache_folder_path

# Single image training

def single_img_train(img_path:str, cache_path:str, res_path:str, model, h:int, w:int):
    # img_path: Path of the image
    # cache_path: Path of cache folder
    # res_path: Path of results
    # model: Training model
    # h: number of slices in height
    # w: number of slices in weidth
    single_image_results = model(img_path)
    sliced_image_path = image_slice(img_path, cache_path, h=h, w=w)
    img_name = os.path.basename(img_path)
    # sliced_images = []
    im = Image.open(img_path)
    width, height = im.size
    block_width = int(width / w)
    block_height = int(height / h)

    slice_results = 0
    img_name = os.path.splitext(img_name)[0]
    for i in range(0, h):
        for j in range(0, w):
            # print("i: "+ str(i)+ " j: ", str(j))
            slice_name = img_name + "_h" + str(i) + "_w" + str(j) + ".jpg"
            slice_path = os.path.join(sliced_image_path, slice_name)
            # sliced_images.append(Image.open(slice_path))
            results = model(slice_path).xyxy[0]

            for result in results:
                result[0] += j * block_width
                result[1] += i * block_height
                result[2] += j * block_width
                result[3] += i * block_height

            # print(results.size())
            if results.size()[0] == 0:
                continue
            if type(slice_results) == int and slice_results == 0:
                slice_results = results
            else:
                slice_results = torch.cat((slice_results, results), 0)
    
    # Postprocess
    skip_list = []
    for index, slice_result in enumerate(slice_results):
        for res in single_image_results.xyxy[0]:
            # if xmin in slice is greater or equal to xmin in the whole image
            # and ymin in slice is greater or equal to xmin in the whole image
            # and xmax in slice is less than or equal to xmin in the whole image
            # and ymax in slice is less than or equal to xmin in the whole image
            if(
                not torch.lt(slice_result[0], res[0])
                and not torch.lt(slice_result[1], res[1])
                and not torch.gt(slice_result[2], res[2])
                and not torch.gt(slice_result[3], res[3])
                and slice_result[5] == res[5]
            ):
               skip_list.append(index)
    
    for index, slice_result in enumerate(slice_results):
        if(index not in skip_list):
            slice_result = torch.tensor(slice_result)
            slice_result = torch.reshape(slice_result, [1, 6])
            final_results = torch.cat((single_image_results.xyxy[0], slice_result), 0)

    # Save results
    save_path = os.path.join(res_path, img_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, img_name + ".txt")
    print("Result is saved at: " + save_file_path)
    np.savetxt(save_file_path, final_results.numpy())


    return final_results
