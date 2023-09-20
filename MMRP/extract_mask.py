from PIL import Image
import os
import cv2
import numpy as np
import argparse
import argparse
import shutil
import copy

temp_dir=''
k= 5

def setAlpha(img, img_name, t=1):
    # rm black pixels
    img1 = copy.deepcopy(img)
    pixdata = img1.load()
    for y in range(img1.size[1]):
        for x in range(img1.size[0]):
            if pixdata[x, y][0] == 0 and pixdata[x, y][1] == 0 and pixdata[x, y][2] == 0:
                pass
            else:
                pixdata[x, y] = (255, 255, 255)

    # mask_output_pth = setAlpha(crop_img,input_name)  # return: temp mask pth
    # mask_before_dilate = cv2.imread(mask_output_pth)
    # gray = cv2.cvtColor(mask_before_dilate, cv2.COLOR_BGR2GRAY)
    # ret, threshold = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY_INV)
    # kernel = np.ones((k, k), np.uint8)
    # image_dilate = cv2.erode(mask_before_dilate, kernel, 1)
    # mask_after_dilate = temp_dir + '/' + input_name + '_mask_dilate_' + str(k) + '.png'  # mask for filtering in-the-boarder lines
    # cv2.imwrite(mask_after_dilate, image_dilate, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    if t<0.6:
        mask_temp = temp_dir + '\\' + img_name + '_mask_t.png'
        img1.save(mask_temp)
        temp_mask_before_dilate = cv2.imread(mask_temp)
        gray = cv2.cvtColor(temp_mask_before_dilate, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((k, k), np.uint8)
        temp_mask_dilated = cv2.erode(temp_mask_before_dilate, kernel, 1)
        cv2.imwrite(mask_temp, temp_mask_dilated, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    else:
        mask_temp = temp_dir + '\\' + img_name + '_mask.png'
        img1.save(mask_temp)
        temp_mask_before_dilate = cv2.imread(mask_temp)
        gray = cv2.cvtColor(temp_mask_before_dilate, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((k, k), np.uint8)
        temp_mask_dilated = cv2.erode(temp_mask_before_dilate, kernel, 1)
        cv2.imwrite(mask_temp, temp_mask_dilated, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return mask_temp


def crop_no_black_boundary(imag_pth):
    imag = Image.open(imag_pth)
    ori_width, ori_height = imag.size
    if ori_width == 512 and ori_height == 512:
        return imag
    else:
        left = 100
        top = 100
        right = ori_width - 100
        bottom = ori_height - 100
        return imag.crop((left, top, right, bottom))

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--input_img', type=str, default=None, help='input_img')
    parse.add_argument('--known_area', type=float, default=None, help='known_area')
    parse.add_argument('--inpainting_dir', type=str, default=None, help='inpainting_dir')
    args=vars(parse.parse_args()) 

    # parse args
    input_img_path = args.get('input_img')
    inpainting_dir = args.get('inpainting_dir')
    if os.path.exists(inpainting_dir) is False: os.makedirs(inpainting_dir)
    input_img_dir = inpainting_dir + "\\input"
    if os.path.exists(input_img_dir) is False: os.makedirs(input_img_dir)
    temp_dir = inpainting_dir+"\\temp"
    if os.path.exists(temp_dir) is False: os.makedirs(temp_dir)

    # get file ext name
    input_img_name= input_img_path.split("\\")[-1]

    if os.path.exists(input_img_path) is True:
        input_name = input_img_name.split(".")[0]
        print(f"extract mask form: {input_img_path}")

        # remove file to input_img_dir
        remove_input_img_pth = input_img_dir + "\\" + input_name + "_orin.png"
        shutil.copy(input_img_path,remove_input_img_pth)
        print(f"save orin file in :{remove_input_img_pth}")

        # crop image
        crop_img = crop_no_black_boundary(input_img_path)
        crop_img.save( temp_dir + '\\' + input_name + '_rm_black_bdry.png')
        crop_img.save( input_img_dir + "\\" + input_name + ".png")
        
        known_area = args.get('known_area')
        if known_area < 0.6:
            ori_mask = setAlpha(crop_img, input_name, known_area)
            # 单边resize
            crp_im_wid, crp_im_hit = crop_img.size
            if crp_im_wid >= crp_im_hit:
                crop_img_pt = crop_img.resize(
                    (512, int(512 * crp_im_hit / crp_im_wid)))
            else:
                crop_img_pt = crop_img.resize(
                    (int(512 * crp_im_wid / crp_im_hit), 512))
            new_cp_img = Image.new('RGB', (512, 512), (0, 0, 0))
            new_cp_img.paste(crop_img_pt, (0, 0))
            crop_img = new_cp_img
            crop_img_512_pth = temp_dir + '\\' + input_name + '_512.png'
            crop_img.save(crop_img_512_pth)

        # extract original masks for in-the-boarder lines filter
        mask_output_pth = setAlpha(crop_img,input_name) 
        mask_before_dilate = cv2.imread(mask_output_pth)
        gray = cv2.cvtColor(mask_before_dilate, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((k, k), np.uint8)
        image_dilate = cv2.erode(mask_before_dilate, kernel, 1)
        mask_after_dilate = temp_dir + '\\' + input_name + '_mask_dilate_' + str(k) + '.png'  # mask for filtering in-the-boarder lines
        cv2.imwrite(mask_after_dilate, image_dilate, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        #save mask for cluster line
        mask_for_cluster_line_pth = temp_dir + '\\' + input_name + '_mask'
        if known_area < 0.6:
           mask_for_cluster_line_pth+='_t.png'
        else:
           mask_for_cluster_line_pth+='.png'

        remove_mask_for_cluster_line_pth = input_img_dir + '\\' + input_name + '_mask_cl.png'
        
        shutil.copy(mask_for_cluster_line_pth,remove_mask_for_cluster_line_pth)
        print(f'save mask for cluster line in {remove_mask_for_cluster_line_pth}')
    else:
        print(f"{input_img_path} is not Exist" )

